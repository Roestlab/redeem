use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::init::{FanInOut, NonLinearity, NormalOrUniform};
use candle_nn::{Dropout, Init, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertEncoder, Config};
use candle_nn::ops::softmax;
use std::env::var;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::utils::utils::get_tensor_stats;

#[derive(Clone)]
pub struct ModuleList {
    modules: Vec<Arc<dyn Module>>,
}

impl ModuleList {
    pub fn new() -> Self {
        Self { modules: Vec::new() }
    }

    pub fn push<M: Module + 'static>(&mut self, module: M) {
        self.modules.push(Arc::new(module));
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Module for ModuleList {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut output = xs.clone();
        for module in &self.modules {
            output = module.forward(&output)?;
        }
        Ok(output)
    }
}

impl Deref for ModuleList {
    type Target = Vec<Arc<dyn Module>>;

    fn deref(&self) -> &Self::Target {
        &self.modules
    }
}

impl DerefMut for ModuleList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.modules
    }
}

// BertEncoderModule remains the same
pub struct BertEncoderModule {
    encoder: BertEncoder,
}

impl BertEncoderModule {
    pub fn new(encoder: BertEncoder) -> Self {
        Self { encoder }
    }
}

impl Module for BertEncoderModule {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let attention_mask = hidden_states.ones_like()?;
        self.encoder.forward(hidden_states, &attention_mask)
    }
}


/// A minimal Transformer encoder layer with multi-head self-attention, feedforward block,
/// dropout, and optional sinusoidal positional encoding and padding mask support.
#[derive(Debug, Clone)]
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    pos_encoding: Tensor,
    dropout: Dropout,
    pub model_dim: usize,
}

impl TransformerEncoder {
    pub fn new(
        varbuilder: &VarBuilder,
        input_dim: usize,
        model_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        max_len: usize,
        dropout_prob: f32,
        device: &Device,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer = TransformerEncoderLayer::new(
                &varbuilder.pp(&format!("layer_{}", i)),
                model_dim,
                ff_dim,
                num_heads,
                dropout_prob,
            )?;
            layers.push(layer);
        }
        let pos_encoding = create_sinusoidal_encoding(max_len, model_dim, device)?;
        let dropout = Dropout::new(dropout_prob);
        Ok(Self { layers, pos_encoding, dropout, model_dim })
    }

    pub fn forward_with_mask(&self, x: &Tensor, padding_mask: Option<&Tensor>, training: bool) -> Result<Tensor> {
        log::trace!("[TransformerEncoder] input x shape: {:?}", x.shape());
        let (mean, min, max) = get_tensor_stats(x)?;
        log::debug!("[TransformerEncoder] input stats: mean={}, min={}, max={}", mean, min, max);
        let (b, t, _) = x.dims3()?;
        let pe = self.pos_encoding.i((..t, ..))?
            .unsqueeze(0)?
            .broadcast_as((b, t, self.pos_encoding.dim(1)?))?;

        log::trace!("[TransformerEncoder] positional encoding shape: {:?}", pe.shape());
        let (mean, min, max) = get_tensor_stats(&pe)?;
        log::debug!("[TransformerEncoder] positional encoding stats: mean={}, min={}, max={}", mean, min, max);

        let mut out = x.broadcast_add(&pe)?;
        let (mean, min, max) = get_tensor_stats(&out)?;
        log::debug!("[TransformerEncoder] after positional encoding stats: mean={}, min={}, max={}", mean, min, max);

        out = self.dropout.forward(&out, training)?;

        log::trace!("[TransformerEncoder] after dropout shape: {:?}", out.shape());
        let (mean, min, max) = get_tensor_stats(&out)?;
        log::debug!("[TransformerEncoder] after dropout stats: mean={}, min={}, max={}", mean, min, max);

        for (i, layer) in self.layers.iter().enumerate() {
            log::trace!("[TransformerEncoder] applying layer {}", i);
            out = layer.forward(&out, padding_mask, training)?;
            log::trace!("[TransformerEncoder] output shape after layer {}: {:?}", i, out.shape());
            let (mean, min, max) = get_tensor_stats(&out)?;
            log::debug!("[TransformerEncoder] output stats after layer {}: mean={}, min={}, max={}", i, mean, min, max);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout1: Dropout,
    dropout2: Dropout,
}

impl TransformerEncoderLayer {
    pub fn new(
        varbuilder: &VarBuilder,
        model_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        dropout_prob: f32,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(varbuilder, model_dim, model_dim, num_heads)?,
            ff: FeedForward::new(varbuilder, model_dim, ff_dim)?,
            norm1: candle_nn::layer_norm(
                model_dim,
                candle_nn::LayerNormConfig::default(),
                varbuilder.pp("norm1"))?,
            norm2: candle_nn::layer_norm(
                model_dim,
                candle_nn::LayerNormConfig::default(),
                varbuilder.pp("norm2"))?,
            dropout1: Dropout::new(dropout_prob),
            dropout2: Dropout::new(dropout_prob),
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, training: bool) -> Result<Tensor> {
        log::trace!("[TransformerEncoderLayer] input x shape: {:?}", x.shape());
        let attn = self.self_attn.forward(x, mask)?;
        let (mean, min, max) = get_tensor_stats(&attn)?;
        log::debug!("[TransformerEncoderLayer] attention stats: mean={}, min={}, max={}", mean, min, max);
        let tmp = self.dropout1.forward(&attn, training)?;
        let (mean, min, max) = get_tensor_stats(&tmp)?;
        log::debug!("[TransformerEncoderLayer] attention after dropout stats: mean={}, min={}, max={}", mean, min, max);
        let tmp2 = x.broadcast_add(&tmp)?;
        let (mean, min, max) = get_tensor_stats(&tmp2)?;
        log::debug!("[TransformerEncoderLayer] after residual connection stats: mean={}, min={}, max={}", mean, min, max);
        let x = self.norm1.forward(&tmp2)?;
        let (mean, min, max) = get_tensor_stats(&x)?;
        log::debug!("[TransformerEncoderLayer] after norm1 stats: mean={}, min={}, max={}", mean, min, max);
        let ff = self.ff.forward(&x)?;
        let (mean, min, max) = get_tensor_stats(&ff)?;
        log::debug!("[TransformerEncoderLayer] feedforward stats: mean={}, min={}, max={}", mean, min, max);
        let result = self.norm2.forward(&x.broadcast_add(&self.dropout2.forward(&ff, training)?)?)?;
        log::trace!("[TransformerEncoderLayer] output shape: {:?}", result.shape());
        let (mean, min, max) = get_tensor_stats(&result)?;
        log::debug!("[TransformerEncoderLayer] output stats: mean={}, min={}, max={}", mean, min, max);
        Ok(result)
    }
}


#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    proj_q: Linear,
    proj_k: Linear,
    proj_v: Linear,
    proj_out: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(
        varbuilder: &VarBuilder,
        input_dim: usize,
        model_dim: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let head_dim = model_dim / num_heads;
        Ok(Self {
            proj_q: candle_nn::linear(input_dim, model_dim, varbuilder.pp("proj_q"))?,
            proj_k: candle_nn::linear(input_dim, model_dim, varbuilder.pp("proj_k"))?,
            proj_v: candle_nn::linear(input_dim, model_dim, varbuilder.pp("proj_v"))?,
            proj_out: candle_nn::linear(model_dim, model_dim, varbuilder.pp("proj_out"))?,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        log::trace!("[MultiHeadAttention] Input shape: b={}, t={}, head_dim={} (num_heads={})", b, t, self.head_dim, self.num_heads);

        let q = self.proj_q.forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        log::trace!("[MultiHeadAttention] Q shape after projection and transpose: {:?}", q.shape());
        let (mean, min, max) = get_tensor_stats(&q)?;
        log::debug!("[MultiHeadAttention] Q stats: mean={}, min={}, max={}", mean, min, max);

        let k = self.proj_k.forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        log::trace!("[MultiHeadAttention] K shape after projection and transpose: {:?}", k.shape());
        let (mean, min, max) = get_tensor_stats(&k)?;
        log::debug!("[MultiHeadAttention] K stats: mean={}, min={}, max={}", mean, min, max);

        let v = self.proj_v.forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        log::trace!("[MultiHeadAttention] V shape after projection and transpose: {:?}", v.shape());
        let (mean, min, max) = get_tensor_stats(&v)?;
        log::debug!("[MultiHeadAttention] V stats: mean={}, min={}, max={}", mean, min, max);


        log::trace!("[MultiHeadAttention] Q/K/V shape after projection and transpose: {:?}", q.shape());

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let mut scores = q.matmul(&k_t)? / (self.head_dim as f64).sqrt();

        let mut scores = match q.matmul(&k_t) {
            Ok(s) => (s / (self.head_dim as f64).sqrt())?,
            Err(e) => {
                log::error!("[MultiHeadAttention] Failed during matmul for scores: {}", e);
                return Err(e.into());
            }
        };

        log::trace!("[MultiHeadAttention] Attention score shape: {:?}", scores.shape());
        let (mean, min, max) = get_tensor_stats(&scores)?;
        log::debug!("[MultiHeadAttention] Attention score stats: mean={}, min={}, max={}", mean, min, max);

        if let Some(mask) = mask {
            log::trace!("[MultiHeadAttention] Applying mask");
            let mask = mask.unsqueeze(1)?;
            let scale = Tensor::new(1e9f32, x.device())?;
            scores = match scores.broadcast_add(&mask.neg()?.mul(&scale)?) {
                Ok(s) => s,
                Err(e) => {
                    log::error!("[MultiHeadAttention] Failed during masking: {}", e);
                    return Err(e.into());
                }
            };
        }

        let attn = match candle_nn::ops::softmax(&scores, scores.dims().len() - 1) {
            Ok(a) => a,
            Err(e) => {
                log::error!("[MultiHeadAttention] Failed during softmax: {}", e);
                return Err(e.into());
            }
        };
        let (attn_mean, attn_min, attn_max) = get_tensor_stats(&attn)?;
        log::debug!("[MultiHeadAttention] Attention stats: mean={}, min={}, max={}", attn_mean, attn_min, attn_max);

        let context = match attn.matmul(&v) {
            Ok(ctx) => ctx.transpose(1, 2)?.reshape((b, t, self.num_heads * self.head_dim))?,
            Err(e) => {
                log::error!("[MultiHeadAttention] Failed during attention context computation: {}", e);
                return Err(e.into());
            }
        };

        log::trace!("[MultiHeadAttention] Final context shape: {:?}", context.shape());
        let (mean, min, max) = get_tensor_stats(&context)?;
        log::debug!("[MultiHeadAttention] Context stats: mean={}, min={}, max={}", mean, min, max);
        self.proj_out.forward(&context)
    }
}

#[derive(Debug, Clone)]
pub struct FeedForward {
    lin1: Linear,
    lin2: Linear,
}

impl FeedForward {
    pub fn new(varbuilder: &VarBuilder, model_dim: usize, ff_dim: usize) -> Result<Self> {
        Ok(Self {
            lin1: candle_nn::linear(model_dim, ff_dim, varbuilder.pp("lin1"))?,
            lin2: candle_nn::linear(ff_dim, model_dim, varbuilder.pp("lin2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.lin1.forward(x)?.relu()?;
        self.lin2.forward(&x)
    }
}

/// Generate sinusoidal positional encoding like in "Attention is All You Need".
pub fn create_sinusoidal_encoding(seq_len: usize, model_dim: usize, device: &Device) -> Result<Tensor> {
    let mut pe = vec![0f32; seq_len * model_dim];
    for pos in 0..seq_len {
        for i in 0..model_dim {
            let angle = pos as f32 / (10000f32).powf(2. * (i / 2) as f32 / model_dim as f32);
            pe[pos * model_dim + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    Tensor::from_vec(pe, (seq_len, model_dim), device)
}

