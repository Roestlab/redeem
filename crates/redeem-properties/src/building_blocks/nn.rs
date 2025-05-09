use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{Dropout, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertEncoder, Config};
use candle_nn::ops::softmax;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

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
                input_dim,
                model_dim,
                ff_dim,
                num_heads,
                dropout_prob,
            )?;
            layers.push(layer);
        }
        let pos_encoding = create_sinusoidal_encoding(max_len, model_dim, device)?;
        let dropout = Dropout::new(dropout_prob);
        Ok(Self { layers, pos_encoding, dropout })
    }

    pub fn forward_with_mask(&self, x: &Tensor, padding_mask: Option<&Tensor>, training: bool) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let pe = self.pos_encoding.i((..t, ..))?.unsqueeze(0)?.broadcast_as((b, t, self.pos_encoding.dim(1)?))?;
        let mut out = x + pe;
        out = self.dropout.forward(&out?, training);
        for layer in &self.layers {
            out = layer.forward(&out?, padding_mask, training);
        }
        Ok(out?)
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
        input_dim: usize,
        model_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        dropout_prob: f32,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(varbuilder, input_dim, model_dim, num_heads)?,
            ff: FeedForward::new(varbuilder, model_dim, ff_dim)?,
            norm1: {
                let weight = varbuilder.get((model_dim,), "norm1.weight")?;
                let bias = varbuilder.get((model_dim,), "norm1.bias")?;
                LayerNorm::new(weight, bias, 1e-5)
            },            
            norm2: {
                let weight = varbuilder.get((model_dim,), "norm2.weight")?;
                let bias = varbuilder.get((model_dim,), "norm2.bias")?;
                LayerNorm::new(weight, bias, 1e-5)
            },            
            dropout1: Dropout::new(dropout_prob),
            dropout2: Dropout::new(dropout_prob),
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, training: bool) -> Result<Tensor> {
        let attn = self.self_attn.forward(x, mask)?;
        let x = self.norm1.forward(&(x + self.dropout1.forward(&attn, training)?)?)?;
        let ff = self.ff.forward(&x)?;
        self.norm2.forward(&(x + self.dropout2.forward(&ff, training)?)?)
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
            proj_q: linear_from_varbuilder(varbuilder, input_dim, model_dim, "proj_q")?,
            proj_k: linear_from_varbuilder(varbuilder, input_dim, model_dim, "proj_k")?,
            proj_v: linear_from_varbuilder(varbuilder, input_dim, model_dim, "proj_v")?,
            proj_out: linear_from_varbuilder(varbuilder, input_dim, model_dim, "proj_out")?,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = self.proj_q.forward(x)?.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.proj_k.forward(x)?.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.proj_v.forward(x)?.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let mut scores = q.matmul(&k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt();
        if let Some(mask) = mask {
            let mask = mask.unsqueeze(1)?;
            let scale = Tensor::new(1e9f32, x.device())?;
            scores = scores?.broadcast_add(&mask.neg()?.mul(&scale)?);
        }

        let scores = scores?; 
        let attn = candle_nn::ops::softmax(&scores, scores.dims().len() - 1)?;
        let context = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, t, self.num_heads * self.head_dim))?;
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
            lin1: linear_from_varbuilder(varbuilder, model_dim, ff_dim, "lin1")?,
            lin2: linear_from_varbuilder(varbuilder, ff_dim, model_dim, "lin2")?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.lin1.forward(x)?.relu()?;
        self.lin2.forward(&x)
    }
}


fn linear_from_varbuilder(
    vb: &VarBuilder,
    in_dim: usize,
    out_dim: usize,
    prefix: &str,
) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), &format!("{}.weight", prefix))?;
    let bias = vb.get((out_dim,), &format!("{}.bias", prefix)).ok();
    Ok(Linear::new(weight, bias))
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
