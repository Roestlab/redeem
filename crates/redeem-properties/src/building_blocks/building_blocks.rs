use anyhow::{Context, Result as AnyHowResult};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn as nn;
use candle_transformers as transformers;
use core::num;
use std::fmt;

use crate::building_blocks::nn::{BertEncoderModule, ModuleList};
use crate::building_blocks::sequential::{Sequential, seq};
use crate::building_blocks::bilstm::BidirectionalLSTM;
use crate::building_blocks::featurize::aa_one_hot;

/// constants used by PeptDeep Models
pub const MOD_FEATURE_SIZE: usize = 109; // TODO: derive from constants yaml
pub const AA_EMBEDDING_SIZE: usize = 27; // TODO: derive from constants yaml
const MAX_INSTRUMENT_NUM: usize = 8; // TODO: derive from constants yaml

#[derive(Clone)]
pub struct DecoderLinear {
    nn: Sequential,
}

impl DecoderLinear {
    pub fn new(in_features: usize, out_features: usize, vb: &nn::VarBuilder) -> Result<Self> {
        let weight = Tensor::zeros((in_features, 64), DType::F32, vb.device())?;
        let bias = Tensor::zeros(64, DType::F32, vb.device())?;

        let linear1 = nn::Linear::new(weight, Some(bias));
        let prelu = nn::PReLU::new(Tensor::zeros(64, DType::F32, vb.device())?, false);

        let weight = Tensor::zeros((64, out_features), DType::F32, vb.device())?;
        let bias = Tensor::zeros(64, DType::F32, vb.device())?;

        let linear2 = nn::Linear::new(weight, Some(bias));

        let mut nn = seq();
        nn = nn.add(linear1);
        nn = nn.add(prelu);
        nn = nn.add(linear2);

        Ok(Self { nn })
    }

    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        in_features: usize,
        out_features: usize,
        names: Vec<&str>,
        names_bias: Vec<&str>,
    ) -> Result<Self> {
        let linear1 = nn::Linear::new(
            varstore.get((64, in_features), names[0]).unwrap(),
            Some(varstore.get(64, names_bias[0]).unwrap()),
        );
        let prelu = nn::PReLU::new(varstore.get(1, names[1]).unwrap(), true);
        let linear2 = nn::Linear::new(
            varstore.get((out_features, 64), names[2]).unwrap(),
            Some(varstore.get(out_features, names_bias[1]).unwrap()),
        );

        let mut nn = seq();
        nn = nn.add(linear1);
        nn = nn.add(prelu);
        nn = nn.add(linear2);

        Ok(Self { nn })
    }
}

impl Module for DecoderLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        println!("DecoderLinear forward");
        println!("x shape: {:?}", x.shape());
        self.nn.forward(x)
    }
}

impl fmt::Debug for DecoderLinear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecoderLinear")
            .field("nn", &"<Sequential>")
            .finish()
    }
    
}

#[derive(Debug, Clone)]
struct AAEmbedding {
    embeddings: nn::Embedding,
}

impl AAEmbedding {
    fn new(hidden_size: usize, device: &Device) -> Result<Self> {
        // Create a VarBuilder
        let vb = nn::VarBuilder::zeros(DType::F32, device);

        // Create the embedding layer
        let embeddings = nn::embedding(AA_EMBEDDING_SIZE, hidden_size, vb.pp("embedding"))?;

        Ok(Self { embeddings })
    }

    fn from_varstore(varstore: &nn::VarBuilder, hidden_size: usize, name: &str) -> Result<Self> {
        let weight = varstore.get((AA_EMBEDDING_SIZE, hidden_size), name);
        let embeddings = nn::Embedding::new(weight.unwrap(), hidden_size);
        Ok(Self { embeddings })
    }
}

impl Module for AAEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        println!("AAEmbedding forward");
        println!("x shape: {:?}", x.shape());
        println!("x first 5 elements: {:?}", x.narrow(1, 0, 5)?);
        self.embeddings.forward(&x.to_dtype(DType::I64)?)
    }
}

#[derive(Debug, Clone)]
struct PositionalEncoding {
    pe: Tensor,
}

impl PositionalEncoding {
    fn new(out_features: usize, max_len: usize, device: &Device) -> Result<Self> {
        let position = Tensor::arange(0, max_len as u32, device)?.unsqueeze(1)?;

        let scalar = Tensor::new(-2.0 * std::f32::consts::PI / out_features as f32, device)?;
        let div_term = Tensor::arange(0f32, (out_features / 2) as f32, device)?
            .mul(&scalar)?
            .exp()?;

        let pe = Tensor::zeros((1, max_len, out_features), DType::F32, device)?;

        let sin_term = position
            .to_dtype(DType::F32)?
            .matmul(&div_term.unsqueeze(0)?)?
            .sin()?;
        let cos_term = position
            .to_dtype(DType::F32)?
            .matmul(&div_term.unsqueeze(0)?)?
            .cos()?;

        // Assign sine terms
        for i in 0..(out_features / 2) {
            pe.slice_assign(
                &[0..1, 0..max_len, (2 * i)..(2 * i + 1)],
                &sin_term.narrow(1, i, 1)?,
            )?;
        }

        // Assign cosine terms
        for i in 0..(out_features / 2) {
            pe.slice_assign(
                &[0..1, 0..max_len, (2 * i + 1)..(2 * i + 2)],
                &cos_term.narrow(1, i, 1)?,
            )?;
        }

        Ok(Self { pe })
    }

    fn from_varstore(
        varstore: &nn::VarBuilder,
        out_features: usize,
        max_len: usize,
        name: &str,
    ) -> Result<Self> {
        let pe = varstore.get((1, max_len, out_features), name);
        Ok(Self { pe: pe.unwrap() })
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let pe_slice = self.pe.narrow(1, 0, seq_len)?;
        x.add(&pe_slice)
    }
}

#[derive(Debug, Clone)]
struct ModEmbeddingFixFirstK {
    k: usize,
    nn: nn::Linear,
}

impl ModEmbeddingFixFirstK {
    fn new(mod_feature_size: usize, out_features: usize, device: &Device) -> Result<Self> {
        let k = 6;
        let vb = nn::VarBuilder::zeros(DType::F32, device);
        let nn = nn::linear(mod_feature_size - k, out_features - k, vb.pp("linear"))?;
        Ok(Self { k, nn })
    }

    fn from_varstore(
        var_store: &nn::VarBuilder,
        mod_feature_size: usize,
        out_features: usize,
        name: &str,
    ) -> Result<Self> {
        let k = 6;
        let weight = var_store.get((out_features - k, MOD_FEATURE_SIZE - k), name);
        let nn = nn::Linear::new(weight.unwrap(), None);
        Ok(Self { k, nn })
    }
}

impl Module for ModEmbeddingFixFirstK {
    fn forward(&self, mod_x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = mod_x.dims3()?;

        // Split the input tensor
        let first_k = mod_x.narrow(D::Minus1, 0, self.k)?;
        let rest = mod_x.narrow(D::Minus1, self.k, mod_x.dim(D::Minus1)? - self.k)?;

        // Apply the linear transformation to the rest
        let transformed = self.nn.forward(&rest)?;

        // Concatenate the results
        Tensor::cat(&[&first_k, &transformed], D::Minus1)
    }
}

#[derive(Debug, Clone)]
pub struct Input26aaModPositionalEncoding {
    mod_nn: ModEmbeddingFixFirstK,
    aa_emb: AAEmbedding,
    pos_encoder: PositionalEncoding,
}

impl Input26aaModPositionalEncoding {
    fn new(out_features: usize, max_len: usize, device: &Device) -> Result<Self> {
        let mod_hidden = 8;
        let mod_nn = ModEmbeddingFixFirstK::new(MOD_FEATURE_SIZE, mod_hidden, device)?;
        let aa_emb = AAEmbedding::new(out_features - mod_hidden, device)?;
        let pos_encoder = PositionalEncoding::new(out_features, max_len, device)?;

        Ok(Self {
            mod_nn,
            aa_emb,
            pos_encoder,
        })
    }

    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        out_features: usize,
        max_len: usize,
        names: Vec<&str>,
    ) -> Result<Self> {
        let mod_hidden = 8;
        Ok(Self {
            mod_nn: ModEmbeddingFixFirstK::from_varstore(
                varstore,
                MOD_FEATURE_SIZE,
                mod_hidden,
                names[0],
            )?,
            aa_emb: AAEmbedding::from_varstore(varstore, out_features - mod_hidden, names[1])?,
            pos_encoder: PositionalEncoding::from_varstore(
                varstore,
                out_features,
                max_len,
                names[2],
            )?,
        })
    }

    pub fn forward(&self, aa_indices: &Tensor, mod_x: &Tensor) -> Result<Tensor> {
        println!("Input26aaModPositionalEncoding forward");
        println!("passing through mod_nn");
        let mod_x = self.mod_nn.forward(mod_x)?;
        println!("passing through aa_emb");
        let x = self.aa_emb.forward(aa_indices)?;

        // Concatenate x and mod_x along the last dimension
        println!("Concatenating x and mod_x");
        println!("x shape: {:?}", x.shape());
        println!("mod_x shape: {:?}", mod_x.shape());
        let concatenated = Tensor::cat(&[&x, &mod_x], 2)?;
        println!("passing through pos_encoder");
        self.pos_encoder.forward(&concatenated)
    }
}

#[derive(Debug, Clone)]
pub struct MetaEmbedding {
    nn: nn::Linear,
}

impl MetaEmbedding {
    fn new(out_features: usize, device: &Device) -> Result<Self> {
        let nn = nn::linear(
            MAX_INSTRUMENT_NUM + 1,
            out_features - 1,
            nn::VarBuilder::zeros(DType::F32, device).pp("linear"),
        )?;
        Ok(Self { nn })
    }

    pub fn from_varstore(
        varstore: &nn::VarBuilder,
        out_features: usize,
        name: Vec<&str>,
    ) -> Result<Self> {
        let weight = varstore.get((out_features - 1, MAX_INSTRUMENT_NUM + 1), name[0]);
        let bias = varstore.get(out_features - 1, name[1]);
        let nn = nn::Linear::new(weight.unwrap(), Some(bias.unwrap()));
        Ok(Self { nn })
    }

    fn one_hot(&self, indices: &Tensor, num_classes: usize) -> AnyHowResult<Tensor> {
        let batch_size = indices.dim(0)?;

        let mut one_hot_data = vec![0.0f32; batch_size * num_classes];

        for i in 0..batch_size {
            let index = indices.get(i)?.to_scalar::<i64>()?;
            let class_idx = index as usize;

            if class_idx < num_classes {
                one_hot_data[i * num_classes + class_idx] = 1.0;
            } else {
                return Err(anyhow::anyhow!(
                    "Index {} out of bounds for one-hot encoding",
                    class_idx
                ));
            }
        }

        Tensor::from_slice(&one_hot_data, (batch_size, num_classes), indices.device())
            .context("Failed to create tensor from one-hot data")
    }

    pub fn forward(
        &self,
        charges: &Tensor,
        nces: &Tensor,
        instrument_indices: &Tensor,
    ) -> Result<Tensor> {
        // One-hot encode the instrument indices
        let inst_x = self.one_hot(&instrument_indices.to_dtype(DType::I64)?, MAX_INSTRUMENT_NUM).unwrap(); 

        // Concatenate the one-hot encoded instrument indices with NCEs
        let combined_input = Tensor::cat(&[&inst_x, nces], 1)?;

        // Pass through the linear layer
        let meta_x = self.nn.forward(&combined_input)?;

        // Concatenate the output with charges
        let meta_x = Tensor::cat(&[&meta_x, charges], 1)?;

        Ok(meta_x)
    }
}

// struct BertEncoder {
//     layers: Vec<transformers::models::bert::BertLayer>,
//     span: tracing::Span,
// }

// impl BertEncoder {
//     fn load(vb: VarBuilder, config: &Config, prefix: &str) -> Result<Self> {
//         let layers = (0..config.num_hidden_layers)
//             .map(|index| transformers::models::bert::BertLayer::load(vb.pp(format!("{prefix}layer.{index}")), config))
//             .collect::<Result<Vec<_>>>()?;
//         let span = tracing::span!(tracing::Level::TRACE, "encoder");
//         Ok(BertEncoder { layers, span })
//     }

//     fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
//         let _enter = self.span.enter();
//         let mut hidden_states = hidden_states.clone();
//         // Use a loop rather than a fold as it's easier to modify when adding debug/...
//         for layer in self.layers.iter() {
//             hidden_states = layer.forward(&hidden_states, attention_mask)?
//         }
//         Ok(hidden_states)
//     }
// }

#[derive(Clone)]
pub struct HiddenHfaceTransformer {
    pub bert: transformers::models::bert::BertEncoder,
    pub config: transformers::models::bert::Config,
}

impl HiddenHfaceTransformer {
    fn new(
        hidden_dim: usize,
        hidden_expand: usize,
        nheads: usize,
        nlayers: usize,
        dropout: f64,
        output_attentions: bool,
        device: &Device,
    ) -> Result<Self> {
        unimplemented!()
    }

    pub fn from_varstore(
        varstore: nn::VarBuilder,
        hidden_dim: usize,
        hidden_expand: usize,
        nheads: usize,
        nlayers: usize,
        dropout: f64,
        output_attentions: bool,
    ) -> Result<Self> {
        let config = transformers::models::bert::Config {
            hidden_size: hidden_dim,
            intermediate_size: hidden_dim * hidden_expand,
            num_attention_heads: nheads,
            num_hidden_layers: nlayers,
            hidden_dropout_prob: dropout,
            ..Default::default()
        };

        let bert = transformers::models::bert::BertEncoder::load(varstore, &config).unwrap();

        Ok(Self { bert, config })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Determine batch size and sequence length from input tensor
        let (batch_size, seq_len, _) = x.shape().dims3()?; // Assuming x has shape [batch_size, seq_len, hidden_dim]
    
        // Create or adjust the mask
        let mask = match mask {
            Some(m) => {
                // Unsqueeze and repeat to match attention scores shape
                m.unsqueeze(1)?
                    .repeat(vec![1, self.config.num_attention_heads as usize, 1])? // Shape: [batch_size, n_heads, seq_len]
                    .unsqueeze(2)? // Add an extra dimension for broadcasting
            },
            None => {
                // Create a new mask filled with zeros
                let zeros_mask = Tensor::zeros((batch_size, 1, seq_len), DType::F32, x.device())?; // Shape: [batch_size, 1, seq_len]
                zeros_mask.repeat(vec![1, self.config.num_attention_heads as usize, 1])? // Shape: [batch_size, n_heads, seq_len]
                    .unsqueeze(2)? // Add an extra dimension for broadcasting
            }
        };
    
        println!("HiddenHfaceTransformer forward");
        println!("x shape: {:?}", x.shape());
        println!("mask shape: {:?}", mask.shape());
    
        // Forward pass through BERT encoder
        self.bert.forward(x, &mask)
    }
    
    
    
    
}

impl fmt::Debug for HiddenHfaceTransformer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HiddenHfaceTransformer")
            .field("bert_encoder", &"<BertEncoder>")
            .finish()
    }
}

#[derive(Clone)]
pub struct ModLossNN {
    modules: ModuleList,
}

impl ModLossNN {
    fn new() -> Self {
        unimplemented!();
    }

    pub fn from_varstore(
        varstore: nn::VarBuilder,
        hidden_dim: usize,
        hidden_expand: usize,
        nheads: usize,
        nlayers: usize,
        dropout: f64,
        output_attentions: bool,
        decoder_linear_output_dim: usize,
        bert_name: &str,
        decoder_linear_names: Vec<&str>,
        decoder_linear_bias_names: Vec<&str>,
    ) -> Result<Self> {
        let config = transformers::models::bert::Config {
            hidden_size: hidden_dim,
            intermediate_size: hidden_dim * hidden_expand,
            num_attention_heads: nheads,
            num_hidden_layers: nlayers,
            hidden_dropout_prob: dropout,
            ..Default::default()
        };

        let bert =
            transformers::models::bert::BertEncoder::load(varstore.clone().pp(bert_name), &config).unwrap();
        let bert_module = BertEncoderModule::new(bert);

        let mut modules = ModuleList::new();
        modules.push(bert_module);

        let decoder_linear = DecoderLinear::from_varstore(
            &varstore,
            hidden_dim,
            decoder_linear_output_dim,
            decoder_linear_names,
            decoder_linear_bias_names,
        )
        .unwrap();
        modules.push(decoder_linear);

        Ok(Self { modules })
    }
}

impl Module for ModLossNN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut output = xs.clone();
        for module in self.modules.iter() {
            output = module.forward(&output)?;
        }
        Ok(output)
    }
}

impl fmt::Debug for ModLossNN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModLossNN")
            .field("modules[0]", &"<Hidden_HFace_Transformer>")
            .field("modules[1]", &"<DecoderLinear>")
            .finish()
    }
    
}

#[derive(Debug, Clone)]
struct SeqCNN {
    cnn_short: nn::Conv1d,
    cnn_medium: nn::Conv1d,
    cnn_long: nn::Conv1d,
}

impl SeqCNN {
    fn new() -> Self {
        unimplemented!();
    }

    pub fn from_varstore( 
        varstore: nn::VarBuilder, 
        embedding_hidden: usize, 
        names_weight: Vec<&str>,
        names_bias: Vec<&str>, 
    ) ->  Result<Self> {
            let cnn_short = nn::Conv1d::new(
                varstore.get((embedding_hidden, embedding_hidden, 3), names_weight[0]).unwrap(),
                Some(varstore.get(embedding_hidden, names_bias[0]).unwrap()),
                nn::Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                }
            );

            let cnn_medium = nn::Conv1d::new(
                varstore.get((embedding_hidden, embedding_hidden, 5), names_weight[1]).unwrap(),
                Some(varstore.get(embedding_hidden, names_bias[1]).unwrap()),
                nn::Conv1dConfig {
                    padding: 2,
                    ..Default::default()
                }
            );

            let cnn_long = nn::Conv1d::new(
                varstore.get((embedding_hidden, embedding_hidden, 7), names_weight[2]).unwrap(),
                Some(varstore.get(embedding_hidden, names_bias[2]).unwrap()),
                nn::Conv1dConfig {
                    padding: 3,
                    ..Default::default()
                }
            );
        
            Ok(Self { cnn_short, cnn_medium, cnn_long })
    }
}

impl Module for SeqCNN {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?;

        let short = self.cnn_short.forward(&x)?;
        let medium = self.cnn_medium.forward(&x)?;
        let long = self.cnn_long.forward(&x)?;

        let output = Tensor::cat(&[x, short, medium, long], 1)?;
        Ok(output.transpose(1, 2)?)
    }
}

#[derive(Debug, Clone)]
struct SeqLSTM{
    lstm: BidirectionalLSTM
}

impl SeqLSTM {
    fn new() -> Self {
        unimplemented!();
    }

    pub fn from_varstore(
        varstore: nn::VarBuilder,
        input_size: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let lstm = BidirectionalLSTM::new(
            input_size,
            hidden_dim,
            num_layers,
            &varstore
        )?;
        Ok(Self { lstm })
    }
}

impl Module for SeqLSTM {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.lstm.forward(&x)?)
    }
}

#[derive(Debug, Clone)]
struct SeqAttentionSum {
    attention: nn::Linear,
}

impl SeqAttentionSum {
    pub fn from_varstore(
        varstore: nn::VarBuilder,
        hidden_dim: usize,
        name: &str,
    ) -> Result<Self> {
        let attention = nn::Linear::new(
            varstore.get((1, hidden_dim), name).unwrap(), 
            None,
        );
        Ok(Self { attention })
    }
}

impl Module for SeqAttentionSum {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attention_weights = self.attention.forward(x)?;
        
        // Apply softmax to normalize weights
        // TODO: This is done in the model itself in the PyTorch implementation
        let attention_weights = nn::ops::softmax(&attention_weights, 1)?;

        let output = x.mul(&attention_weights)?;
        
        Ok(output.sum(1)?)
    }
}

#[derive(Debug, Clone)]
pub struct Encoder26aaModChargeCnnLstmAttnSum{
    mod_nn: ModEmbeddingFixFirstK,
    input_cnn: SeqCNN,
    input_lstm: SeqLSTM,
    attn_sum: SeqAttentionSum,
}

impl Encoder26aaModChargeCnnLstmAttnSum {
    fn new() -> Self {
        unimplemented!();
    }

    pub fn from_varstore(
        varstore: nn::VarBuilder,
        mod_hidden_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        names_mod_nn: Vec<&str>,
        names_input_cnn_weight: Vec<&str>,
        names_input_cnn_bias: Vec<&str>,
        lstm_pp: &str,
        names_attn_sum: Vec<&str>,
    ) -> Result<Self> {
        let input_dim = AA_EMBEDDING_SIZE + mod_hidden_dim + 1;
        Ok(Self {
            mod_nn: ModEmbeddingFixFirstK::from_varstore(
                &varstore,
                MOD_FEATURE_SIZE,
                mod_hidden_dim,
                names_mod_nn[0],
            )?,
            input_cnn: SeqCNN::from_varstore(
                varstore.clone(),
                input_dim,
                names_input_cnn_weight,
                names_input_cnn_bias,
            )?,
            input_lstm: SeqLSTM::from_varstore(
                varstore.pp(lstm_pp).clone(),
                input_dim * 4,
                hidden_dim,
                num_layers,
            )?,
            attn_sum: SeqAttentionSum::from_varstore(
                varstore.clone(),
                hidden_dim * 2,
                names_attn_sum[0],
            )?,
        })
    }

    pub fn forward(
        &self,
        aa_indices: &Tensor,
        mod_x: &Tensor,
        charges: &Tensor
    ) -> AnyHowResult<Tensor> {
        println!("Encoder26aaModChargeCnnLstmAttnSum forward");

        let mod_x = self.mod_nn.forward(mod_x)?;

        let charges_repeated = charges.unsqueeze(1)?.repeat(&[1, mod_x.dim(1)?, 1])?;
        let additional_tensors: Vec<&Tensor> = vec![&mod_x, &charges_repeated];
        let x = aa_one_hot(&aa_indices, &additional_tensors)?;

        let x = self.input_cnn.forward(&x)?;
        let x = self.input_lstm.forward(&x)?;
        let x = self.attn_sum.forward(&x)?;
        Ok(x)
    }
    
}

#[cfg(test)]
mod tests {
    use crate::model_interface::ModelInterface;
    use crate::models::rt_cnn_lstm_model::RTCNNLSTMModel;
    use crate::utils::peptdeep_utils::load_modifications;
    use candle_core::Device;
    use candle_nn::VarBuilder;
    use std::path::PathBuf;
    use std::time::Instant;
    // use itertools::izip;

    use super::*;

    #[test]
    fn test_decoder_linear_new() -> Result<()> {
        // Set up the device and random seed
        let device = Device::Cpu;

        // Create an instance of DecoderLinear
        let in_features = 10;
        let out_features = 5;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let decoder_linear = DecoderLinear::new(in_features, out_features, &vb.pp("output_nn"))?;

        // Create the input tensor
        let x = Tensor::new(
            &[
                [
                    0.0698, -1.6635, -0.7651, 0.7928, 0.2882, -1.1787, -2.1401, -0.4903, 0.6598,
                    -0.0703,
                ],
                [
                    0.2745, 0.5569, -0.2322, -0.0597, 0.1211, 0.2984, 1.3448, 1.4614, 1.0566,
                    -0.5461,
                ],
                [
                    -0.2178, -0.2809, -0.3869, -0.3572, -1.1568, -1.7660, -2.5380, -0.0334,
                    -1.7017, 0.5863,
                ],
            ],
            &device,
        )?
        .to_dtype(DType::F32)?;

        // Perform forward pass
        let output = decoder_linear.forward(&x)?;

        println!("Output:\n{}", output);

        // Expected output from PyTorch
        let expected_output = Tensor::new(
            &[
                [-0.3355, 0.0190, 0.3605, -0.6334, -0.4229],
                [0.0960, 0.1447, 0.1348, -0.1506, 0.0620],
                [-0.4250, 0.2105, -0.0994, -0.5364, -0.6821],
            ],
            &device,
        )?
        .to_dtype(DType::F32)?;

        // Check output shape
        assert_eq!(
            output.shape(),
            expected_output.shape(),
            "Output shape mismatch"
        );

        // Check output values
        let output_vec = output.to_vec2::<f32>()?;
        let expected_vec = expected_output.to_vec2::<f32>()?;

        for (row_out, row_exp) in output_vec.iter().zip(expected_vec.iter()) {
            for (val_out, val_exp) in row_out.iter().zip(row_exp.iter()) {
                assert!(
                    (val_out - val_exp).abs() < 1e-4,
                    "Output value {:.4} doesn't match expected {:.4}",
                    val_out,
                    val_exp
                );
            }
        }

        // Print shapes for verification
        println!("Input shape: {:?}", x.shape());
        println!("Output shape: {:?}", output.shape());
        println!("Output:\n{}", output);

        Ok(())
    }

    #[test]
    fn test_decoder_linear() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let decoder_linear = DecoderLinear::from_varstore(
            &var_store,
            256,
            4,
            vec!["output_nn.nn.0.weight", "output_nn.nn.1.weight", "output_nn.nn.2.weight"],
            vec!["output_nn.nn.0.bias", "output_nn.nn.2.bias"],
        )
        .unwrap();

        println!("decoder_linear : {:?}", decoder_linear);
    }

    #[test]
    fn test_mod_embedding_fix_first_k() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let mod_nn = ModEmbeddingFixFirstK::from_varstore(
            &var_store,
            MOD_FEATURE_SIZE,
            8,
            "input_nn.mod_nn.nn.weight",
        )
        .unwrap();

        println!("mod_nn : {:?}", mod_nn);
    }

    #[test]
    fn test_aa_embedding() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let aa_emb = AAEmbedding::from_varstore(&var_store, 256, "input_nn.aa_emb.weight").unwrap();

        println!("aa_emb : {:?}", aa_emb);
    }

    #[test]
    fn test_positional_encoding() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let pos_enc =
            PositionalEncoding::from_varstore(&var_store, 248, 200, "input_nn.pos_encoder.pe")
                .unwrap();

        println!("pos_enc : {:?}", pos_enc);
    }

    #[test]
    fn test_input_26aa_mod_positional_encoding() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let input_26aa_mod_pos_enc = Input26aaModPositionalEncoding::from_varstore(
            &var_store,
            256 - 8,
            200,
            vec![
                "input_nn.mod_nn.nn.weight",
                "input_nn.aa_emb.weight",
                "input_nn.pos_encoder.pe",
            ],
        )
        .unwrap();

        println!("input_26aa_mod_pos_enc : {:?}", input_26aa_mod_pos_enc);
    }

    #[test]
    fn test_meta_embedding() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let meta_emb = MetaEmbedding::from_varstore(
            &var_store,
            8,
            vec!["meta_nn.nn.weight", "meta_nn.nn.bias"],
        )
        .unwrap();

        println!("meta_emb : {:?}", meta_emb);
    }

    #[test]
    fn test_hidden_hface_transformer() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let hidden_hface_transformer = HiddenHfaceTransformer::from_varstore(
            var_store.pp("hidden_nn.bert"),
            256,
            4,
            8,
            4,
            0.1,
            false,
        )
        .unwrap();

        println!("hidden_hface_transformer : {:?}", hidden_hface_transformer);
    }

    #[test]
    fn test_mod_loss_nn(){
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let mod_loss_nn = ModLossNN::from_varstore(
            var_store,
            256,
            4,
            8,
            1,
            0.1,
            false,
            4,
            "modloss_nn.0.bert",
            vec!["modloss_nn.1.nn.0.weight", "modloss_nn.1.nn.1.weight", "modloss_nn.1.nn.2.weight"],
            vec!["modloss_nn.1.nn.0.bias", "modloss_nn.1.nn.2.bias"],
        )
        .unwrap();

        println!("mod_loss_nn : {:?}", mod_loss_nn);
    }

    #[test]
    fn test_seq_cnn(){
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let seq_cnn = SeqCNN::from_varstore(
            var_store,
            36,
            vec!["ccs_encoder.input_cnn.cnn_short.weight", "ccs_encoder.input_cnn.cnn_medium.weight", "ccs_encoder.input_cnn.cnn_long.weight"],
            vec!["ccs_encoder.input_cnn.cnn_short.bias", "ccs_encoder.input_cnn.cnn_medium.bias", "ccs_encoder.input_cnn.cnn_long.bias"],
        )
        .unwrap();

        println!("seq_cnn : {:?}", seq_cnn);
    }

    #[test]
    fn test_seq_lstm(){
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let seq_lstm = SeqLSTM::from_varstore(
            var_store.pp("ccs_encoder.hidden_nn"),
            144,
            128,
            2
        )
        .unwrap();

        println!("seq_lstm : {:?}", seq_lstm);
    }

    #[test]
    fn test_seq_attention_sum(){
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let var_store =
            VarBuilder::from_pth(model_path, candle_core::DType::F32, &Device::Cpu).unwrap();

        let seq_attention_sum = SeqAttentionSum::from_varstore(
            var_store,
            256,
            "ccs_encoder.attn_sum.attn.0.weight"
        )
        .unwrap();

        println!("seq_attention_sum : {:?}", seq_attention_sum);
    }

}
