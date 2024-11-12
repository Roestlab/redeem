use candle_core::{Result, Tensor};
use candle_nn::Module;
use std::ops::{Deref, DerefMut};
use candle_transformers::models::bert::{BertEncoder, Config};
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
