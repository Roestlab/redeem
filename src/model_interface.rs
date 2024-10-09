// In model_interface.rs
use anyhow::Result;
use std::path::Path;
use candle_core::{Device, Tensor};

pub trait ModelInterface {
    fn new<P: AsRef<Path>>(model_path: P, constants_path: P, device: Device) -> Result<Self> where Self: Sized;
    fn predict(&self, peptide_sequence: &[String], mods: &str, mod_sites: &str) -> Result<Vec<f32>>;
    fn encode_peptides(&self, peptide_sequences: &[String], mods: &str, mod_sites: &str) -> Result<Tensor>;
    fn set_evaluation_mode(&mut self);
    fn set_training_mode(&mut self);
    fn print_summary(&self);
    fn print_weights(&self);
}
