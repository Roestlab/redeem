use anyhow::{Context, Result};
use candle_core::Device;
use redeem_properties::{
    models::{
        model_interface::{ModelInterface, PredictionResult},
        ccs_cnn_lstm_model::CCSCNNLSTMModel,
    },
    utils::{data_handling::PeptideData, peptdeep_utils::{load_modifications, ccs_to_mobility_bruker, ion_mobility_to_ccs_bruker}},
};
use std::path::PathBuf;

struct PredictionContext {
    peptides: Vec<String>,
    mods: Vec<String>,
    mod_sites: Vec<String>,
    charges: Vec<i32>,
    observed_ccs: Vec<f32>,
}

impl PredictionContext {
    fn new(test_peptides: &Vec<(&str, &str, &str, i32, f32)>) -> Self {
        let peptides: Vec<String> = test_peptides.iter().map(|(pep, _, _, _, _)| pep.to_string()).collect();
        let mods: Vec<String> = test_peptides.iter().map(|(_, mod_, _, _, _)| mod_.to_string()).collect();
        let mod_sites: Vec<String> = test_peptides.iter().map(|(_, _, sites, _, _)| sites.to_string()).collect();
        let charges: Vec<i32> = test_peptides.iter().map(|(_, _, _, charge, _)| *charge).collect();
        let observed_ccs: Vec<f32> = test_peptides.iter().map(|(_, _, _, _, ccs)| *ccs).collect();

        Self {
            peptides,
            mods,
            mod_sites,
            charges,
            observed_ccs,
        }
    }
}

fn run_prediction(model: &mut CCSCNNLSTMModel, prediction_context: &PredictionContext) -> Result<()> { 
    match model.predict(
        &prediction_context.peptides,
        &prediction_context.mods,
        &prediction_context.mod_sites,
        Some(prediction_context.charges.clone()), 
        None,
        None,
    ) {
        Ok(predictions) => {
            if let PredictionResult::IMResult(ccs_preds) = predictions {  
                let total_error: f32 = ccs_preds
                    .iter()
                    .zip(prediction_context.observed_ccs.iter())
                    .map(|(pred, obs)| (pred - obs).abs())
                    .sum();

                print_predictions(&prediction_context.peptides, &ccs_preds, &prediction_context.observed_ccs); 

                let mean_absolute_error = total_error / ccs_preds.len() as f32;
                println!("Mean Absolute Error: {:.6}", mean_absolute_error);
            } else {
                println!("Unexpected prediction result type.");
            }
        }
        Err(e) => {
            println!("Error during batch prediction: {:?}", e);
        }
    }
    Ok(())
}

fn print_predictions(peptides: &[String], ccs_preds: &[f32], observed_ccs: &[f32]) { // Changed
    let mut peptides_iter = peptides.iter();
    let mut ccs_preds_iter = ccs_preds.iter(); // Changed
    let mut observed_ccs_iter = observed_ccs.iter(); // Changed

    loop {
        match (peptides_iter.next(), ccs_preds_iter.next(), observed_ccs_iter.next()) {
            (Some(pep), Some(pred), Some(obs)) => {
                println!("Peptide: {}, Predicted CCS: {}, Observed CCS: {}", pep, pred, obs); // Changed
            }
            _ => break, // Exit the loop if any iterator is exhausted
        }
    }
}

fn main() -> Result<()> {
    let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
    let constants_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");

    // let device use cuda if available otherwise use cpu
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    println!("Device: {:?}", device);

    let mut model = CCSCNNLSTMModel::new(&model_path, Some(&constants_path), 0, 8, 4, true, device)
        .context("Failed to create CCSCNNLSTMModel")?;

    // Define training data
    let training_data = vec![
        PeptideData::new("EHVIIQAEFYLNPDQ", Some(2), None, None, None, Some(1.10), None),
        PeptideData::new("KTLTGKTITLEVEPS", Some(2), None, None, None, Some(1.04), None),
        PeptideData::new("SLLAQNTSWLL", Some(1), None, None, None, Some(1.67), None),
        PeptideData::new("SLQEVAM[+15.9949]FL", Some(1), None, None, None, Some(1.53), None),
        PeptideData::new("VLADQVWTL", Some(2), None, None, None, Some(0.839), None),
        PeptideData::new("LLMEPGAMRFL", Some(2), None, None, None, Some(0.949), None),
        PeptideData::new("SGEIKIAYTYSVS", Some(2), None, None, None, Some(0.974), None),
        PeptideData::new("HTEIVFARTSPQQKL", Some(2), None, None, None, Some(1.13), None),
        PeptideData::new("SM[+15.9949]ADIPLGFGV", Some(1), None, None, None, Some(1.59), None),
        PeptideData::new("KLIDHQGLYL", Some(2), None, None, None, Some(0.937), None),
    ];

    // Sequence	Monoisotopic Mass (Da)	Charge	m/z
    // SKEEETSIDVAGKP	1488.7308	2	745.3727
    // LPILVPSAKKAIYM	1542.9208	2	772.4677
    // RTPKIQVYSRHPAE	1680.906	3	561.3093
    // EEVQIDILDTAGQE	1558.7362	2	780.3754
    // GAPLVKPLPVNPTDPA	1584.8875	2	793.4511
    // FEDENFILK	1153.5655	2	577.7901
    // YPSLPAQQV	1001.5182	1	1002.5255
    // YLPPATQVV	986.5437	2	494.2792
    // YISPDQLADLYK	1424.7187	2	713.3667
    // PSIVRLLQCDPSSAGQF	1816.9142	2	909.4644

    let test_peptides = vec![
        ("SKEEETSIDVAGKP", "", "", 2, ion_mobility_to_ccs_bruker(0.998, 2, 745.3727)),
        ("LPILVPSAKKAIYM", "", "", 2, ion_mobility_to_ccs_bruker(1.12, 2, 772.4677)),
        ("RTPKIQVYSRHPAE", "", "", 3, ion_mobility_to_ccs_bruker(0.838, 3, 561.3093)),
        ("EEVQIDILDTAGQE", "", "", 2, ion_mobility_to_ccs_bruker(1.02, 2, 780.3754)),
        ("GAPLVKPLPVNPTDPA", "", "", 2, ion_mobility_to_ccs_bruker(1.01, 2, 793.4511)),
        ("FEDENFILK", "", "", 2, ion_mobility_to_ccs_bruker(0.897, 2, 577.7901)),
        ("YPSLPAQQV", "", "", 1, ion_mobility_to_ccs_bruker(1.45, 1, 1002.5255)),
        ("YLPPATQVV", "", "", 2, ion_mobility_to_ccs_bruker(0.846, 2, 494.2792)),
        ("YISPDQLADLYK", "", "", 2, ion_mobility_to_ccs_bruker(0.979, 2, 713.3667)),
        ("PSIVRLLQCDPSSAGQF", "", "", 2, ion_mobility_to_ccs_bruker(1.10, 2, 909.4644)),
    ];

    let prediction_context = PredictionContext::new(&test_peptides);

    run_prediction(&mut model, &prediction_context)?;

    // Fine-tune the model
    let modifications = load_modifications().context("Failed to load modifications")?;
    let learning_rate = 0.001;
    let epochs = 5;
    model
        .fine_tune(&training_data, modifications, 10, learning_rate, epochs)
        .context("Failed to fine-tune the model")?;

    // Test prediction again with a few peptides after fine-tuning
    run_prediction(&mut model, &prediction_context)?;

    Ok(())
}
