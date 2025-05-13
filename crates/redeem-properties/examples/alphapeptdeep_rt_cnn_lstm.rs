use anyhow::{Context, Result};
use candle_core::Device;
use redeem_properties::{
    models::{
        model_interface::{ModelInterface, PredictionResult},
        rt_cnn_lstm_model::RTCNNLSTMModel,
    },
    utils::{
        data_handling::{PeptideData, PeptideBatchData},
        peptdeep_utils::{load_modifications, remove_mass_shift, get_modification_string, get_modification_indices},
    },
};
use std::path::PathBuf;
use std::sync::Arc;

struct PredictionContext {
    peptides: Vec<&'static str>,
    mods: Vec<&'static str>,
    mod_sites: Vec<&'static str>,
    observed_rts: Vec<f32>,
}

impl PredictionContext {
    fn new(test_peptides: &[(&'static str, &'static str, &'static str, f32)]) -> Self {
        let peptides = test_peptides.iter().map(|(pep, _, _, _)| *pep).collect();
        let mods = test_peptides.iter().map(|(_, m, _, _)| *m).collect();
        let mod_sites = test_peptides.iter().map(|(_, _, site, _)| *site).collect();
        let observed_rts = test_peptides.iter().map(|(_, _, _, rt)| *rt).collect();

        Self {
            peptides,
            mods,
            mod_sites,
            observed_rts,
        }
    }
}

fn run_prediction(model: &mut RTCNNLSTMModel, context: &PredictionContext) -> Result<()> {
    match model.predict(
        &context.peptides,
        &context.mods,
        &context.mod_sites,
        None,
        None,
        None,
    ) {
        Ok(preds) => {
            if let PredictionResult::RTResult(rt_preds) = preds {
                let total_error: f32 = rt_preds
                    .iter()
                    .zip(&context.observed_rts)
                    .map(|(p, o)| (p - o).abs())
                    .sum();

                for ((pep, pred), obs) in context.peptides.iter().zip(rt_preds.iter()).zip(&context.observed_rts) {
                    println!("Peptide: {}, Predicted RT: {:.6}, Observed RT: {:.6}", pep, pred, obs);
                }

                println!(
                    "Mean Absolute Error: {:.6}",
                    total_error / rt_preds.len() as f32
                );
            }
        }
        Err(e) => println!("Prediction error: {e}"),
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    let model_path = PathBuf::from("/home/singjc/Documents/github/redeem/rt_fine_tuned.safetensors");
    let constants_path = PathBuf::from("/home/singjc/Documents/github/redeem/crates/redeem-properties/data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    println!("Device: {:?}", device);

    let mut model = RTCNNLSTMModel::new(&model_path, Some(&constants_path), 0, 8, 4, true, device)
        .context("Failed to create RTCNNLSTMModel")?;

    let modifications = load_modifications().context("Failed to load modifications")?;

    let training_data: Vec<PeptideData> = vec![
        "AKPLMELIER",
        "TEM[+15.9949]VTISDASQR",
        "AGKFPSLLTHNENMVAK",
        "LSELDDRADALQAGASQFETSAAK",
        "FLLQDTVELR",
        "SVTEQGAELSNEER",
        "EHALLAYTLGVK",
        "TVQSLEIDLDSM[+15.9949]R",
        "VVSQYSSLLSPMSVNAVM[+15.9949]K",
        "TFLALINQVFPAEEDSKK",
    ]
    .into_iter()
    .enumerate()
    .map(|(i, seq)| {
        let naked = remove_mass_shift(seq);
        let mods = get_modification_string(seq, &modifications);
        let sites = get_modification_indices(seq);
        PeptideData::new(
            seq,
            &naked,
            &mods,
            &sites,
            None,
            None,
            None,
            None,
            Some(i as f32 / 10.0),
            None,
            None,
            None,
        )
    })
    .collect();

    let test_peptides_data = vec![
        ("QPYAVSELAGHQTSAESWGTGR", "", "", 0.4328955),
        ("GMSVSDLADKLSTDDLNSLIAHAHR", "Oxidation@M", "1", 0.6536107),
        ("TVQHHVLFTDNMVLICR", "Oxidation@M;Carbamidomethyl@C", "11;15", 0.7811949),
        ("EAELDVNEELDKK", "", "", 0.2934583),
        ("YTPVQQGPVGVNVTYGGDPIPK", "", "", 0.5863009),
    ];

    let prediction_context = PredictionContext::new(&test_peptides_data);

    run_prediction(&mut model, &prediction_context)?;

    model.fine_tune(&training_data, modifications, 10, 0.001, 5)?;

    run_prediction(&mut model, &prediction_context)?;

    model.save("alphapeptdeep_rt_cnn_lstm_finetuned.safetensors")?;

    Ok(())
}
