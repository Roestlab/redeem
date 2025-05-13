use anyhow::{Context, Result};
use candle_core::Device;
use redeem_properties::{
    models::{
        ccs_cnn_lstm_model::CCSCNNLSTMModel,
        model_interface::{ModelInterface, PredictionResult},
    },
    utils::{
        data_handling::{PeptideData},
        peptdeep_utils::{ion_mobility_to_ccs_bruker},
    },
};
use std::{path::PathBuf, sync::Arc};

struct PredictionContext<'a> {
    peptides: Vec<&'a str>,
    mods: Vec<&'a str>,
    mod_sites: Vec<&'a str>,
    charges: Vec<i32>,
    observed_ccs: Vec<f32>,
}

impl<'a> PredictionContext<'a> {
    fn new(test_peptides: &'a [(&'a str, &'a str, &'a str, i32, f32)]) -> Self {
        let peptides = test_peptides.iter().map(|(pep, _, _, _, _)| *pep).collect();
        let mods = test_peptides.iter().map(|(_, m, _, _, _)| *m).collect();
        let mod_sites = test_peptides.iter().map(|(_, _, s, _, _)| *s).collect();
        let charges = test_peptides.iter().map(|(_, _, _, c, _)| *c).collect();
        let observed_ccs = test_peptides.iter().map(|(_, _, _, _, ccs)| *ccs).collect();

        Self {
            peptides,
            mods,
            mod_sites,
            charges,
            observed_ccs,
        }
    }
}

fn run_prediction(model: &mut CCSCNNLSTMModel, ctx: &PredictionContext) -> Result<()> {
    match model.predict(
        &ctx.peptides,
        &ctx.mods,
        &ctx.mod_sites,
        Some(ctx.charges.clone()),
        None,
        None,
    )? {
        PredictionResult::CCSResult(preds) => {
            let total_error: f32 = preds
                .iter()
                .zip(ctx.observed_ccs.iter())
                .map(|(pred, obs)| (pred - obs).abs())
                .sum();

            for (pep, pred, obs) in itertools::izip!(&ctx.peptides, &preds, &ctx.observed_ccs) {
                println!("Peptide: {}, Predicted CCS: {:.4}, Observed CCS: {:.4}", pep, pred, obs);
            }

            let mae = total_error / preds.len() as f32;
            println!("Mean Absolute Error: {:.6}", mae);
        }
        _ => println!("Unexpected prediction result type."),
    }
    Ok(())
}

fn main() -> Result<()> {
    let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
    let constants_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    println!("Device: {:?}", device);

    let mut model = CCSCNNLSTMModel::new(
        &model_path,
        Some(&constants_path),
        0,
        8,
        4,
        true,
        device,
    )?;

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

    let ctx = PredictionContext::new(&test_peptides);
    run_prediction(&mut model, &ctx)?;

    Ok(())
}
