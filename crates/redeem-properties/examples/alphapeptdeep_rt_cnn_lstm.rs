use anyhow::{Context, Result};
use candle_core::Device;
use redeem_properties::{
    models::{
        model_interface::{ModelInterface, PredictionResult},
        rt_cnn_lstm_model::RTCNNLSTMModel,
    },
    utils::{data_handling::PeptideData, peptdeep_utils::load_modifications},
};
use std::path::PathBuf;

struct PredictionContext {
    peptides: Vec<String>,
    mods: Vec<String>,
    mod_sites: Vec<String>,
    observed_rts: Vec<f32>,
}

impl PredictionContext {
    fn new(test_peptides: &Vec<(&str, &str, &str, f32)>) -> Self {
        let peptides: Vec<String> = test_peptides
            .iter()
            .map(|(pep, _, _, _)| pep.to_string())
            .collect();
        let mods: Vec<String> = test_peptides
            .iter()
            .map(|(_, mod_, _, _)| mod_.to_string())
            .collect();
        let mod_sites: Vec<String> = test_peptides
            .iter()
            .map(|(_, _, sites, _)| sites.to_string())
            .collect();
        let observed_rts: Vec<f32> = test_peptides.iter().map(|(_, _, _, rt)| *rt).collect();

        Self {
            peptides,
            mods,
            mod_sites,
            observed_rts,
        }
    }
}

fn run_prediction(
    model: &mut RTCNNLSTMModel,
    prediction_context: &PredictionContext,
) -> Result<()> {
    match model.predict(
        &prediction_context.peptides,
        &prediction_context.mods,
        &prediction_context.mod_sites,
        None,
        None,
        None,
    ) {
        Ok(predictions) => {
            if let PredictionResult::RTResult(rt_preds) = predictions {
                let total_error: f32 = rt_preds
                    .iter()
                    .zip(prediction_context.observed_rts.iter())
                    .map(|(pred, obs)| (pred - obs).abs())
                    .sum();

                print_predictions(
                    &prediction_context.peptides,
                    &rt_preds,
                    &prediction_context.observed_rts,
                );

                let mean_absolute_error = total_error / rt_preds.len() as f32;
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

fn print_predictions(peptides: &[String], rt_preds: &[f32], observed_rts: &[f32]) {
    let mut peptides_iter = peptides.iter();
    let mut rt_preds_iter = rt_preds.iter();
    let mut observed_rts_iter = observed_rts.iter();

    loop {
        match (
            peptides_iter.next(),
            rt_preds_iter.next(),
            observed_rts_iter.next(),
        ) {
            (Some(pep), Some(pred), Some(obs)) => {
                println!(
                    "Peptide: {}, Predicted RT: {}, Observed RT: {}",
                    pep, pred, obs
                );
            }
            _ => break, // Exit the loop if any iterator is exhausted
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    // let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
    let model_path = PathBuf::from("/home/singjc/Documents/github/redeem/rt_fine_tuned.safetensors");
    let constants_path = PathBuf::from("/home/singjc/Documents/github/redeem/crates/redeem-properties/data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");

    // let device use cuda if available otherwise use cpu
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    println!("Device: {:?}", device);

    let mut model = RTCNNLSTMModel::new(&model_path, &constants_path, 0, 8, 4, true, device)
        .context("Failed to create RTCNNLSTMModel")?;

    // Define training data
    let training_data: Vec<PeptideData> = vec![
        PeptideData::new("AKPLMELIER", None, None, None, Some(0.4231399), None, None),
        PeptideData::new(
            "TEM[+15.9949]VTISDASQR",
            None,
            None,
            None,
            Some(0.2192762),
            None,
            None,
        ),
        PeptideData::new(
            "AGKFPSLLTHNENMVAK",
            None,
            None,
            None,
            Some(0.3343900),
            None,
            None,
        ),
        PeptideData::new(
            "LSELDDRADALQAGASQFETSAAK",
            None,
            None,
            None,
            Some(0.5286755),
            None,
            None,
        ),
        PeptideData::new("FLLQDTVELR", None, None, None, Some(0.6522490), None, None),
        PeptideData::new(
            "SVTEQGAELSNEER",
            None,
            None,
            None,
            Some(0.2388270),
            None,
            None,
        ),
        PeptideData::new(
            "EHALLAYTLGVK",
            None,
            None,
            None,
            Some(0.5360210),
            None,
            None,
        ),
        PeptideData::new(
            "TVQSLEIDLDSM[+15.9949]R",
            None,
            None,
            None,
            Some(0.5787880),
            None,
            None,
        ),
        PeptideData::new(
            "VVSQYSSLLSPMSVNAVM[+15.9949]K",
            None,
            None,
            None,
            Some(0.6726230),
            None,
            None,
        ),
        PeptideData::new(
            "TFLALINQVFPAEEDSKK",
            None,
            None,
            None,
            Some(0.8345350),
            None,
            None,
        ),
    ];

    // Test prediction data
    let test_peptides_data = vec![
        ("QPYAVSELAGHQTSAESWGTGR", "", "", 0.4328955),
        ("GMSVSDLADKLSTDDLNSLIAHAHR", "Oxidation@M", "1", 0.6536107),
        (
            "TVQHHVLFTDNMVLICR",
            "Oxidation@M;Carbamidomethyl@C",
            "11;15",
            0.7811949,
        ),
        ("EAELDVNEELDKK", "", "", 0.2934583),
        ("YTPVQQGPVGVNVTYGGDPIPK", "", "", 0.5863009),
        ("YYAIDFTLDEIK", "", "", 0.8048359),
        ("VSSLQAEPLPR", "", "", 0.3201348),
        (
            "NHAVVCQGCHNAIDPEVQR",
            "Carbamidomethyl@C;Carbamidomethyl@C",
            "5;8",
            0.1730425,
        ),
        ("IPNIYAIGDVVAGPMLAHK", "", "", 0.8220097),
        ("AELGIPLEEVPPEEINYLTR", "", "", 0.8956433),
        ("NESTPPSEELELDKWK", "", "", 0.4471560),
        ("SIQEIQELDKDDESLR", "", "", 0.4157068),
        ("EMEENFAVEAANYQDTIGR", "Oxidation@M", "1", 0.6388353),
        ("MDSFDEDLARPSGLLAQER", "Oxidation@M", "0", 0.5593624),
        ("SLLTEADAGHTEFTDEVYQNESR", "", "", 0.5538696),
        ("NQDLAPNSAEQASILSLVTK", "", "", 0.7682227),
        ("GKVEEVELPVEK", "", "", 0.2943246),
        ("IYVASVHQDLSDDDIK", "", "", 0.3847130),
        ("IKGDMDISVPK", "", "", 0.2844255),
        ("IIPVLLEHGLER", "", "", 0.5619017),
        ("AGYTDKVVIGMDVAASEFFR", "", "", 0.8972052),
        ("TDYNASVSVPDSSGPER", "", "", 0.3279318),
        ("DLKPQNLLINTEGAIK", "", "", 0.6046495),
        ("VAEAIAASFGSFADFK", "", "", 0.8935943),
        ("AMVSNAQLDNEK", "Oxidation@M", "1", 0.1724159),
        ("THINIVVIGHVDSGK", "", "", 0.4865058),
        ("LILPHVDIQLK", "", "", 0.6268850),
        ("LIAPVAEEEATVPNNK", "", "", 0.4162872),
        ("FTASAGIQVVGDDLTVTNPK", "", "", 0.7251064),
        ("HEDLKDMLEFPAQELR", "", "", 0.6529368),
        ("LLPDFLLER", "", "", 0.7852863),
    ];

    let prediction_context = PredictionContext::new(&test_peptides_data);

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

    model.save("alphapeptdeep_rt_cnn_lstm_finetuned.safetensors")?;

    Ok(())
}
