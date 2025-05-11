use anyhow::{Context, Result};
use redeem_properties::models::model_interface::ModelInterface;
use redeem_properties::models::rt_model::load_retention_time_model;
use redeem_properties::models::{rt_cnn_lstm_model::RTCNNLSTMModel, rt_cnn_transformer_model::RTCNNTFModel};
use redeem_properties::utils::data_handling::PeptideData;
use redeem_properties::utils::peptdeep_utils::load_modifications;
use redeem_properties::utils::utils::get_device;
use report_builder::{
    plots::{plot_boxplot, plot_pp, plot_scatter, plot_score_histogram},
    Report, ReportSection,
};
use maud::{html, PreEscaped};

use input::PropertyTrainConfig;
use load_data::load_peptide_data;
use crate::properties::load_data;
use crate::properties::train::plot::plot_losses;
use crate::properties::train::sample_peptides;
use crate::properties::util::write_bytes_to_file;

use super::input;

pub fn run_training(config: &PropertyTrainConfig) -> Result<()> {

    // Load training data
    let (train_peptides, norm_factor) = load_peptide_data(&config.train_data, Some(config.nce), Some(config.instrument.clone()), true)?;
    log::info!("Loaded {} training peptides", train_peptides.len());

    // Load validation data if specified
    let (val_peptides, _val_norm_factor) = if let Some(ref val_path) = config.validation_data {
        let (peptides, norm) = load_peptide_data(val_path, Some(config.nce), Some(config.instrument.clone()), true)
            .context("Failed to load validation data")?;
        (Some(peptides), Some(norm))
    } else {
        (None, None)
    };
    

    if let Some(ref val_data) = val_peptides {
        log::info!("Loaded {} validation peptides", val_data.len());
    } else {
        log::warn!("No validation data provided.");
    }

    // Dispatch model training based on architecture
    let model_arch = config.model_arch.as_str();
    let device = get_device(&config.device)?;
    log::trace!("Loading model architecture: {} on device: {:?}", model_arch, device);

    let mut model: Box<dyn ModelInterface + Send + Sync> = match &config.checkpoint_file {
        Some(checkpoint_path) => {
            log::info!("Loading model from checkpoint: {}", checkpoint_path);
            match config.model_arch.as_str() {
                "rt_cnn_lstm" => Box::new(RTCNNLSTMModel::new(checkpoint_path, None, 0, 8, 4, true, device.clone())?),
                "rt_cnn_tf" => Box::new(RTCNNTFModel::new(checkpoint_path, None, 0, 8, 4, true, device.clone())?),
                _ => return Err(anyhow::anyhow!("Unsupported model architecture: {}", config.model_arch)),
            }
        }
        None => {
            match config.model_arch.as_str() {
                "rt_cnn_lstm" => Box::new(RTCNNLSTMModel::new_untrained(device.clone())?),
                "rt_cnn_tf" => Box::new(RTCNNTFModel::new_untrained(device.clone())?),
                _ => return Err(anyhow::anyhow!("Unsupported model architecture: {}", config.model_arch)),
            }
        }
    };
    
    
    log::trace!("Model loaded successfully");
    

    log::trace!("Loading modifications map");
    let modifications = load_modifications().context("Failed to load modifications")?;

    let start_time = std::time::Instant::now();
    log::trace!("Training started");
    let epoch_losses = model.train(
        &train_peptides,
        val_peptides.as_ref(),
        modifications.clone(),
        config.batch_size,
        config.validation_batch_size.unwrap_or(config.batch_size),
        config.learning_rate as f64,
        config.epochs,
        config.early_stopping_patience,
    )?;
    log::info!("Training completed in {:?}", start_time.elapsed());

    // Generate report
    let mut report = Report::new(
        "ReDeeM",
        &config.version,
        Some("https://github.com/singjc/redeem/blob/master/img/redeem_logo.png?raw=true"),
        "ReDeeM Trainer Report",
    );

    /* Section 1: Overview */
    {
        let mut overview_section = ReportSection::new("Overview");

        overview_section.add_content(html! {
            "This report summarizes the training process of the ReDeeM model."
        });

        let losses_plot = plot_losses(&epoch_losses);
        overview_section.add_plot(losses_plot);

        // Lets perform inference on 1000 random samples from the validation set
        let val_peptides: Vec<PeptideData> = sample_peptides(&val_peptides.as_ref().unwrap(), 1000);
        let inference_results: Vec<PeptideData> = model.inference(
            &val_peptides,
            config.batch_size,
            modifications,
            norm_factor,
        )?;
        let (true_rt, pred_rt): (Vec<f64>, Vec<f64>) = val_peptides
            .iter()
            .zip(&inference_results)
            .filter_map(|(true_pep, pred_pep)| {
                match (true_pep.retention_time, pred_pep.retention_time) {
                    (Some(t), Some(p)) => {
                        let t_denorm = t as f64 * norm_factor.unwrap().1 as f64 + norm_factor.unwrap().0 as f64;  // de-normalized true RT
                        Some((t_denorm, p as f64))  // assume predicted is already de-normalized
                    },
                    _ => None,
                }
            })
            .unzip();


        let scatter_plot = plot_scatter(
            &vec![true_rt.clone()],
            &vec![pred_rt.clone()],
            vec!["RT Prediction".to_string()],
            "Predicted vs True RT",
            "Target RT",
            "Predicted RT"
        ).unwrap();
        overview_section.add_plot(scatter_plot);
        report.add_section(overview_section);    
    }

    /* Section 2: Configuration */
    {
        let mut config_section = ReportSection::new("Configuration");
        config_section.add_content(html! {
            style {
                ".code-container {
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-family: monospace;
                    white-space: pre-wrap;
                }"
            }
            div class="code-container" {
                pre {
                    code { (PreEscaped(serde_json::to_string_pretty(&config)?)) }
                }
            }
        });
        report.add_section(config_section);
    }

    // Save the report to HTML file
    let path = "redeem_trainer_report.html";
    report.save_to_file(&path.to_string())?;

    model.save(&config.output_file)?;
    log::info!("Model saved to: {}", config.output_file);

    let path = "redeem_trainer_config.json";
    let json = serde_json::to_string_pretty(&config)?;
    println!("{}", json);
    let bytes = serde_json::to_vec_pretty(&config)?;
    write_bytes_to_file(path, &bytes)?;

    Ok(())
}
