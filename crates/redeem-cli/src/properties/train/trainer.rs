use anyhow::{Context, Result};
use maud::{PreEscaped, html};
use redeem_properties::models::model_interface::ModelInterface;
use redeem_properties::models::rt_model::load_retention_time_model;
use redeem_properties::models::{
    ccs_cnn_lstm_model::CCSCNNLSTMModel, ccs_cnn_tf_model::CCSCNNTFModel,
    rt_cnn_lstm_model::RTCNNLSTMModel, rt_cnn_transformer_model::RTCNNTFModel,
};
use redeem_properties::utils::data_handling::{PeptideData, TargetNormalization};
use redeem_properties::utils::peptdeep_utils::load_modifications;
use redeem_properties::utils::utils::get_device;
use report_builder::{
    Report, ReportSection,
    plots::plot_scatter,
};

use crate::properties::load_data;
use crate::properties::train::plot::{plot_losses, plot_training_metric};
use crate::properties::train::sample_peptides;
use crate::properties::util::write_bytes_to_file;
use input::PropertyTrainConfig;
use load_data::load_peptide_data;

use super::input;

pub fn run_training(config: &PropertyTrainConfig) -> Result<()> {
    log::trace!("Loading modifications map");
    let modifications = load_modifications().context("Failed to load modifications")?;

    // Load training data
    let (train_peptides, norm_factor) = load_peptide_data(
        &config.train_data,
        &config.model_arch,
        Some(config.nce),
        Some(config.instrument.clone()),
        Some(config.rt_normalization.clone().unwrap()),
        &modifications,
    )?;
    log::info!("Loaded {} training peptides", train_peptides.len());

    // Load validation data if specified
    let (val_peptides, _val_norm_factor) = if let Some(ref val_path) = config.validation_data {
        let (peptides, norm) = load_peptide_data(
            val_path,
            &config.model_arch,
            Some(config.nce),
            Some(config.instrument.clone()),
            Some(config.rt_normalization.clone().unwrap()),
            &modifications,
        )
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
    log::trace!(
        "Loading model architecture: {} on device: {:?}",
        model_arch,
        device
    );

    let mut model: Box<dyn ModelInterface + Send + Sync> = match &config.checkpoint_file {
        Some(checkpoint_path) => {
            log::info!("Loading model from checkpoint: {}", checkpoint_path);
            match config.model_arch.as_str() {
                "rt_cnn_lstm" => Box::new(RTCNNLSTMModel::new(
                    checkpoint_path,
                    None,
                    0,
                    8,
                    4,
                    true,
                    device.clone(),
                )?),
                "rt_cnn_tf" => Box::new(RTCNNTFModel::new(
                    checkpoint_path,
                    None,
                    0,
                    8,
                    4,
                    true,
                    device.clone(),
                )?),
                "ccs_cnn_lstm" => Box::new(CCSCNNLSTMModel::new(
                    checkpoint_path,
                    None,
                    0,
                    8,
                    4,
                    true,
                    device.clone(),
                )?),
                "ccs_cnn_tf" => Box::new(CCSCNNTFModel::new(
                    checkpoint_path,
                    None,
                    0,
                    8,
                    4,
                    true,
                    device.clone(),
                )?),
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported model architecture: {}",
                        config.model_arch
                    ));
                }
            }
        }
        None => match config.model_arch.as_str() {
            "rt_cnn_lstm" => Box::new(RTCNNLSTMModel::new_untrained(device.clone())?),
            "rt_cnn_tf" => Box::new(RTCNNTFModel::new_untrained(device.clone())?),
            "ccs_cnn_lstm" => Box::new(CCSCNNLSTMModel::new_untrained(device.clone())?),
            "ccs_cnn_tf" => Box::new(CCSCNNTFModel::new_untrained(device.clone())?),
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported model architecture: {}",
                    config.model_arch
                ));
            }
        },
    };

    log::trace!("Model loaded successfully");

    let start_time = std::time::Instant::now();
    log::trace!("Training started");
    let train_step_metrics = model.train(
        &train_peptides,
        val_peptides.as_ref(),
        modifications.clone(),
        config.batch_size,
        config.validation_batch_size.unwrap_or(config.batch_size),
        config.learning_rate as f64,
        config.epochs,
        config.early_stopping_patience,
    ).with_context(|| "Training failed: an error occurred during the model training process")?;
    log::info!("Training completed in {:?}", start_time.elapsed());
    model.save(&config.output_file)?;
    log::info!("Model saved to: {}", config.output_file);

    // Generate report
    let mut report = Report::new(
        "ReDeeM",
        &config.version,
        Some("https://github.com/singjc/redeem/blob/master/img/redeem_logo.png?raw=true"),
        &format!("ReDeeM {:?} Trainer Report", config.model_arch),
    );

    /* Section 1: Overview */
    {
        let mut overview_section = ReportSection::new("Overview");

        overview_section.add_content(html! {
            "This report summarizes the training process of the ReDeeM model. It includes epoch-level summaries and step-wise dynamics such as learning rate scheduling and accuracy tracking over time. These plots provide insight into model convergence behavior and training stability."
        });

        let epoch_losses = train_step_metrics.summarize_loss_for_plotting();
        let losses_plot = plot_losses(&epoch_losses);
        overview_section.add_plot(losses_plot);

        // Step-wise learning rate plot
        let lr_plot = plot_training_metric(
            &train_step_metrics,
            "lr",
            "Learning Rate Over Steps",
            "Step",
            "Learning Rate",
        );
        overview_section.add_plot(lr_plot);

        // Step-wise loss plot
        let step_loss_plot = plot_training_metric(
            &train_step_metrics,
            "loss",
            "Loss Over Steps",
            "Step",
            "Loss",
        );
        overview_section.add_plot(step_loss_plot);

        // Step-wise accuracy plot
        let acc_plot = plot_training_metric(
            &train_step_metrics,
            "accuracy",
            "Accuracy Over Steps",
            "Step",
            "Accuracy",
        );
        overview_section.add_plot(acc_plot);

        // Inference scatter plot
        let val_peptides: Vec<PeptideData> = sample_peptides(&val_peptides.as_ref().unwrap(), 5000);
        let inference_results: Vec<PeptideData> =
            model.inference(&val_peptides, config.batch_size, modifications, norm_factor)?;
        let (true_rt, pred_rt): (Vec<f64>, Vec<f64>) = val_peptides
            .iter()
            .zip(&inference_results)
            .filter_map(|(true_pep, pred_pep)| {
                match (true_pep.retention_time, pred_pep.retention_time) {
                    (Some(t), Some(p)) => {
                        let t_denorm = match norm_factor {
                            TargetNormalization::ZScore(mean, std) => t as f64 * std as f64 + mean as f64,
                            TargetNormalization::MinMax(min, range) => t as f64 * range as f64 + min as f64,
                            TargetNormalization::None => t as f64,
                        };
                        Some((t_denorm, p as f64))
                    }
                    _ => None,
                }
            })
            .unzip();
        

        let scatter_plot = plot_scatter(
            &vec![true_rt.clone()],
            &vec![pred_rt.clone()],
            vec!["Prediction".to_string()],
            "Predicted vs True (Random 1000 Validation Peptides)",
            "Target",
            "Predicted",
        )
        .unwrap();
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

    // Save configuration to JSON file
    let path = "redeem_trainer_config.json";
    let json = serde_json::to_string_pretty(&config)?;
    println!("{}", json);
    let bytes = serde_json::to_vec_pretty(&config)?;
    write_bytes_to_file(path, &bytes)?;

    Ok(())
}
