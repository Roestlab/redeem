use anyhow::{Context, Result};
use maud::{PreEscaped, html};
use redeem_properties::models::ccs_cnn_lstm_model::CCSCNNLSTMModel;
use redeem_properties::models::ccs_cnn_tf_model::CCSCNNTFModel;
use redeem_properties::models::model_interface::ModelInterface;
use redeem_properties::models::rt_cnn_lstm_model::RTCNNLSTMModel;
use redeem_properties::utils::data_handling::{PeptideData, TargetNormalization};
use redeem_properties::utils::peptdeep_utils::{load_modifications, MODIFICATION_MAP};
use redeem_properties::utils::utils::get_device;
use report_builder::{
    Report, ReportSection,
    plots::plot_scatter,
};

use crate::properties::inference::input::PropertyInferenceConfig;
use crate::properties::inference::output::write_peptide_data;
use crate::properties::train::sample_peptides;
use crate::properties::load_data::load_peptide_data;
use crate::properties::util::write_bytes_to_file;

pub fn run_inference(config: &PropertyInferenceConfig) -> Result<()> {
    let modifications = load_modifications().context("Failed to load modifications")?;

    // Load inference data
    let (inference_data, norm_factor) = load_peptide_data(
        &config.inference_data,
        &config.model_arch,
        Some(config.nce),
        Some(config.instrument.clone()),
        Some(config.normalization.clone().unwrap()),
        &modifications,
    )?;
    log::info!("Loaded {} peptides", inference_data.len());

    // Dispatch model training based on architecture
    let model_arch = config.model_arch.as_str();
    let device = get_device(&config.device)?;

    let mut model: Box<dyn ModelInterface + Send + Sync> = match model_arch {
        "rt_cnn_lstm" => Box::new(RTCNNLSTMModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        "rt_cnn_tf" => Box::new(RTCNNLSTMModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        "ccs_cnn_lstm" => Box::new(CCSCNNLSTMModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        "ccs_cnn_tf" => Box::new(CCSCNNTFModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported RT model architecture: {}",
                model_arch
            ));
        }
    };

    let start_time = std::time::Instant::now();
    model.set_evaluation_mode();
    let inference_results: Vec<PeptideData> = model.inference(
        &inference_data,
        config.batch_size,
        modifications,
        norm_factor,
    )?;
    log::info!("Inference completed in {:?}", start_time.elapsed());

    log::info!("Predictions saved to: {}", config.output_file);
    write_peptide_data(&inference_results, &config.output_file)?;

    // Generate report
    let mut report = Report::new(
        "ReDeeM",
        &config.version,
        Some("https://github.com/singjc/redeem/blob/master/img/redeem_logo.png?raw=true"),
        &format!("ReDeeM {:?} Inference Report", config.model_arch),
    );

    /* Section 1: Overview */
    {
        let mut overview_section = ReportSection::new("Overview");

        overview_section.add_content(html! {
            "This report summarizes the inference process of the ReDeeM model."
        });

        let modifications = MODIFICATION_MAP.clone();

        let normalize_field = if config.model_arch.contains("ccs") {
            "ccs"
        } else {
            "retention time"
        };

        // Inference scatter plot
        let inference_data_sampled: Vec<PeptideData> = sample_peptides(&inference_data, 5000);

        let (true_rt, pred_rt): (Vec<f64>, Vec<f64>) = inference_data_sampled
            .iter()
            .zip(&inference_results)
            .filter_map(|(true_pep, pred_pep)| {
                match normalize_field {
                    "ccs" => {
                        match (true_pep.ccs, pred_pep.ccs) {
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
                    },
                    _ => {
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
                }
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
    let path = "redeem_inference_report.html";
    report.save_to_file(&path.to_string())?;

    let path = "redeem_inference_config.json";
    let json = serde_json::to_string_pretty(&config)?;
    println!("{}", json);
    let bytes = serde_json::to_vec_pretty(&config)?;
    write_bytes_to_file(path, &bytes)?;

    Ok(())
}
