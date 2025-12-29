use anyhow::Result;
use clap::{Arg, ArgMatches, Command, ValueHint};
use log::LevelFilter;
use std::path::PathBuf;

use redeem_cli::classifiers::score::score::score_pin_with_config;
use redeem_cli::properties::inference::inference;
use redeem_cli::properties::inference::input::PropertyInferenceConfig;
use redeem_cli::properties::train::input::PropertyTrainConfig;
use redeem_cli::properties::train::trainer;

fn main() -> Result<()> {
    env_logger::Builder::default()
        .filter_level(LevelFilter::Error)
        .parse_env(env_logger::Env::default().filter_or("REDEEM_LOG", "error,redeem=info"))
        .init();

    let matches = Command::new("redeem")
        .version(clap::crate_version!())
        .author("Justin Sing <justincsing@gmail.com>")
        .about("\u{1F9EA} ReDeeM CLI - Modular Deep Learning Tools for Proteomics")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("properties")
                .about("Train or run peptide property prediction models")
                .subcommand(
                    Command::new("train")
                        .about("Train a new property prediction model from scratch")
                        .arg(
                            Arg::new("config")
                                .help("Path to training configuration file")
                                .required(true)
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("train_data")
                                .short('d')
                                .long("train_data")
                                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                                .help(
                                    "Path to training data. Overrides the training data file \
                                     specified in the configuration file.",
                                )
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("validation_data")
                                .short('v')
                                .long("validation_data")
                                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                                .help(
                                    "Path to validation data. Overrides the validation data file \
                                     specified in the configuration file.",
                                )
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("output_file")
                                .short('o')
                                .long("output_file")
                                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                                .help(
                                    "File path that the safetensors trained model will be written to. \
                                     Overrides the directory specified in the configuration file.",
                                )
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("model_arch")
                                .short('m')
                                .long("model_arch")
                                .help(
                                    "Model architecture to train. \
                                     Overrides the model architecture specified in the configuration file.",
                                )
                                .value_parser([
                                    "rt_cnn_lstm",
                                    "rt_cnn_tf",
                                    "ms2_bert",
                                    "ccs_cnn_lstm",
                                ])
                                .required(false)
                        )
                        .arg(
                            Arg::new("checkpoint_file")
                                .short('c')
                                .long("checkpoint_file")
                                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                                .help(
                                    "File path of the checkpoint safetensors file to load. \
                                     Overrides the checkpoint_file specified in the configuration file.",
                                )
                                .value_hint(ValueHint::FilePath),
                        ),
                )
                .subcommand(Command::new("inference")
                    .about("Perform inference on new data using a trained model")
                    .arg(
                        Arg::new("pretrained")
                            .long("pretrained")
                            .help("Name of a known pretrained model to use instead of passing --model. Examples: 'redeem-ccs', 'redeem-rt', 'alphapeptdeep-ccs'")
                            .value_parser(clap::builder::NonEmptyStringValueParser::new())
                            .value_hint(ValueHint::Other),
                    )
                    .arg(
                        Arg::new("config")
                            .help("Path to training configuration file")
                            .required(true)
                            .value_parser(clap::value_parser!(PathBuf))
                            .value_hint(ValueHint::FilePath),
                    )
                    .arg(
                        Arg::new("model_path")
                            .short('m')
                            .long("model")
                            .help("Path to the trained model file (*.safetensors)")
                            .value_parser(clap::value_parser!(PathBuf))
                            .value_hint(ValueHint::FilePath),
                    )
                    .arg(
                        Arg::new("inference_data")
                            .short('d')
                            .long("inference_data")
                            .help("Path to the input data file")
                            .value_parser(clap::value_parser!(PathBuf))
                            .value_hint(ValueHint::FilePath),
                    )
                    .arg(
                        Arg::new("output_file")
                            .short('o')
                            .long("output_file")
                            .help("Path to the output file for predictions (*.tsv or *.csv)")
                            .value_parser(clap::value_parser!(PathBuf))
                            .value_hint(ValueHint::FilePath),
                    )
                ),
        )
        .subcommand(
            Command::new("classifiers")
                .about("Run classification tools such as rescoring")
                .subcommand(
                    Command::new("rescore")
                        .about("Run rescoring tool with specified configuration")
                        .arg(
                            Arg::new("config")
                                .help("Path to classifier configuration file")
                                .required(true)
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        ),
                ),
                .subcommand(
                    Command::new("score")
                        .about("Score a Percolator .pin file with the semi-supervised classifier")
                        .arg(
                            Arg::new("pin")
                                .help("Path to the Percolator .pin input file")
                                .required(true)
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("config")
                                .help("Path to classifier JSON configuration file")
                                .required(true)
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        ),
                ),
        )
        .help_template(
            "{usage-heading} {usage}\n\n\
             {about-with-newline}\n\
             Written by {author-with-newline}Version {version}\n\n\
             {all-args}{after-help}",
        )
        .get_matches();

    match matches.subcommand() {
        Some(("properties", sub_m)) => handle_properties(sub_m),
        Some(("classifiers", sub_m)) => handle_classifiers(sub_m),
        _ => unreachable!("Subcommand is required by CLI configuration"),
    }
}

fn handle_properties(matches: &ArgMatches) -> Result<()> {
    match matches.subcommand() {
        Some(("train", train_matches)) => {
            let config_path: &PathBuf = train_matches.get_one("config").unwrap();
            log::info!(
                "[ReDeeM::Properties] Training from config: {:?}",
                config_path
            );

            let params: PropertyTrainConfig =
                PropertyTrainConfig::from_arguments(config_path, train_matches)?;

            match trainer::run_training(&params) {
                Ok(_) => Ok(()),
                Err(e) => {
                    log::error!("Training failed: {:#}", e);
                    std::process::exit(1)
                }
            }
        }
        Some(("inference", inference_matches)) => {
            let config_path: &PathBuf = inference_matches.get_one("config").unwrap();
            log::info!(
                "[ReDeeM::Properties] Inference using config: {:?}",
                config_path
            );

            let params: PropertyInferenceConfig =
                PropertyInferenceConfig::from_arguments(config_path, inference_matches)?;

            match inference::run_inference(&params) {
                Ok(_) => Ok(()),
                Err(e) => {
                    log::error!("Inference failed: {:#}", e);
                    std::process::exit(1)
                }
            }
        }
        _ => unreachable!(),
    }
}

fn handle_classifiers(matches: &ArgMatches) -> Result<()> {
    match matches.subcommand() {
        Some(("rescore", rescore_matches)) => {
            let config_path: &PathBuf = rescore_matches.get_one("config").unwrap();
            println!(
                "[ReDeeM::Classifiers] Rescoring using config: {:?}",
                config_path
            );
            // Call your classifier logic using config_path
            Ok(())
        }
        Some(("score", score_matches)) => {
            let pin_path: &PathBuf = score_matches.get_one("pin").unwrap();
            let config_path: &PathBuf = score_matches.get_one("config").unwrap();
            println!(
                "[ReDeeM::Classifiers] Scoring PIN file: {:?} using config: {:?}",
                pin_path, config_path
            );
            let result = score_pin_with_config(pin_path, config_path)?;
            println!(
                "[ReDeeM::Classifiers] Completed scoring {} PSMs.",
                result.predictions.as_slice().len()
            );
            Ok(())
        }
        _ => unreachable!(),
    }
}
