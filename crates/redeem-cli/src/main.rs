use clap::{Arg, Command, ArgMatches, ValueHint};
use log::LevelFilter;
use std::path::PathBuf;
use anyhow::Result;

use redeem_cli::properties::train::input::{self, PropertyTrainConfig};
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
                        .help_template(
                            "{usage-heading} {usage}\n\n\
                             {about-with-newline}\n\
                             Written by {author-with-newline}Version {version}\n\n\
                             {all-args}{after-help}",
                        ),
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
            println!("[ReDeeM::Properties] Training from config: {:?}", config_path);
            let params: PropertyTrainConfig = input::PropertyTrainConfig::from_arguments(config_path, train_matches)?;
            let _ = trainer::run_training(&params);
            Ok(())
        }
        _ => unreachable!(),
    }
}

fn handle_classifiers(matches: &ArgMatches) -> Result<()> {
    match matches.subcommand() {
        Some(("rescore", rescore_matches)) => {
            let config_path: &PathBuf = rescore_matches.get_one("config").unwrap();
            println!("[ReDeeM::Classifiers] Rescoring using config: {:?}", config_path);
            // Call your classifier logic using config_path
            Ok(())
        }
        _ => unreachable!(),
    }
}
