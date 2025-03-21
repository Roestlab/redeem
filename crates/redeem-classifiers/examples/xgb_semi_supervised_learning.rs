use anyhow::{Context, Result};
use csv::ReaderBuilder;
use machine_info::Machine;
use ndarray::{Array1, Array2};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::process;

use redeem_classifiers::psm_scorer::SemiSupervisedLearner;
use redeem_classifiers::models::utils::ModelType;

fn read_features_tsv(path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_path(path)?;

    let mut data = Vec::new();

    for result in reader.records() {
        let record = result?;
        let row: Vec<f32> = record
            .iter()
            .map(|field| field.parse::<f32>())
            .collect::<Result<_, _>>()?;
        data.push(row);
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    Array2::from_shape_vec(
        (n_samples, n_features),
        data.into_iter().flatten().collect(),
    )
    .map_err(|e| e.into())
}

fn read_labels_tsv(path: &str) -> Result<Array1<i32>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b'\t')
        .from_path(path)?;

    let labels: Vec<i32> = reader
        .records()
        .map(|r| {
            let record = r?;
            let value = record.get(0).ok_or_else(|| "Empty row".to_string())?;
            value.parse::<i32>().map_err(|e| e.into())
        })
        .collect::<Result<_, Box<dyn Error>>>()?;

    Ok(Array1::from_vec(labels))
}

fn save_predictions_to_csv(
    predictions: &Array1<f32>,
    file_path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(file_path)?;

    for &pred in predictions.iter() {
        writeln!(file, "{}", pred)?;
    }

    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    let mut m = Machine::new();
    m.track_process(process::id() as i32).unwrap();
    
    // Load the test data from the TSV files
    let x = read_features_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_for_testing.csv").unwrap();
    // Select first 10 columns of data
    let x = x.slice(ndarray::s![.., ..10]).to_owned();

    let y = read_labels_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_labels_for_testing.csv").unwrap();

    println!("Loaded features shape: {:?}", x.shape());
    println!("Loaded labels shape: {:?}", y.shape());

    // Create and train your SemiSupervisedLearner
    let xgb_params = ModelType::XGBoost {
            max_depth: 6,
            num_boost_round: 100,
            early_stopping_rounds: 10,
            verbose_eval: false,
        };
    let mut learner = SemiSupervisedLearner::new(
        xgb_params,
        0.01,
        1.0,
        5,
        Some((1.0, 1.0))
    );
    let predictions = learner.fit(x, y.clone());

    println!("Labels: {:?}", y);

    // Evaluate the predictions
    println!("Predictions: {:?}", predictions);

    let processes = m.processes_status();
    let system = m.system_status();
    let graphics = m.graphics_status();
    println!("{:?} {:?} {:?}", processes, system, graphics);

    // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();
    Ok(())
}