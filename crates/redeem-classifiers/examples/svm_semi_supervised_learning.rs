use anyhow::{Context, Ok, Result};
use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Write};

use redeem_classifiers::data_handling::PsmMetadata;
use redeem_classifiers::math::{Array1, Array2};
use redeem_classifiers::models::utils::ModelType;
use redeem_classifiers::psm_scorer::SemiSupervisedLearner;

/// Load a test PSM CSV file into feature matrix, labels, and metadata.
///
/// # Arguments
/// * `path` - Path to the CSV file
///
/// # Returns
/// A tuple of (`x`, `y`, `PsmMetadata`)
pub fn load_test_psm_csv(path: &str) -> Result<(Array2<f32>, Array1<i32>, PsmMetadata)> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(file));

    let headers = reader
        .headers()?
        .iter()
        .map(|h| h.to_string())
        .collect::<Vec<_>>();

    // Find indices
    let file_id_idx = headers.iter().position(|h| h == "file_id").unwrap();
    let spec_id_idx = headers.iter().position(|h| h == "spec_id").unwrap();
    let label_idx = headers.iter().position(|h| h == "label").unwrap();

    // Everything else is a feature
    let feature_indices: Vec<usize> = (0..headers.len())
        .filter(|&i| i != file_id_idx && i != spec_id_idx && i != label_idx)
        .collect();

    let feature_names = feature_indices
        .iter()
        .map(|&i| headers[i].clone())
        .collect::<Vec<_>>();

    let mut file_ids = Vec::new();
    let mut spec_ids = Vec::new();
    let mut labels = Vec::new();
    let mut features = Vec::new();

    for result in reader.records() {
        let record = result?;

        file_ids.push(record[file_id_idx].parse::<usize>()?);
        spec_ids.push(record[spec_id_idx].to_string());
        labels.push(record[label_idx].parse::<i32>()?);

        let row = feature_indices
            .iter()
            .map(|&i| record[i].parse::<f32>().unwrap_or(f32::NAN))
            .collect::<Vec<f32>>();

        features.extend(row);
    }

    let n_rows = labels.len();
    let n_cols = feature_indices.len();

    let x = Array2::from_shape_vec((n_rows, n_cols), features)?;
    let y = Array1::from_vec(labels);

    let metadata = PsmMetadata {
        file_id: file_ids,
        spec_id: spec_ids,
        feature_names,
    };

    Ok((x, y, metadata))
}

// fn save_predictions_to_csv(
//     predictions: &Array1<f32>,
//     file_path: &str,
// ) -> Result<(), Box<dyn Error>> {
//     let mut file = File::create(file_path)?;

//     for &pred in predictions.iter() {
//         writeln!(file, "{}", pred)?;
//     }

//     Ok(())
// }

#[cfg(feature = "svm")]
fn run_psm_scorer(x: &Array2<f32>, y: &Array1<i32>, metadata: &PsmMetadata) -> Result<Array1<f32>> {
    let params = ModelType::SVM {
        eps: 0.1,
        c: (1.0, 1.0),
        kernel: "linear".to_string(),
        gaussian_kernel_eps: 0.1,
        polynomial_kernel_constant: 1.0,
        polynomial_kernel_degree: 3.0,
    };
    let mut learner = SemiSupervisedLearner::new(params, 0.001, 1.0, 500, Some((0.15, 1.0)));
    let predictions = learner.fit(x, y.clone(), metadata);
    Ok(predictions)
}

#[cfg(not(feature = "svm"))]
fn run_psm_scorer(x: &Array2<f32>, y: &Array1<i32>, metadata: &PsmMetadata) -> Result<Array1<f32>> {
    unimplemented!("SVM is not available in this build. Please enable the svm feature.");
}

fn main() -> Result<()> {
    env_logger::init();
    // Load the test data from the TSV files
    let (x, y, metadata) = load_test_psm_csv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_with_metadata_for_testing_redeem.csv")?;

    // Select first 10 columns of data
    let x = x.select_columns(0..10);

    println!("Loaded features shape: {:?}", x.shape());
    println!("Loaded labels shape: {:?}", y.shape());

    // Create and train your SemiSupervisedLearner
    let predictions = run_psm_scorer(&x, &y, &metadata).context("Failed to run PSM scorer")?;

    println!("Labels: {:?}", y);

    // Evaluate the predictions
    println!("Predictions: {:?}", predictions);
    // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();
    Ok(())
}
