use anyhow::{Context, Result};
use candle_core::Device;
use csv::Reader;
use redeem_properties::{
    models::{
        model_interface::{ModelInterface, PredictionResult},
        ms2_bert_model::MS2BertModel,
    },
    utils::{data_handling::PeptideData, peptdeep_utils::{get_modification_indices, get_modification_string, load_modifications, remove_mass_shift, ModificationMap}},
};
use std::{
    collections::HashMap, fs::File, path::PathBuf
};

struct PredictionContext {
    peptides: Vec<String>,
    naked_peptides: Vec<String>,
    mods: Vec<String>,
    mod_sites: Vec<String>,
    charges: Vec<i32>,
    nces: Vec<i32>,
    instruments: Vec<String>,
    ms2_intensities: Vec<Vec<Vec<f32>>>,
}

impl PredictionContext {
    fn new(training_data: &Vec<PeptideData>, modification_map: &HashMap<(String, Option<char>), ModificationMap>) -> Self {
        let peptides: Vec<String> = training_data.iter().map(|p| p.sequence.clone()).collect();

        let naked_peptides: Vec<String> = training_data.iter().map(|p| remove_mass_shift(&p.sequence)).collect();
        let naked_peptides: Vec<String> = naked_peptides.iter().map(|p| p.trim_start_matches("-").to_string()).collect();


        // Get mod_str with get_modification_string
        let mod_strs: Vec<String> = training_data.iter().map(|p| get_modification_string(&p.sequence, modification_map)).collect();

        /// Get modification indices with get_modification_indices
        let mod_sites: Vec<String> = training_data.iter().map(|p| get_modification_indices(&p.sequence)).collect();


        let charges: Vec<i32> = training_data.iter().map(|p| p.charge.unwrap()).collect();
        let nces: Vec<i32> = training_data.iter().map(|p| p.nce.unwrap()).collect();
        let instruments: Vec<String> = training_data.iter().map(|p| p.instrument.clone().unwrap()).collect();
        let ms2_intensities: Vec<Vec<Vec<f32>>> = training_data.iter().map(|p| p.ms2_intensities.clone().unwrap()).collect();

        Self {
            peptides,
            naked_peptides,
            mods: mod_strs,
            mod_sites,
            charges,
            nces,
            instruments,
            ms2_intensities,
        }
    }
}

fn run_prediction(model: &mut MS2BertModel, prediction_context: &PredictionContext) -> Result<()> { // Changed Model
    match model.predict(
        &prediction_context.naked_peptides,
        &prediction_context.mods,
        &prediction_context.mod_sites,
        Some(prediction_context.charges.clone()), 
        Some(prediction_context.nces.clone()),
        Some(prediction_context.instruments.clone()),
    ) {
        Ok(predictions) => {
            if let PredictionResult::MS2Result(ms2_preds) = predictions {  
                let total_error: f32 = ms2_preds
                    .iter()
                    .zip(prediction_context.ms2_intensities.iter())
                    .map(|(outer_pred, outer_obs)| {
                        outer_pred
                            .iter()
                            .zip(outer_obs.iter())
                            .map(|(inner_pred, inner_obs)| {
                                inner_pred
                                    .iter()
                                    .zip(inner_obs.iter())
                                    .map(|(pred, obs)| (pred - obs).abs())
                                    .sum::<f32>() // Sum the innermost differences
                            })
                            .sum::<f32>() // Sum the differences from the middle vectors
                    })
                    .sum::<f32>(); // Sum the differences from the outer vectors


                print_predictions(&prediction_context.peptides, &ms2_preds, &prediction_context.ms2_intensities); 

                let mean_absolute_error = total_error / ms2_preds.len() as f32;
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

fn print_predictions(
    peptides: &[String],
    predicted_ms2_intensities: &Vec<Vec<Vec<f32>>>,
    observed_ms2_intensities: &Vec<Vec<Vec<f32>>>,
) {
    let mut peptides_iter = peptides.iter();
    let mut predicted_iter = predicted_ms2_intensities.iter();
    let mut observed_iter = observed_ms2_intensities.iter();

    loop {
        match (
            peptides_iter.next(),
            predicted_iter.next(),
            observed_iter.next(),
        ) {
            (Some(pep), Some(predicted), Some(observed)) => {
                let predicted_sum: f32 = predicted.iter().flat_map(|inner_vec| inner_vec.iter().copied()).sum();
                let observed_sum: f32 = observed.iter().flat_map(|inner_vec| inner_vec.iter().copied()).sum();


                println!("Peptide: {}", pep);
                println!("  Sum of Predicted Intensities: {:.6}", predicted_sum);
                println!("  Sum of Observed Intensities: {:.6}", observed_sum);
            }
            _ => break, // Exit the loop if any iterator is exhausted
        }
    }
}

fn main() -> Result<()> {
    let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
    let constants_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");

    // let device use cuda if available otherwise use cpu
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    println!("Device: {:?}", device);

    let mut model = MS2BertModel::new(&model_path, &constants_path, 0, 8, 4, true, device)
        .context("Failed to create MS2BertModel")?;

    // Open the CSV file
    let file_path = "data/predicted_fragment_intensities.csv";
    let file = File::open(file_path).unwrap();

    // Create a CSV reader
    let mut rdr = Reader::from_reader(file);

    // Group fragment intensities by peptide sequence
    let mut peptide_data_map: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut peptide_charges: HashMap<String, i32> = HashMap::new();

    for result in rdr.records() {
        let record = result.unwrap();
        let peptide_sequence = &record[0];
        let precursor_charge: i32 = record[1].parse().unwrap();
        let fragment_type = &record[2];
        let fragment_ordinal: usize = record[3].parse().unwrap();
        let fragment_charge: i32 = record[4].parse().unwrap();
        let experimental_intensity: f32 = record[6].parse().unwrap();

        // Get naked peptide sequence
        let naked_peptide = remove_mass_shift(peptide_sequence);

        // Get length of the peptide sequence
        let peptide_len = naked_peptide.len() - 1;

        // Initialize the peptide's intensity matrix if it doesn't exist
        peptide_data_map
            .entry(peptide_sequence.to_string())
            .or_insert_with(|| vec![vec![0.0; 8]; peptide_len]); // Initialize with enough rows

        // Update the peptide's charge
        peptide_charges.insert(peptide_sequence.to_string(), precursor_charge);

        // Determine the column index based on fragment type and charge
        let col = match (fragment_type, fragment_charge) {
            ("B", 1) => 0, // b_z1
            ("B", 2) => 1, // b_z2
            ("Y", 1) => 2, // y_z1
            ("Y", 2) => 3, // y_z2
            _ => continue, // Skip unsupported fragment types or charges
        };

        // Update the MS2 intensities matrix
        let row = peptide_len - 1; // Convert to zero-based index
        peptide_data_map
            .get_mut(peptide_sequence)
            .unwrap()
            .resize(row + 1, vec![0.0; 8]); // Ensure the matrix has enough rows
        peptide_data_map.get_mut(peptide_sequence).unwrap()[row][col] = experimental_intensity;
    }

    // Create PeptideData instances for each peptide
    let mut training_data: Vec<PeptideData> = Vec::new();

    for (sequence, ms2_intensities) in peptide_data_map {
        let charge = peptide_charges.get(&sequence).copied();
        let peptide_data = PeptideData::new(
            &sequence,
            charge,
            Some(20), // Example NCE
            Some("QE"), // Example instrument
            None, // Retention time
            None, // Ion mobility
            Some(ms2_intensities), // MS2 intensities
        );
        training_data.push(peptide_data);
    }

    println!("Loaded {} peptides from the CSV file.", training_data.len());

    // Create the prediction context using the training data
    let modifications = load_modifications().context("Failed to load modifications")?;
    let prediction_context = PredictionContext::new(&training_data, &modifications);

    // Run prediction using the training data as the test data
    let result = run_prediction(&mut model, &prediction_context, );

    match result {
        Ok(_) => println!("Ran prediction successfully."),
        Err(e) => println!("Failed to run prediction: {:?}", e),
    }

    // Fine-tune the model
    
    let learning_rate = 0.001;
    let epochs = 5;
    let result = model
        .fine_tune(&training_data, modifications, 3, learning_rate, epochs)
        .context("Failed to fine-tune the model");

    match result {
        Ok(_) => println!("Model fine-tuned successfully."),
        Err(e) => println!("Failed to fine-tune model: {:?}", e),
    }

    // Test prediction again with a few peptides after fine-tuning
    let result = run_prediction(&mut model, &prediction_context);

    match result {
        Ok(_) => println!("Ran prediction successfully."),
        Err(e) => println!("Failed to run prediction: {:?}", e),
    }

    Ok(())
}
