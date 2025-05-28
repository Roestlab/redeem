use std::{collections::HashMap, sync::Arc};
use std::fs::File;
use std::path::Path;
use std::io::BufReader;
use anyhow::{Result, Context};
use csv::ReaderBuilder;
use redeem_properties::utils::peptdeep_utils::{get_modification_indices, get_modification_string, ModificationMap};
use redeem_properties::utils::{data_handling::{PeptideData, TargetNormalization}, peptdeep_utils::remove_mass_shift};



/// Load peptide training data from a CSV or TSV file and optionally normalize RT.
///
/// Returns both the peptide vector and optionally (mean, std) of retention times.
pub fn load_peptide_data<P: AsRef<Path>>(
    path: P,
    model_arch: &str,
    nce: Option<i32>,
    instrument: Option<String>,
    normalize_target: Option<String>,
    modifications: &HashMap<(String, Option<char>), ModificationMap>,
) -> Result<(Vec<PeptideData>, TargetNormalization)> {
    let file = File::open(&path)
        .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);

    let is_tsv = path.as_ref().extension().map(|e| e == "tsv").unwrap_or(false);
    let delimiter = if is_tsv { b'\t' } else { b',' };

    let mut rdr = ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(true)
        .from_reader(reader);

    let headers = rdr.headers()?.clone();
    let mut peptides = Vec::new();
    let mut target_values = Vec::new();

    let normalize_field = if model_arch.contains("ccs") {
        "ccs"
    } else {
        "retention time"
    };

    for result in rdr.records() {
        let record = result?;

        let sequence_bytes: Arc<[u8]> = Arc::from(
            record
                .get(headers.iter().position(|h| h.to_lowercase() == "sequence").unwrap_or(2))
                .unwrap_or("")
                .as_bytes()
                .to_vec()
                .into_boxed_slice(),
        );

        let sequence_str = String::from_utf8_lossy(&sequence_bytes);

        let naked_sequence = Arc::from(remove_mass_shift(&sequence_str).as_bytes().to_vec().into_boxed_slice());
        let mods: Arc<[u8]> = Arc::from(get_modification_string(&sequence_str, modifications).into_bytes().into_boxed_slice());
        let mod_sites: Arc<[u8]> = Arc::from(get_modification_indices(&sequence_str).into_bytes().into_boxed_slice());

        let retention_time = record
            .get(headers.iter().position(|h| h.to_lowercase() == "retention time").unwrap_or(3))
            .and_then(|s| s.parse::<f32>().ok());

        let charge = match model_arch {
            "rt_cnn_lstm" | "rt_cnn_tf" => None,
            _ => record
                .get(headers.iter().position(|h| h.to_lowercase() == "charge").unwrap_or(usize::MAX))
                .and_then(|s| s.parse::<i32>().ok()),
        };

        let precursor_mass = record
            .get(headers.iter().position(|h| h.to_lowercase() == "precursor_mass").unwrap_or(usize::MAX))
            .and_then(|s| s.parse::<f32>().ok());

        let ion_mobility = record
            .get(headers.iter().position(|h| h.to_lowercase() == "ion_mobility").unwrap_or(usize::MAX))
            .and_then(|s| s.parse::<f32>().ok());

        let ccs = record
            .get(headers.iter().position(|h| h.to_lowercase() == "ccs").unwrap_or(usize::MAX))
            .and_then(|s| s.parse::<f32>().ok());

        let in_nce = match model_arch {
            "ms2_bert" => nce.or_else(|| {
                record
                    .get(headers.iter().position(|h| h.to_lowercase() == "nce").unwrap_or(usize::MAX))
                    .and_then(|s| s.parse::<i32>().ok())
            }),
            _ => None,
        };

        let in_instrument = match model_arch {
            "ms2_bert" => instrument
                .as_ref()
                .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
                .or_else(|| {
                    record
                        .get(headers.iter().position(|h| h.to_lowercase() == "instrument").unwrap_or(usize::MAX))
                        .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
                }),
            _ => None,
        };

        if let Some(val) = match normalize_field {
            "ccs" => ccs,
            _ => retention_time,
        } {
            target_values.push(val);
        }

        peptides.push(PeptideData {
            modified_sequence: sequence_bytes,
            naked_sequence,
            mods,
            mod_sites,
            charge,
            precursor_mass,
            nce: in_nce,
            instrument: in_instrument,
            retention_time,
            ion_mobility,
            ccs,
            ms2_intensities: None,
        });
    }

    match TargetNormalization::from_str(normalize_target) {
        TargetNormalization::ZScore(_, _) if !target_values.is_empty() => {
            let mean = target_values.iter().copied().sum::<f32>() / target_values.len() as f32;
            let std = (target_values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / target_values.len() as f32).sqrt();
            for peptide in &mut peptides {
                match normalize_field {
                    "ccs" => if let Some(val) = peptide.ccs.as_mut() {
                        *val = (*val - mean) / std;
                    },
                    _ => if let Some(val) = peptide.retention_time.as_mut() {
                        *val = (*val - mean) / std;
                    },
                }
            }
            Ok((peptides, TargetNormalization::ZScore(mean, std)))
        }
        TargetNormalization::MinMax(_, _) if !target_values.is_empty() => {
            let min = *target_values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let max = *target_values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let range = max - min;
            for peptide in &mut peptides {
                match normalize_field {
                    "ccs" => if let Some(val) = peptide.ccs.as_mut() {
                        *val = (*val - min) / range;
                    },
                    _ => if let Some(val) = peptide.retention_time.as_mut() {
                        *val = (*val - min) / range;
                    },
                }
            }
            Ok((peptides, TargetNormalization::MinMax(min, max)))
        }
        _ => Ok((peptides, TargetNormalization::None)),
    }
}
