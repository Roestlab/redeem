use std::fs::File;
use std::path::Path;
use std::io::BufReader;
use anyhow::{Result, Context};
use csv::ReaderBuilder;
use redeem_properties::utils::data_handling::PeptideData;

/// Load peptide training data from a CSV or TSV file and optionally normalize RT.
///
/// Returns both the peptide vector and optionally (mean, std) of retention times.
pub fn load_peptide_data<P: AsRef<Path>>(
    path: P,
    nce: Option<i32>,
    instrument: Option<String>,
    normalize_rt: bool,
) -> Result<(Vec<PeptideData>, Option<(f32, f32)>)> {
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
    let mut rt_values = Vec::new();

    for result in rdr.records() {
        let record = result?;

        let sequence = record
            .get(headers.iter().position(|h| h.to_lowercase() == "sequence").unwrap_or(2))
            .unwrap_or("")
            .to_string();

        let retention_time = record
            .get(headers.iter().position(|h| h.to_lowercase() == "retention time").unwrap_or(3))
            .and_then(|s| s.parse::<f32>().ok());

        let charge = record
            .get(headers.iter().position(|h| h.to_lowercase() == "charge").unwrap_or(usize::MAX))
            .and_then(|s| s.parse::<i32>().ok());

        let precursor_mass = record
            .get(headers.iter().position(|h| h.to_lowercase() == "precursor_mass").unwrap_or(usize::MAX))
            .and_then(|s| s.parse::<f32>().ok());

        let ion_mobility = record
            .get(headers.iter().position(|h| h.to_lowercase() == "ion_mobility").unwrap_or(usize::MAX))
            .and_then(|s| s.parse::<f32>().ok());

        let ccs = record
            .get(headers.iter().position(|h| h.to_lowercase() == "ccs").unwrap_or(usize::MAX))
            .and_then(|s| s.parse::<f32>().ok());

        let in_nce = nce.or_else(|| {
            record
                .get(headers.iter().position(|h| h.to_lowercase() == "nce").unwrap_or(usize::MAX))
                .and_then(|s| s.parse::<i32>().ok())
        });

        let in_instrument = instrument.clone().or_else(|| {
            record
                .get(headers.iter().position(|h| h.to_lowercase() == "instrument").unwrap_or(usize::MAX))
                .map(|s| s.to_string())
        });

        if let Some(rt) = retention_time {
            rt_values.push(rt);
        }

        peptides.push(PeptideData {
            sequence,
            charge,
            precursor_mass,
            nce: in_nce,
            instrument: in_instrument,
            retention_time,
            ion_mobility,
            ccs,
            ms2_intensities: None
        });
    }

    if normalize_rt && !rt_values.is_empty() {
        let mean = rt_values.iter().copied().sum::<f32>() / rt_values.len() as f32;
        let std = (rt_values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>()
            / rt_values.len() as f32)
            .sqrt();

        for peptide in &mut peptides {
            if let Some(rt) = peptide.retention_time.as_mut() {
                *rt = (*rt - mean) / std;
            }
        }

        Ok((peptides, Some((mean, std))))
    } else {
        Ok((peptides, None))
    }
}

