use std::fs::File;
use std::path::Path;
use std::io::BufReader;
use anyhow::{Result, Context};
use csv::ReaderBuilder;
use redeem_properties::utils::data_handling::PeptideData;

/// Load peptide training data from a CSV or TSV file.
///
/// Automatically determines the delimiter and supports RT models.
/// Currently expects columns: "sequence", "retention time" (others optional).
///
/// # Arguments
/// * `path` - Path to the input CSV/TSV file
///
/// # Returns
/// Vector of parsed `PeptideData` records
pub fn load_peptide_data<P: AsRef<Path>>(path: P) -> Result<Vec<PeptideData>> {
    let file = File::open(&path).with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);

    let is_tsv = path.as_ref().extension().map(|e| e == "tsv").unwrap_or(false);
    let delimiter = if is_tsv { b'\t' } else { b',' };

    let mut rdr = ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(true)
        .from_reader(reader);

    let headers = rdr.headers()?.clone();

    let mut peptides = Vec::new();
    for result in rdr.records() {
        let record = result?;

        let sequence = record
            .get(headers.iter().position(|h| h == "sequence").unwrap_or(2))
            .unwrap_or("")
            .to_string();

        let retention_time = record
            .get(headers.iter().position(|h| h == "retention time").unwrap_or(3))
            .and_then(|s| s.parse::<f32>().ok());

        let charge = record
            .get(headers.iter().position(|h| h == "charge").unwrap_or(usize::MAX))
            .and_then(|s| s.parse::<i32>().ok());

        let nce = record
            .get(headers.iter().position(|h| h == "nce").unwrap_or(usize::MAX))
            .and_then(|s| s.parse::<i32>().ok());

        let instrument = record
            .get(headers.iter().position(|h| h == "instrument").unwrap_or(usize::MAX))
            .map(|s| s.to_string());

        peptides.push(PeptideData::new(
            &sequence,
            charge,
            nce,
            instrument.as_deref(),
            retention_time,
            None,
            None,
        ));
    }

    Ok(peptides)
}
