use std::fs::File;
use std::io::BufWriter;
use anyhow::{Result, Context};
use std::path::Path;
use redeem_properties::utils::data_handling::PeptideData;

/// Write a vector of PeptideData to a CSV or TSV file based on file extension.
pub fn write_peptide_data<P: AsRef<Path>>(data: &[PeptideData], output_path: P) -> Result<()> {
    

    let path = output_path.as_ref();
    let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("csv");
    let delimiter = match extension {
        "tsv" => '\t',
        _ => ',',
    };

    let file = File::create(path).with_context(|| format!("Failed to create output file: {:?}", path))?;
    let mut writer = csv::WriterBuilder::new()
        .delimiter(delimiter as u8)
        .from_writer(BufWriter::new(file));

    // Write headers
    writer.write_record(&[
        "modified_sequence",
        "naked_sequence",
        "mods",
        "mod_sites",
        "charge",
        "precursor_mass",
        "nce",
        "instrument",
        "retention_time",
        "ion_mobility",
        "ccs",
        "ms2_intensities",
    ])?;

    for entry in data {
        let ms2_str = entry.ms2_intensities.as_ref()
            .map(|intensities| {
                intensities.iter()
                    .map(|v| v.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","))
                    .collect::<Vec<_>>().join("|")
            })
            .unwrap_or_default();

        writer.write_record(&[
            entry.modified_sequence_str(),
            entry.naked_sequence_str(),
            entry.mods_str(),
            entry.mod_sites_str(),
            &entry.charge.map_or(String::new(), |c| c.to_string()),
            &entry.precursor_mass.map_or(String::new(), |m| format!("{:.4}", m)),
            &entry.nce.map_or(String::new(), |n| n.to_string()),
            &entry.instrument_str().unwrap_or_default().to_string(),
            &entry.retention_time.map_or(String::new(), |r| format!("{:.4}", r)),
            &entry.ion_mobility.map_or(String::new(), |im| format!("{:.4}", im)),
            &entry.ccs.map_or(String::new(), |c| format!("{:.4}", c)),
            &ms2_str,
        ])?;
    }

    writer.flush()?;
    Ok(())
}
