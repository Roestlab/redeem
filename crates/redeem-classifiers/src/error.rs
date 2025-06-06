use std::error::Error;
use std::fmt;

/// Custom error type for TDC calculation failures
#[derive(Debug)]
pub enum TdcError {
    NaNFound(usize), // Number of NaN values found
    LengthMismatch,
}

impl fmt::Display for TdcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TdcError::NaNFound(count) => write!(f, "Found {} NaN values in scores array", count),
            TdcError::LengthMismatch => write!(f, "Scores and target arrays must have equal length"),
        }
    }
}

impl Error for TdcError {}