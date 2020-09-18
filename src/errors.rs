use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug)]
pub enum ValidationError {
    /// `field_name` wasn't in [0, 1)
    UnitInterval { field_name: &'static str },
}

impl Display for ValidationError {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match self {
            ValidationError::UnitInterval { field_name } => {
                f.write_str(field_name)?;
                f.write_str(" was not within the range [0, 1)")
            }
        }
    }
}

impl Error for ValidationError {}
