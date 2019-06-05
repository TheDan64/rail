#[macro_use]
pub extern crate arrayfire;

mod utils;
pub mod functions;
pub mod layers;
pub mod model;

pub type Weight = f64;
pub type Matrix = arrayfire::Array<Weight>;
