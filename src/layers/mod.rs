pub mod dense;
pub mod conv;
pub mod activations;
pub mod dropout;

use crate::{ Weight, Matrix };
use crate::layers::activations::Activation;


// TODO: refactor to const strings from each layer
pub fn load_layer(src: &[&str]) -> Box<dyn Layer> {
    use crate::layers::dense::Dense;

    match src[0] {
        "dense" => Dense::deserialize(&src[1..]),
        v => panic!("wrong kind of layer {:?}!", v),
    }

}

pub trait Layer {
    fn feedforward(&self, input: Matrix) -> Matrix;
    fn backpropagate(&mut self, input: &Matrix, output: &Matrix, d_error: Matrix, lr: Weight) -> Matrix;
    fn initialize(&mut self, inputs: &[u64; 4]);
    fn output_shape(&self) -> [u64; 4];
    fn serialize(&self) -> String;

    fn a_function(&self) -> Activation;

    fn display(&self);
}
