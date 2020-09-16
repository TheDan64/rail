use crate::layers::activations::Activation;
use crate::layers::Layer;
use crate::{Matrix, Weight};

pub struct Dropout {
    keep_probability: f64,
}

impl Dropout {
    pub fn new(keep_probability: f64) -> Self {
        Dropout {
            keep_probability,
        }
    }
}

impl Layer for Dropout {
    fn feedforward(&self, _input: Matrix) -> Matrix {
        unimplemented!()
    }

    fn backpropagate(&mut self, _input: &Matrix, _output: &Matrix, _d_error: Matrix, _lr: Weight) -> Matrix {
        unimplemented!()
    }

    fn initialize(&mut self, _input_shape: &[u64; 4]) {
        unimplemented!()
    }

    fn output_shape(&self) -> [u64; 4] {
        unimplemented!()
    }

    fn serialize(&self) -> String {
        unimplemented!();
    }


    fn a_function(&self) -> Activation {
        unimplemented!()
    }

    fn display(&self) {
        unimplemented!()
    }
}
