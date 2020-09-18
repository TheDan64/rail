use crate::errors::ValidationError;
use crate::layers::activations::Activation;
use crate::layers::Layer;
use crate::{Matrix, Weight};
use arrayfire::{ge, mul, randu};

pub struct Dropout {
    drop_rate: f64,
    input_shape: [u64; 4],
}

impl Dropout {
    pub fn new(drop_rate: f64) -> Result<Box<Self>, ValidationError> {
        if drop_rate < 0. || drop_rate > 1. {
            return Err(ValidationError::UnitInterval { field_name: "drop_rate" });
        }

        Ok(Box::new(Dropout {
            drop_rate,
            input_shape: [0; 4],
        }))
    }
}

impl Layer for Dropout {
    fn feedforward(&self, input: Matrix) -> Matrix {
        let keep_prob = 1. - self.drop_rate;
        let scale = 1. / keep_prob;
        let dims = input.dims();
        let scale_mat = scale * input;
        let random_mat = randu::<f64>(dims);
        let keep_mask = ge(&random_mat, &self.drop_rate, true);

        mul(&scale_mat, &keep_mask, true)
    }

    fn backpropagate(&mut self, _input: &Matrix, _output: &Matrix, d_error: Matrix, _lr: Weight) -> Matrix {
        d_error
    }

    fn initialize(&mut self, input_shape: &[u64; 4]) {
        self.input_shape = *input_shape;
    }

    fn output_shape(&self) -> [u64; 4] {
        self.input_shape
    }

    fn serialize(&self) -> String {
        unimplemented!("Dropout::serialize");
    }

    fn a_function(&self) -> Activation {
        unimplemented!("Dropout::a_function")
    }

    fn display(&self) {
        unimplemented!("Dropout::display")
    }
}

#[test]
fn test_dropout_feedforward() {
    use arrayfire::sum_all;

    let dims = dim4!(4, 5, 1, 1);
    let input_mat = randu::<f64>(dims);
    let sum = sum_all(&input_mat);
    let dropout = Dropout::new(0.2).unwrap();
    let output_mat = dropout.feedforward(input_mat);

    // TODO: What's the best way to test this?
    // assert_eq!(sum.0, sum_all(&output_mat).0);
}
