use rand::distributions::Normal;
use rand::prelude::*;

use crate::{ Weight, Matrix };
use crate::utils::map_array;

use std::fmt::{ self, Display };

#[derive(Copy, Clone)]
pub enum Activation {
    Relu,
    /// https://arxiv.org/pdf/1511.07289v5.pdf
    Elu,
    Tanh,
    Sigmoid,
    Softmax
}

impl From<&str> for Activation {
    fn from(src: &str) -> Self {
        match src {
            "Relu"    => Activation::Relu,
            "Elu"     => Activation::Elu,
            "Tanh"    => Activation::Tanh,
            "Sigmoid" => Activation::Sigmoid,
            "Softmax" => Activation::Softmax,
            other       => panic!("invalid activation function {:?}", other)
        }
    }
}

impl Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Activation::Relu    => write!(f, "Relu"),
            Activation::Elu     => write!(f, "Elu"),
            Activation::Tanh    => write!(f, "Tanh"),
            Activation::Sigmoid => write!(f, "Sigmoid"),
            Activation::Softmax => write!(f, "Softmax")
        }
    }
}

pub fn initialize_weights(activation: Activation, inputs: usize, neurons: usize) -> (Vec<Weight>, Vec<Weight>) {
    let normal = Normal::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    let c = match activation {
        Activation::Relu => 2.0,
        Activation::Elu => 2.0,
        Activation::Tanh => 1.0,
        Activation::Sigmoid => 1.0,
        Activation::Softmax => 1.0
    };

    let weight_values = (0..inputs * neurons)
        .map(|_| (normal.sample(&mut rng) * c))
        .collect::<Vec<Weight>>();

    let bias_values = (0..inputs * neurons)
        .map(|_| (normal.sample(&mut rng) * c))
        .collect::<Vec<Weight>>();

    (weight_values, bias_values)
}

pub fn relu(input: Matrix) -> Matrix {
    map_array(input, |d| Weight::max(0.0, *d))
}

pub fn drelu(input: Matrix) -> Matrix {
    let dims = input.dims();
    let size = dims[0] * dims[1] * dims[2] * dims[3];
    let buf = vec![1.0; size as usize];
    Matrix::new(&buf, dims)
}

pub fn tanh(input: Matrix) -> Matrix {
    arrayfire::tanh(&input)
}

pub fn dtanh(input: Matrix) -> Matrix {
    map_array(input, |v| 1.0 - v.powf(2.0))
}

pub fn sigmoid(input: Matrix) -> Matrix {
    arrayfire::sigmoid(&input)
}

pub fn dsigmoid(input: Matrix) -> Matrix {
    map_array(input, |v| v * ( 1.0 - v ))
}


pub fn softmax(input: Matrix) -> Matrix {
    let exps = arrayfire::exp(&input);
    let sums = arrayfire::sum(&exps, 1);
    arrayfire::div(&exps, &sums, true)
}

pub fn dsoftmax(input: Matrix) -> Matrix {
    //  zi ((i == j) - zj)
    input
}

const ELU_ALPHA: f64 = 1.;

fn elu_inner(x: &f64) -> f64 {
    if *x >= 0. {
        *x
    } else {
        ELU_ALPHA * (x.exp() - 1.)
    }
}

pub fn elu(input: Matrix) -> Matrix {
    map_array(input, elu_inner)
}

pub fn delu(input: Matrix) -> Matrix {
    map_array(input, |&x| if x > 0. {
        1.
    } else {
        elu_inner(&x) + ELU_ALPHA
    })
}


#[cfg(test)]
mod tests {
    use arrayfire::Dim4;
    use crate::utils::array_data;
    use super::*;

    #[test]
    fn relu_test() {
        let data = Matrix::new(&[1.0, 0.0, -1.0, 0.4], Dim4::new(&[4, 1, 1, 1]));
        let data2 = array_data(relu(data));
        assert_eq!(data2, &[1.0, 0.0, 0.0, 0.4]);

        let data = Matrix::new(&[-1.0, 1.0, -0.2, 0.9], Dim4::new(&[4, 1, 1, 1]));
        let data2 = array_data(relu(data));
        assert_eq!(data2, &[0.0, 1.0, 0.0, 0.9]);
    }

    #[test]
    fn softmax_test() {
        let data = Matrix::new(&[-1.0, 1.0], Dim4::new(&[1,2,1,1]));
        let expected_1 = array_data(softmax(data));

        let data = Matrix::new(&[0.2, 0.4], Dim4::new(&[1,2,1,1]));
        let expected_2 = array_data(softmax(data));

        let data = Matrix::new(&[-1.0, 1.0, 0.2, 0.4], Dim4::new(&[1,2,2,1]));
        let data2 = softmax(data);

        let d = array_data(data2);

        let mut expected = Vec::new();
        expected.extend_from_slice(&expected_1);
        expected.extend_from_slice(&expected_2);
        assert_eq!(d, expected);
    }


    #[test]
    fn accel_softmax_text() {
        let data = Matrix::new(&[-1.0, 1.0, 0.2, 0.4], Dim4::new(&[1,2,2,1]));
        let expected = array_data(softmax(data));

        let data = Matrix::new(&[-1.0, 1.0, 0.2, 0.4], Dim4::new(&[1,2,2,1]));
        let exps = arrayfire::exp(&data);
        let sums = arrayfire::sum(&exps, 1);
        let sfmx = arrayfire::div(&exps, &sums, true);
        assert_eq!(expected, array_data(sfmx));

    }
}


