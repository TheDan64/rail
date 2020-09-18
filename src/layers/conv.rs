// TODO: add biases

use arrayfire::{
    Dim4,
    convolve2,
    ConvDomain,
    ConvMode,
    randu,
};

use crate::layers::Layer;
use crate::layers::activations::Activation;
use crate::{ Weight, Matrix };

pub struct Conv {
    filters: Matrix,
    num_filters: u64,
    filter_size: (u64, u64),
    input_shape: Dim4,
    a_function: Activation,
    flatten: bool,
    activation: Box<dyn Fn(Matrix) -> Matrix>,
    dactivation: Box<dyn Fn(Matrix) -> Matrix>
}


impl Layer for Conv {
    fn feedforward(&self, input: Matrix) -> Matrix {
        let output = convolve2(&input, &self.filters, ConvMode::DEFAULT, ConvDomain::AUTO);
        let output = (self.activation)(output);

        if !self.flatten {
            return output
        }

        let dims = output.dims();
        let batch_size = dims[3];

        let mut dims = self.output_shape();
        let a = dims.iter().fold(1, |acc, x| acc * x);
        dims = [batch_size, a, 1, 1];

        let flat = arrayfire::moddims(&output, Dim4::new(&dims));
        flat
    }

    // FIXME: this is totally broken
    fn backpropagate(&mut self, input: &Matrix, output: &Matrix, d_error: Matrix, lr: Weight) -> Matrix {
        let flipped = arrayfire::flip(&self.filters, 0);
        let next_errors = arrayfire::convolve2(&d_error, &flipped, ConvMode::DEFAULT, ConvDomain::SPATIAL);

        let d_activation = (self.dactivation)(output.clone());
        let gradient = arrayfire::mul(&d_activation, &d_error, true) * lr;
        let mut delta = arrayfire::convolve2(&input, &gradient, ConvMode::DEFAULT, ConvDomain::SPATIAL);
        println!("entering backprop");
        delta = arrayfire::sum(&delta, 3);
        af_print!("deltas:", delta);
        let sf = &self.filters;
        af_print!("self.filters:", sf);
        assert_eq!(delta.dims(), self.filters.dims());
        self.filters = arrayfire::add(&self.filters, &delta, true);
        // self.filters += delta;

        println!("exiting backprop");
        next_errors
    }

    fn initialize(&mut self, input_shape: &[u64; 4]) {
        self.input_shape = Dim4::new(input_shape);
        self.filters = randu(Dim4::new(&[
           self.filter_size.0,
           self.filter_size.1,
           self.num_filters,
           1
        ]));
    }

    fn output_shape(&self) -> [u64; 4] {
        let mut output_shape = self.input_shape.get().clone();
        if !self.flatten {
            output_shape[3] = self.num_filters;
            output_shape
        }
        else {
            let size = output_shape.iter().fold(1, |acc, x| acc * x);
            [1, size, 1, 1]
        }
    }

    fn serialize(&self) -> String {
        unimplemented!();
    }


    fn a_function(&self) -> Activation {
        self.a_function
    }

    fn display(&self) {
        let filters = &self.filters;
        af_print!("Kernel: ", filters);
    }
}


impl Conv {
    pub fn new(num_filters: u64, filter_size: (u64, u64)) -> Box<Self> {
        // temporary weights
        let filters = Matrix::new(&[1.0], Dim4::new(&[1,1,1,1]));

        use crate::layers::activations::{relu, drelu};
        Box::new(Self { filters, filter_size, num_filters,
            input_shape: Dim4::new(&[1,1,1,1]),
            a_function: Activation::Relu,
            flatten: false,
            activation: Box::new(relu), dactivation: Box::new(drelu) })
    }


    pub fn flatten(mut self, flatten: bool) -> Box<Self> {
        self.flatten = flatten;
        Box::new(self)
    }

    pub fn deserialize(_src: &[&str]) -> Box<Self> {
        unimplemented!();
        // let size = src[0]
        //     .split(',')
        //     .map(|v| v.parse::<Weight>().unwrap())
        //     .collect::<Vec<_>>();
        // let inputs  = size[0];
        // let neurons = size[1];

        // let activation = Activation::from(src[1]);

        // let weights: Vec<_> = src[2]
        //     .split(',')
        //     .map(|v| v.parse())
        //     .take_while(|v| v.is_ok())
        //     .collect::<Result<_,_>>()
        //     .unwrap();

        // let biases: Vec<_> = src[3]
        //     .split(',')
        //     .map(|v| v.parse())
        //     .take_while(|v| v.is_ok())
        //     .collect::<Result<_,_>>()
        //     .unwrap();

        // let mut layer = Conv::new(neurons as u64);
        // layer.weights = Matrix::new(&weights, Dim4::new(&[inputs as u64, neurons as u64, 1, 1]));
        // layer.biases = Matrix::new(&biases, Dim4::new(&[1, neurons as u64, 1, 1]));
        // layer.activation(activation)
    }

    pub fn activation(mut self, activation: Activation) -> Box<Self> {
        use crate::layers::activations::{
            relu, drelu,
            tanh, dtanh,
            sigmoid, dsigmoid,
            softmax, dsoftmax,
            elu, delu,
        };

        self.a_function = activation;

        match activation {
            Activation::Relu => {
                self.activation = Box::new(relu);
                self.dactivation = Box::new(drelu);
            },
            Activation::Tanh => {
                self.activation = Box::new(tanh);
                self.dactivation = Box::new(dtanh);
            },
            Activation::Sigmoid => {
                self.activation = Box::new(sigmoid);
                self.dactivation = Box::new(dsigmoid);
            },
            Activation::Softmax => {
                self.activation = Box::new(softmax);
                self.dactivation = Box::new(dsoftmax);
            },
            Activation::Elu => {
                self.activation = Box::new(elu);
                self.dactivation = Box::new(delu);
            },
        }

        Box::new(self)
    }
}

