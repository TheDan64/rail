use arrayfire::{
    Dim4,
    Array,
    MatProp,
    matmul,
    print_gen
};

use crate::layers::Layer;
use crate::layers::activations::Activation;
use crate::{ Weight, Matrix };

pub struct Dense {
    weights: Matrix,
    biases: Matrix,
    neurons: u64,
    a_function: Activation,
    activation: Box<Fn(Matrix) -> Matrix>,
    dactivation: Box<Fn(Matrix) -> Matrix>
}

impl Layer for Dense {
    fn feedforward(&self, inputs: Matrix) -> Matrix {
        let mut result = matmul(&inputs, &self.weights, MatProp::NONE, MatProp::NONE);
        result = arrayfire::add(&result, &self.biases, true);
        (self.activation)(result)
    }


    fn backpropagate(&mut self, input: &Matrix, output: &Matrix, d_error: Matrix, lr: Weight) -> Matrix {
        let next_errors = matmul(&d_error, &self.weights, MatProp::NONE, MatProp::TRANS);
        next_errors.eval();

        let d_activation = (self.dactivation)(output.clone());
        let gradient = arrayfire::mul(&d_activation, &d_error, true) * lr;
        let mut delta_w = matmul(&input, &gradient, MatProp::TRANS, MatProp::NONE);

        delta_w = arrayfire::sum(&delta_w, 2);
        let delta_b = arrayfire::sum(&gradient, 2);

        delta_w.eval();
        delta_b.eval();
        self.weights += delta_w;
        self.biases += delta_b;

        next_errors
    }

    fn initialize(&mut self, input_shape: &[u64; 4]) {
        use crate::layers::activations::initialize_weights;
        let inputs = input_shape[1];

        let (weight_values, bias_values) =
            initialize_weights(self.a_function, inputs as usize, self.neurons as usize);

        let new_weights = Array::new(
            &weight_values, 
            Dim4::new(&[
                inputs,
                self.neurons,
                1, 1
            ])
        );

        let new_biases = Array::new(&bias_values, Dim4::new(&[1, self.neurons, 1, 1]));

        self.weights = new_weights;
        self.biases = new_biases;
    }

    fn a_function(&self) -> Activation {
        self.a_function
    }

    fn output_shape(&self) -> [u64; 4] {
        [1, self.neurons, 1, 1]
    }


    fn serialize(&self) -> String {
        use crate::utils::array_data;

        let mut buf = String::with_capacity(300);
        let dims = self.weights.dims();
        buf.push_str(&format!("dense\n"));
        buf.push_str(&format!("{},{}\n", dims[0], dims[1]));
        buf.push_str(&format!("{}\n", self.a_function));
        for w in array_data(self.weights.clone()) {
            buf.push_str(&format!("{},", w));
        }

        buf.pop();
        buf.push('\n');

        for b in array_data(self.biases.clone()) {
            buf.push_str(&format!("{},", b));
        }
        buf.push('\n');

        buf
    }

    fn display(&self) {
        let w = &self.weights;
        let b = &self.biases;

        af_print!("neurons:", w);
        af_print!("biases:", b);
        println!("activation: {}", self.a_function);
    }
}

impl Dense {
    pub fn new(neurons: u64) -> Box<Self> {
        // temporary weights
        let weights = Array::new(&[1.0], Dim4::new(&[1,1,1,1]));
        let biases = Array::new(&[1.0], Dim4::new(&[1,1,1,1]));

        use crate::layers::activations::{relu, drelu};
        Box::new(Self { neurons, weights, biases,
            a_function: Activation::Relu,
            activation: Box::new(relu), dactivation: Box::new(drelu) })
    }

    pub fn deserialize(src: &[&str]) -> Box<Self> {
        let size = src[0]
            .split(',')
            .map(|v| v.parse::<Weight>().unwrap())
            .collect::<Vec<_>>();
        let inputs  = size[0];
        let neurons = size[1];

        let activation = Activation::from(src[1]);

        let weights: Vec<_> = src[2]
            .split(',')
            .map(|v| v.parse())
            .take_while(|v| v.is_ok())
            .collect::<Result<_,_>>()
            .unwrap();

        let biases: Vec<_> = src[3]
            .split(',')
            .map(|v| v.parse())
            .take_while(|v| v.is_ok())
            .collect::<Result<_,_>>()
            .unwrap();

        let mut layer = Dense::new(neurons as u64);
        layer.weights = Matrix::new(&weights, Dim4::new(&[inputs as u64, neurons as u64, 1, 1]));
        layer.biases = Matrix::new(&biases, Dim4::new(&[1, neurons as u64, 1, 1]));
        layer.activation(activation)
    }

    pub fn activation(mut self, activation: Activation) -> Box<Self> {
        use crate::layers::activations::{
            relu, drelu,
            tanh, dtanh,
            sigmoid, dsigmoid, 
            softmax, dsoftmax,
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
            }
        }


        Box::new(self)
    }
}
