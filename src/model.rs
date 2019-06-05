use crate::{ Weight, Matrix };
use crate::layers::Layer;
use crate::utils::array_data;
use crate::functions::loss::{ 
    LossFuntion,
    DLossFuntion,
    LossFuntionGenerator,
    mean_squared_error
};

use arrayfire::{
    Array,
    Dim4,
};

type TrainingData = Vec<(Vec<Weight>, Vec<Weight>)>;

pub enum ModelError {
    NoLayers,
    NoInputShape,
    NoOutputShape,
    NoLearningRate,
    NoLossFuntion,
}


use std::fmt;
impl fmt::Debug for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelError::NoLayers => 
                write!(f, "Model has no Layers!"),
            ModelError::NoInputShape => 
                write!(f, "Model has no Input Shape!"),
            ModelError::NoOutputShape => 
                write!(f, "Model has no Output Shape!"),
            ModelError::NoLearningRate => 
                write!(f, "Model has no Learning Rate!"),
            ModelError::NoLossFuntion => 
                write!(f, "Model has no Loss Function!"),
        }
    }
}

pub struct Model {
    input_shape: [u64; 4],
    output_shape: [u64; 4],
    layers: Vec<Box<Layer>>,
    loss: Box<LossFuntion>,
    d_loss: Box<DLossFuntion>,
    lr: Weight
}

pub struct ModelBuilder {
    input_shape: Option<[u64; 4]>,
    output_shape: Option<[u64; 4]>,
    layers: Vec<Box<Layer>>,
    loss: Box<LossFuntion>,
    d_loss: Box<DLossFuntion>,
    lr: Option<Weight>
}

impl ModelBuilder {
    pub fn layer(mut self, layer: Box<Layer>) -> Self {
        self.layers.push(layer);
        self
    }

    pub fn loss(mut self, loss_fn: Box<LossFuntionGenerator>) -> Self {
        let (loss, d_loss) = loss_fn();
        self.loss = loss;
        self.d_loss = d_loss;
        self
    }

    pub fn learning_rate(mut self, lr: Weight) -> Self {
        self.lr = Some(lr);
        self
    }

    pub fn input_size(mut self, amount: u64) -> Self {
        self.input_shape = Some([1, amount, 1, 1]);
        self
    }
    pub fn input_shape(mut self, shape: &[u64; 4]) -> Self {
        self.input_shape = Some(shape.clone());
        self
    }

    pub fn output_shape(mut self, shape: &[u64; 4]) -> Self {
        self.output_shape = Some(shape.clone());
        self
    }

    pub fn build(mut self, init: bool) -> Result<Model, ModelError> {
        let input_shape = self.input_shape
            .ok_or(ModelError::NoInputShape)?;
        let output_shape = self.layers
            .last()
            .ok_or(ModelError::NoLayers)?
            .output_shape();

        let loss = self.loss;
        let d_loss = self.d_loss;

        let lr = self.lr
            .ok_or(ModelError::NoLearningRate)?;

        let mut model = Model {
            loss, d_loss,
            input_shape: input_shape, 
            output_shape: output_shape, 
            layers: self.layers,
            lr
        };
        

        if init {
            model.initialize_weights();
        }

        Ok(model)
    }
}

impl Model {
    pub fn new() -> ModelBuilder {
        let (loss, d_loss) = mean_squared_error();
        ModelBuilder {
            input_shape: None, 
            output_shape: None,
            loss, d_loss,
            layers: Vec::new(),
            lr: None
        }
    }

    pub fn display(&self) {
        for layer in &self.layers {
            layer.display();
        }
    }

    pub fn output_shape(&self) -> [u64; 4] {
        self.output_shape
    }

    pub fn predict(&self, input: Vec<Weight>) -> Vec<Weight> {
        let amount_per_input = self.input_shape
            .iter()
            .fold(1, |acc, x| acc * x);
        let amount = input.len() as u64 / amount_per_input;

        // TODO: make sure this works
        // 1d inputs should be mapped to 2d,
        // whereas 2d or more should be mapped to 3d to allow
        // 3 channel batch convolutions
        let dims = match self.input_shape[1] {
            1 => Dim4::new(&[self.input_shape[0], self.input_shape[1], amount, 1]),
            _ => Dim4::new(&[self.input_shape[0], self.input_shape[1], self.input_shape[2], amount])
        };

        let mut xs = Array::new(&input, dims);
        for layer in &self.layers {
            xs = layer.feedforward(xs);
        }

        array_data(xs)
    }

    fn batch_data(&self, mut data: TrainingData, batch_size: usize) -> Vec<(Matrix, Matrix)> {
        let elements_per_input = self.input_shape
            .iter()
            .fold(1, |acc, x| acc * x) as usize;
        let elements_per_output = self.output_shape
            .iter()
            .fold(1, |acc, x| acc * x) as usize;

        let num_inputs  = batch_size * elements_per_input;
        let num_outputs = batch_size * elements_per_output;
        let num_batches = data.len() / batch_size;

        let mut input_buf =  Vec::with_capacity(num_inputs);
        let mut output_buf = Vec::with_capacity(num_outputs);
        let mut result_buf = Vec::with_capacity(num_batches);

        while let Some(d) = data.pop() {
            input_buf.extend(d.0);
            output_buf.extend(d.1);
            if input_buf.len() == num_inputs {
                let w = input_buf.len() / elements_per_input;
                let dims = self.generate_dims(&self.input_shape, w);
                let input_mat = Matrix::new(&input_buf, dims);
                let dims = self.generate_dims(&self.output_shape, w);
                let output_mat = Matrix::new(&output_buf, dims);
                result_buf.push((input_mat, output_mat));
                input_buf = Vec::new();
                output_buf = Vec::new();
            }
        }

        result_buf
    }

    // TODO: find better name, move to utils
    fn generate_dims(&self, shape: &[u64; 4], amount: usize) -> Dim4 {
        let amount = amount as u64;
        match shape {
            [_, _, 1, _] => Dim4::new(&[shape[0], shape[1], amount, 1]),
            [_, _, _, 1] => Dim4::new(&[shape[0], shape[1], shape[2], amount]),
            [_, _, _, _] => panic!("unrecognized shape! {:?}", shape)
        }
    }

    pub fn train(&mut self, data: &TrainingData, batch_size: usize, epochs: usize) {
        use std::time::Instant;
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let mut data = self.batch_data(data.clone(), batch_size);

        println!("training with batch size {} for {} epochs", batch_size, epochs);
        for epoch in 1..epochs+1 {
            let now = Instant::now();
            data.shuffle(&mut rng);
            let error = data
                .iter()
                .map(|(inputs, outputs)| self.backpropagate_mat(&inputs, &outputs).abs())
                .sum::<Weight>() / data.len() as f64;


            println!("Epoch {:3}. Avg Error: {:1.4}. Time elapsed: {:.2}s",
                     epoch, error, now.elapsed().as_millis() as f32 / 1000.0);

             arrayfire::device_gc();
        }


        println!("done");
    }


    pub fn backpropagate_mat(&mut self, input: &Matrix, expected: &Matrix) -> Weight {
        // get batch size
        let batch_size = input.dims()[2] as f64;
        let mut xs = input.clone();

        // Feedforward and caching of values
        let mut x_cache = Vec::new();
        let mut y_cache = Vec::new();
        for layer in &self.layers {
            x_cache.push(xs.clone());
            let ys = layer.feedforward(xs);
            y_cache.push(ys.clone());

            xs = ys;
        }

        // Error calculation
        let error = (self.loss)(&xs, &expected);
        let mut d_error = (self.d_loss)(&xs, expected);
        // println!("error: {}", error);
        // println!("fixed: {}\n", error / batch_size);

        for ((layer, ys), xs) in self.layers.iter_mut().zip(y_cache.iter()).zip(x_cache.iter()).rev() {
            d_error = layer.backpropagate(xs, ys, d_error, self.lr);
        }

        error / batch_size

    }

    pub fn backpropagate(&mut self, input: &Vec<Weight>, expected: &Vec<Weight>) -> Weight {
        let amount_operations = input.len() as u64 / self.input_shape
            .iter()
            .fold(1, |acc, x| acc * x);

        let mut expected_dims = self.output_shape;
        let mut input_dims = self.input_shape;

        if self.input_shape[1] == 1 {
            input_dims[1] = amount_operations;
        } else {
            input_dims[3] = amount_operations;
        }

        if self.output_shape[1] == 1 {
            expected_dims[1] = amount_operations;
        } else {
            expected_dims[3] = amount_operations;
        }


        let xs = Array::new(&input, Dim4::new(&input_dims));
        let expected_mat = Array::new(&expected, Dim4::new(&expected_dims));
        self.backpropagate_mat(&xs, &expected_mat)
    }


    pub fn load(filename: &str) -> Result<Self, String> {
        use std::fs::File;
        use std::io::Read;

        use crate::layers::load_layer;

        let mut file = File::open(filename)
            .map_err(|_| String::from("unable to open file!"))
            .unwrap();

        let mut buf = String::with_capacity(4000);
        file.read_to_string(&mut buf)
            .map_err(|_| String::from("unable to load file!"))
            .unwrap();

        let mut lines = buf.split('\n').collect::<Vec<_>>();

        let buf = lines[0]
            .split(',')
            .map(|v| v.parse::<Weight>().expect("error in line 0"))
            .collect::<Vec<_>>();

        let mut nn = Model::new()
            .learning_rate(buf[2]);

        let mut i = 1;
        while i < lines.len() - 1 {
            let l = load_layer(&lines[i..(i + 6)]);
            nn = nn.layer(l);
            
            i+= 5;
        }

        // nn.build()
        unimplemented!();
    }


    pub fn save_weights(&self, filename: &str) {
        unimplemented!();
        use std::fs::File;
        use std::io::Write;

        let mut fd = File::create(filename).expect("unable to create file!");
        let mut buf = String::with_capacity(2000);

        // FIXME: 
        // buf.push_str(&format!("{},{},{}\n", self.inputs, self.outputs, self.lr));
        for layer in &self.layers {
            buf.push_str(&layer.serialize());
        }

        fd.write(buf.as_bytes()).expect("unable to write to file!");
    }

    pub fn initialize_weights(&mut self) {
        let mut input_shape = self.input_shape;
        for layer in self.layers.iter_mut() {
            layer.initialize(&input_shape);
            input_shape = layer.output_shape();
        }
    }
}
