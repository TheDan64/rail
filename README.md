# R.A.I.L: A Rust Artificial Intelligence Library
RAIL is designed to be a library for easily creating and training Neural Networks,
akin to the Keras API. It aims to be fast and easy to use.

### Dependencies
RAIL depends on [arrayfire-rust](https://github.com/arrayfire/arrayfire-rust),
so before using RAIL make sure you have arrayfire installed.

### A Simple XOR Problem
Solving the XOR Problem with Mold is super easy! Simply add the crate to your
Cargo.toml:
```toml
rail = { git = "https://github.com/nlsnightmare/rail" }
```
Then add this to your code
```rust
use rail::model::Model;
use rail::layers::dense::Dense;
use rail::layers::activations::Activation;

pub fn main() {
    let mut model = Model::new()
        .learning_rate(0.01)
        .input_size(2)
        .layer(Dense::new(2).activation(Activation::Tanh))
        .layer(Dense::new(1).activation(Activation::Tanh))
        .build(true)
        .unwrap();

    let tranining_data = vec![
        (vec![0., 0.], vec![0.]),
        (vec![0., 1.], vec![1.]),
        (vec![1., 0.], vec![1.]),
        (vec![1., 1.], vec![0.]),
    ];

    // Train with a batch of 2 for 4000 epochs
    model.train(&tranining_data, 2, 4000);

    println!("[0, 0] -> {}", model.predict(vec![0., 0.])[0]); // should be close to 0
    println!("[0, 1] -> {}", model.predict(vec![0., 1.])[0]); // should be close to 1
    println!("[1, 0] -> {}", model.predict(vec![1., 0.])[0]); // should be close to 1
    println!("[1, 1] -> {}", model.predict(vec![1., 1.])[0]); // should be close to 0
}
```

### Plans
As of now, RAIL is in a very early state, and under heavy development.
The API _will_ change a lot.<br />
So far, only Dense (aka fully connected) layers are supported, and batched SGD is the only way of training the network.
However, there are plans to support:
- Convolutional Layers
- RNN Cells
- LSTM Cells
- Genetic Crossover
- ADAM optimizer
- More Activation functions
- More Error functions
- Documentation
