//! This program implements a simple neural network to classify handwritten digits from the MNIST dataset.
//! It uses a two-layer network with ReLU activation in the hidden layer and softmax in the output layer.

use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};

/// Input layer size (28x28 pixels)
const INPUT_SIZE: usize = 784;
/// Hidden layer size
const HIDDEN_SIZE: usize = 256;
/// Output layer size (10 digits)
const OUTPUT_SIZE: usize = 10;
/// Learning rate for gradient descent
const LEARNING_RATE: f32 = 0.001;
/// Number of training epochs
const EPOCHS: usize = 20;
/// Size of each training batch
const BATCH_SIZE: usize = 64;
/// Size of each image (28x28)
const IMAGE_SIZE: usize = 28;
/// Proportion of data used for training (vs. testing)
const TRAIN_SPLIT: f32 = 0.8;

/// Path to the MNIST training images file
const TRAIN_IMG_PATH: &str = "data/images";
/// Path to the MNIST training labels file
const TRAIN_LBL_PATH: &str = "data/labels";

/// Represents a layer in the neural network
struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

/// Represents the entire neural network
struct Network {
    hidden: Layer,
    output: Layer,
}

/// Applies the softmax function to a slice of floats
fn softmax(input: &mut [f32]) {
    let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = input
        .iter_mut()
        .map(|x| {
            *x = (*x - max).exp();
            *x
        })
        .sum();
    input.iter_mut().for_each(|x| *x /= sum);
}

impl Layer {
    /// Creates a new layer with random weights
    fn new(in_size: usize, out_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / in_size as f32).sqrt();
        let between = Uniform::from(-scale..scale);

        Layer {
            weights: (0..in_size * out_size)
                .map(|_| between.sample(&mut rng))
                .collect(),
            biases: vec![0.0; out_size],
            input_size: in_size,
            output_size: out_size,
        }
    }

    /// Performs forward propagation through the layer
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        for i in 0..self.output_size {
            output[i] = self.biases[i]
                + input
                    .iter()
                    .zip(self.weights[i..].iter().step_by(self.output_size))
                    .map(|(&x, &w)| x * w)
                    .sum::<f32>();
        }
    }

    /// Performs backward propagation through the layer and updates weights and biases
    /// This implements the Stochastic Gradient Descent (SGD) optimizer
    /// Learn more about SGD: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    fn backward(
        &mut self,
        input: &[f32],
        output_grad: &[f32],
        mut input_grad: Option<&mut [f32]>,
        lr: f32,
    ) {
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                let idx = j * self.output_size + i;
                let grad = output_grad[i] * input[j];
                self.weights[idx] -= lr * grad;
                if let Some(ref mut ig) = input_grad {
                    ig[j] += output_grad[i] * self.weights[idx];
                }
            }
            self.biases[i] -= lr * output_grad[i];
        }
    }
}

impl Network {
    /// Creates a new neural network
    fn new() -> Self {
        Network {
            hidden: Layer::new(INPUT_SIZE, HIDDEN_SIZE),
            output: Layer::new(HIDDEN_SIZE, OUTPUT_SIZE),
        }
    }

    /// Trains the network on a single input
    fn train(&mut self, input: &[f32], label: u8, lr: f32) {
        let mut hidden_output = vec![0.0; HIDDEN_SIZE];
        let mut final_output = vec![0.0; OUTPUT_SIZE];
        let mut output_grad = vec![0.0; OUTPUT_SIZE];
        let mut hidden_grad = vec![0.0; HIDDEN_SIZE];

        // Forward pass
        self.hidden.forward(input, &mut hidden_output);
        hidden_output.iter_mut().for_each(|x| *x = x.max(0.0)); // ReLU activation

        self.output.forward(&hidden_output, &mut final_output);
        softmax(&mut final_output);

        // Compute gradients
        for i in 0..OUTPUT_SIZE {
            output_grad[i] = final_output[i] - if i == label as usize { 1.0 } else { 0.0 };
        }

        // Backward pass
        self.output
            .backward(&hidden_output, &output_grad, Some(&mut hidden_grad), lr);

        for i in 0..HIDDEN_SIZE {
            hidden_grad[i] *= if hidden_output[i] > 0.0 { 1.0 } else { 0.0 }; // ReLU derivative
        }

        self.hidden.backward(input, &hidden_grad, None, lr);
    }

    /// Predicts the digit for a given input
    fn predict(&self, input: &[f32]) -> usize {
        let mut hidden_output = vec![0.0; HIDDEN_SIZE];
        let mut final_output = vec![0.0; OUTPUT_SIZE];

        self.hidden.forward(input, &mut hidden_output);
        hidden_output.iter_mut().for_each(|x| *x = x.max(0.0)); // ReLU activation

        self.output.forward(&hidden_output, &mut final_output);
        softmax(&mut final_output);

        final_output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    }
}

/// Reads MNIST images from a file
fn read_mnist_images(filename: &str) -> io::Result<(Vec<u8>, usize)> {
    let mut file = BufReader::new(File::open(filename)?);
    let mut buffer = [0u8; 4];

    file.read_exact(&mut buffer)?;
    file.read_exact(&mut buffer)?;
    let n_images = u32::from_be_bytes(buffer) as usize;

    file.read_exact(&mut buffer)?; // rows
    file.read_exact(&mut buffer)?; // cols

    let mut images = vec![0u8; n_images * IMAGE_SIZE * IMAGE_SIZE];
    file.read_exact(&mut images)?;

    Ok((images, n_images))
}

/// Reads MNIST labels from a file
fn read_mnist_labels(filename: &str) -> io::Result<(Vec<u8>, usize)> {
    let mut file = BufReader::new(File::open(filename)?);
    let mut buffer = [0u8; 4];

    file.read_exact(&mut buffer)?;
    file.read_exact(&mut buffer)?;
    let n_labels = u32::from_be_bytes(buffer) as usize;

    let mut labels = vec![0u8; n_labels];
    file.read_exact(&mut labels)?;

    Ok((labels, n_labels))
}

/// Shuffles the training data and labels
fn shuffle_data(images: &mut [u8], labels: &mut [u8]) {
    let mut rng = rand::thread_rng();
    for i in (1..labels.len()).rev() {
        let j = rng.gen_range(0..=i);
        labels.swap(i, j);
        for k in 0..INPUT_SIZE {
            images.swap(i * INPUT_SIZE + k, j * INPUT_SIZE + k);
        }
    }
}

fn main() -> io::Result<()> {
    let mut net = Network::new();
    let (mut images, n_images) = read_mnist_images(TRAIN_IMG_PATH)?;
    let (mut labels, _) = read_mnist_labels(TRAIN_LBL_PATH)?;

    assert_eq!(n_images, labels.len());

    shuffle_data(&mut images, &mut labels);

    let train_size = (n_images as f32 * TRAIN_SPLIT) as usize;
    let test_size = n_images - train_size;

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0;
        for i in (0..train_size).step_by(BATCH_SIZE) {
            let progress = (i + BATCH_SIZE).min(train_size) as f32 / train_size as f32 * 100.0;
            print!(
                "\rEpoch {}, Batch {}/{}: {:.2}% complete",
                epoch + 1,
                i / BATCH_SIZE + 1,
                train_size / BATCH_SIZE,
                progress
            );
            io::stdout().flush()?;
            for j in 0..BATCH_SIZE.min(train_size - i) {
                let idx = i + j;
                let img: Vec<f32> = images[idx * INPUT_SIZE..(idx + 1) * INPUT_SIZE]
                    .iter()
                    .map(|&x| x as f32 / 255.0)
                    .collect();

                net.train(&img, labels[idx], LEARNING_RATE);

                // Calculate cross-entropy loss
                // Learn more about cross-entropy loss: https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression
                let mut hidden_output = vec![0.0; HIDDEN_SIZE];
                let mut final_output = vec![0.0; OUTPUT_SIZE];
                net.hidden.forward(&img, &mut hidden_output);
                hidden_output.iter_mut().for_each(|x| *x = x.max(0.0)); // ReLU
                net.output.forward(&hidden_output, &mut final_output);
                softmax(&mut final_output);

                total_loss -= (final_output[labels[idx] as usize] + 1e-10).ln();
            }
        }
        print!(", ");

        // Evaluate on test set
        let correct = (train_size..n_images)
            .filter(|&i| {
                let img: Vec<f32> = images[i * INPUT_SIZE..(i + 1) * INPUT_SIZE]
                    .iter()
                    .map(|&x| x as f32 / 255.0)
                    .collect();
                net.predict(&img) == labels[i] as usize
            })
            .count();

        println!(
            "Accuracy: {:.2}%, Avg Loss: {:.4}",
            correct as f32 / test_size as f32 * 100.0,
            total_loss / train_size as f32
        );
    }

    Ok(())
}
