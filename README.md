# A minimal implementation of neural network for classifying handwritten digits from the [MNIST handwritten digit database](https://yann.lecun.com/exdb/mnist/)  

- **Network Architecture**

    Two-layer network: Hidden layer with 256 neurons and an output layer with 10 neurons (one for each digit).  

    Activation functions: ReLU (Rectified Linear Unit) in the hidden layer and softmax in the output layer.  



- **Training Process**

    Stochastic Gradient Descent (SGD) optimizer is used to update weights and biases.  

    Mini-batch training: Processes data in batches of size 64 for efficiency.  

    Cross-entropy loss function is used to measure the difference between predicted and actual labels.  


- **Data Handling**

    MNIST dataset is used for training, containing images and corresponding labels for handwritten digits.  

    Images are normalized (divided by 255) to bring pixel values between 0 and 1.  

    Data shuffling is performed before training to avoid biases.  


- **Evaluation**

    Accuracy is calculated on the test set (a portion of the data not used for training) after each epoch (training iteration).  

    Average loss is also reported to track how well the model is learning.  


## Prerequisites  
- [rustup.rs - The Rust toolchain installer](https://rustup.rs/)

## Dependencies

This project requires the following Rust libraries:  

`rand`: For random number generation (weights initialization, shuffling data) [rand - crates.io](https://crates.io/crates/rand)  

`std::fs`: For file system access (reading MNIST images and labels)  [std::fs - Rust](https://doc.rust-lang.org/std/fs/index.html)  

`std::io`: For file input/output operations  [std::io - Rust](https://doc.rust-lang.org/std/io/index.html)  

These dependencies are automatically managed by rust's package manager, cargo.  
They will be installed when you build the project.

## usage  
- clone the repository
    ```sh
    git clone https://github.com/thekitp/mnist-rs.git
    ```
- navigate to the project directory
    ```sh
    cd mnist-rs
    ```
- run the following command  
    ```sh
    cargo run --release
    ```

## sample output  
```sh
     Running `target/release/mnist-rs`
Epoch 1, Batch 750/750: 100.00% complete, Accuracy: 90.60%, Avg Loss: 0.4668
Epoch 2, Batch 750/750: 100.00% complete, Accuracy: 92.27%, Avg Loss: 0.2355
Epoch 3, Batch 750/750: 100.00% complete, Accuracy: 93.66%, Avg Loss: 0.1891
Epoch 4, Batch 750/750: 100.00% complete, Accuracy: 94.39%, Avg Loss: 0.1570
Epoch 5, Batch 750/750: 100.00% complete, Accuracy: 94.84%, Avg Loss: 0.1331
Epoch 6, Batch 750/750: 100.00% complete, Accuracy: 95.23%, Avg Loss: 0.1147
Epoch 7, Batch 750/750: 100.00% complete, Accuracy: 95.62%, Avg Loss: 0.1001
Epoch 8, Batch 750/750: 100.00% complete, Accuracy: 95.96%, Avg Loss: 0.0883
Epoch 9, Batch 750/750: 100.00% complete, Accuracy: 96.27%, Avg Loss: 0.0784
Epoch 10, Batch 750/750: 100.00% complete, Accuracy: 96.45%, Avg Loss: 0.0702
Epoch 11, Batch 750/750: 100.00% complete, Accuracy: 96.55%, Avg Loss: 0.0632
Epoch 12, Batch 750/750: 100.00% complete, Accuracy: 96.68%, Avg Loss: 0.0573
Epoch 13, Batch 750/750: 100.00% complete, Accuracy: 96.82%, Avg Loss: 0.0521
Epoch 14, Batch 750/750: 100.00% complete, Accuracy: 96.91%, Avg Loss: 0.0476
Epoch 15, Batch 750/750: 100.00% complete, Accuracy: 96.97%, Avg Loss: 0.0436
Epoch 16, Batch 750/750: 100.00% complete, Accuracy: 97.04%, Avg Loss: 0.0401
Epoch 17, Batch 750/750: 100.00% complete, Accuracy: 97.07%, Avg Loss: 0.0370
Epoch 18, Batch 750/750: 100.00% complete, Accuracy: 97.09%, Avg Loss: 0.0343
Epoch 19, Batch 750/750: 100.00% complete, Accuracy: 97.10%, Avg Loss: 0.0318
Epoch 20, Batch 750/750: 100.00% complete, Accuracy: 97.06%, Avg Loss: 0.0296
```
## Here are some additional points to consider

This is a simplified network. More complex architectures with convolutional layers are typically used for better performance on image recognition tasks.  

Hyperparameters like learning rate, number of epochs, and network size can be tuned for better results.  

Error handling and advanced training techniques can be added for a more robust implementation.  

