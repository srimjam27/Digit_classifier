# Digit Classifier using Deep Neural Network

## Introduction
This project implements a Deep Neural Network (DNN) to classify handwritten digits from the MNIST dataset using TensorFlow and Keras. The model architecture is designed to classify images of size 28x28 pixels into one of the 10 digit classes (0-9).

## Model Architecture
The DNN model architecture used for this classification task is as follows:

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  # Input layer: Flattens the 28x28 image into a 1D array
    keras.layers.Dense(100, activation='relu'),  # Hidden layer with 100 neurons and ReLU activation
    keras.layers.Dense(10, activation='sigmoid')  # Output layer with 10 neurons (one for each digit) and sigmoid activation
])
```

### Model Summary:
- **Input Layer**: Flatten layer transforms the 2D image (28x28 pixels) into a 1D array (784 pixels).
- **Hidden Layer**: Dense layer with 100 neurons and ReLU activation function to learn complex representations.
- **Output Layer**: Dense layer with 10 neurons and sigmoid activation function for multi-class classification (one neuron per digit).

### Compilation:
- **Loss Function**: Sparse categorical crossentropy (suitable for integer-coded labels).
- **Optimizer**: Adam optimizer, a popular choice for deep learning tasks.
- **Metrics**: Accuracy metric to evaluate model performance during training and testing.

## Dataset
The model is trained on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is grayscale and normalized to have pixel values between 0 and 1.

## Training
The model is trained using the training set with 10 epochs and a batch size of 32. During training, the model learns to classify digits by adjusting its weights and biases through backpropagation and gradient descent.

## Evaluation
The performance of the model is evaluated using the test set to measure accuracy and other relevant metrics. This ensures that the model generalizes well to unseen data and performs robustly in real-world scenarios.

## Usage
1. **Data Preparation**:
   - Ensure your input images are preprocessed to match the model's input shape (28x28 pixels).

2. **Model Training**:
   - Use the provided script or Jupyter Notebook to train the model on the MNIST dataset.

3. **Model Evaluation**:
   - Evaluate the trained model on the test dataset to assess its accuracy and performance metrics.

4. **Inference**:
   - Use the trained model to predict digits from new input images.

## Future Improvements
- Experiment with different architectures (e.g., convolutional neural networks) for enhanced performance.
- Fine-tune hyperparameters such as learning rate, batch size, and number of neurons for optimal results.
- Implement data augmentation techniques to increase the diversity of training data.

## Author
- [Your Name]

## References
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

