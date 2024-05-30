# Pattern Recognition Tasks - Semester Project

## Overview
This repository contains the implementation and documentation for 9 pattern recognition tasks assigned during the AI course in the current semester. Each task focuses on different aspects of pattern recognition, utilizing various algorithms and techniques. The tasks cover a range of topics including image processing, signal analysis, and machine learning.

## Table of Contents
- [Tasks Overview](#tasks-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Task Details](#task-details)
  - [Task 1: Gradient Descent Optimizer](#task-1-gradient-descent-optimizer)
  - [Task 2: Linear Regression for House Price Prediction](#task-2-linear-regression-for-house-price-prediction)
  - [Task 3: Spam Email Detection (Binary Classification)](#task-3-spam-email-detection-binary-classification)
  - [Task 4: Handwritten Digit Recognition (Multiclass Classification)](#task-4-handwritten-digit-recognition-multiclass-classification)
  - [Task 5: Multilabel Classification](#task-5-multilabel-classification)
  - [Task 6: Deep Neural Networks for Handwritten Digit Recognition](#task-6-deep-neural-networks-for-handwritten-digit-recognition)
  - [Task 7: Convolutional Neural Networks with LeNet-5](#task-7-convolutional-neural-networks-with-lenet-5)
  - [Task 8: Recurrent Neural Networks for Sentiment Analysis](#task-8-recurrent-neural-networks-for-sentiment-analysis)
  - [Task 9: Machine Translation using RNN Seq2Seq Modeling](#task-9-machine-translation-using-rnn-seq2seq-modeling)
- [Contributing](#contributing)
- [License](#license)

## Tasks Overview
This project consists of nine distinct tasks that collectively cover a broad spectrum of pattern recognition techniques:

1. **Gradient Descent Optimizer**: Implementing a gradient descent optimizer from scratch using Python and numpy.
2. **Linear Regression for House Price Prediction**: Predicting house prices using linear regression with both numpy and Keras implementations.
3. **Spam Email Detection (Binary Classification)**: Classifying emails as spam or not using a binary classification model.
4. **Handwritten Digit Recognition (Multiclass Classification)**: Recognizing handwritten digits using multiclass classification.
5. **Multilabel Classification**: Classifying documents into multiple topics using multilabel classification.
6. **Deep Neural Networks for Handwritten Digit Recognition**: Implementing deep feedforward neural networks for handwritten digit recognition.
7. **Convolutional Neural Networks with LeNet-5**: Implementing the LeNet-5 network for digit recognition.
8. **Recurrent Neural Networks for Sentiment Analysis**: Classifying the sentiment of movie reviews using RNN.
9. **Machine Translation using RNN Seq2Seq Modeling**: Implementing a neural machine translation algorithm using RNN with attention.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.8 or higher
- Jupyter Notebook
- Git


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pattern-recognition-tasks.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pattern-recognition-tasks
   ```

## Usage
Each task is contained within its own directory. To run a task, navigate to the respective directory and execute the Jupyter Notebook associated with that task. For example, to run Task 1:
```bash
cd Task1_Gradient_Descent_Optimizer
jupyter notebook Gradient_Descent_Optimizer.ipynb
```

## Task Details

### Task 1: Gradient Descent Optimizer
- **Description**: Implement a gradient descent optimizer from scratch using Python and numpy.
- **Techniques**: Gradient Descent, Optimization.
- **Libraries**: numpy.

### Task 2: Linear Regression for House Price Prediction
- **Description**: Predict house prices using linear regression.
- **Datasets**: tf.keras.datasets.boston_housing.load_data
- **Techniques**: Linear Regression, Gradient Descent.
- **Libraries**: numpy, Keras.

### Task 3: Spam Email Detection (Binary Classification)
- **Description**: Classify emails as spam or not using a binary classification model.
- **Datasets**: [UCI Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase)
- **Techniques**: Logistic Regression, Naive Bayes, Support Vector Machines.
- **Libraries**: numpy, Scikit-learn.

### Task 4: Handwritten Digit Recognition (Multiclass Classification)
- **Description**: Recognize handwritten digits using multiclass classification.
- **Datasets**: [UCI Optical Recognition of Handwritten Digits Dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- **Techniques**: Multiclass Classification, Support Vector Machines, Neural Networks.
- **Libraries**: numpy, Keras.

### Task 5: Multilabel Classification
- **Description**: Classify documents into multiple topics.
- **Datasets**: [Kaggle Multilabel Classification Dataset](https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset/data)
- **Techniques**: Multilabel Classification, Neural Networks.
- **Libraries**: numpy, Keras.

### Task 6: Deep Neural Networks for Handwritten Digit Recognition
- **Description**: Recognize handwritten digits using deep feedforward neural networks.
- **Datasets**: [UCI Optical Recognition of Handwritten Digits Dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- **Techniques**: Deep Learning, Neural Networks.
- **Libraries**: numpy, Keras.

### Task 7: Convolutional Neural Networks with LeNet-5
- **Description**: Implement the LeNet-5 network for digit recognition.
- **Datasets**: MNIST Dataset.
- **Techniques**: Convolutional Neural Networks (CNNs).
- **Libraries**: Keras.

### Task 8: Recurrent Neural Networks for Sentiment Analysis
- **Description**: Classify the sentiment of movie reviews using RNN.
- **Datasets**: tf.keras.datasets.imdb.load_data
- **Techniques**: Recurrent Neural Networks (RNNs), LSTM.
- **Libraries**: Keras.

### Task 9: Machine Translation using RNN Seq2Seq Modeling
- **Description**: Implement a neural machine translation algorithm using RNN with attention.
- **Datasets**: [AR-EN Translation Dataset](https://github.com/SamirMoustafa/nmt-with-attention-for-ar-to-en/blob/master/ara_.txt)
- **Techniques**: Sequence-to-Sequence (Seq2Seq) Modeling, Attention Mechanisms.
- **Libraries**: Keras.

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
