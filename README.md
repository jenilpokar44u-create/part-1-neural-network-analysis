# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Overview
This project builds a feed-forward neural network to predict customer churn based on various features like tenure, monthly charges, and contract type. The goal is to demonstrate the fundamentals of neural network training, including forward passes, backpropagation, and hyperparameter tuning.

## Dataset
The dataset used is `customer_churn_nn.csv`. It contains features like customer demographics, plan types, and billing information. The target variable is `churn` (1 for churned, 0 for retained).

## Project Structure
- `notebook.ipynb`: Contains all the EDA, data preprocessing, model building, and hyperparameter experiments.
- `requirements.txt`: List of dependencies needed to run the notebook.
- `results/`: Contains screenshots of the evaluation metrics and the model comparison table.

## Steps Taken
1. **EDA & Preprocessing**: Checked for missing values, dropped unnecessary ID columns, one-hot encoded categorical variables, and scaled numerical inputs using StandardScaler.
2. **Model Building**: Created a sequential model using TensorFlow/Keras with ReLU activations in the hidden layers and a Sigmoid activation for the binary output.
3. **Experiments**: Tested different hyperparameter configurations (network depth, learning rate, and batch size) to observe their impact on test accuracy. 

## Instructions to Run
1. Install requirements: `pip install -r requirements.txt`
2. Ensure `customer_churn_nn.csv` is in the root directory.
3. Run `notebook.ipynb` cell by cell.
