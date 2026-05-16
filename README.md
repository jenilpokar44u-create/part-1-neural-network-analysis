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

Task 6: Final Reflection
What role do weights and biases play in the model?
Basically, weights determine how much importance the network gives to a specific input feature. For example, the model probably assigns a larger weight to a feature like monthly_charges than it does to region when trying to predict if someone will churn. Biases act like the y-intercept in a math equation—they allow the activation function to shift left or right, making sure the model can still output a baseline prediction even if all the input features happen to be zero.

Why is an activation function required?
Without activation functions, a neural network is essentially just doing basic linear regression, no matter how many hidden layers you stack together. By adding activation functions like ReLU, we introduce non-linearity. This is what actually allows the model to learn complex, messy, real-world patterns in the customer data instead of just trying to draw straight lines through everything.

What happens when the learning rate is too high or too low?
If the learning rate is too low, the model takes tiny steps and might take way too many epochs to learn anything useful. If it's too high, it updates the weights way too aggressively and overshoots the best solution. I actually saw this happen in my own hyperparameter tuning: Experiment 3 used a high learning rate (0.01) and ended up with the worst test accuracy of the bunch (75.25%) because it likely bounced around and missed the optimal minimum.

Did your model show signs of underfitting or overfitting? Explain.
Yes, looking at my evaluation_outputs.png graphs, the model definitely shows signs of slight overfitting. In the loss graph, the training loss keeps dropping nicely all the way to 50 epochs, but the validation loss essentially flatlines and stops improving relatively early on. This means the model started memorizing the training dataset rather than actually learning general patterns that apply to new, unseen data.
