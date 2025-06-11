# Solving the Cauchy-Euler Differential Equation using Physics-Informed Neural Networks (PINNs)

## Table of Contents
1.  [Project Overview](#1-project-overview)
2.  [Mathematical Formulation](#2-mathematical-formulation)
3.  [PINN Design & Architecture](#3-pinn-design--architecture)
4.  [Training Process](#4-training-process)
5.  [Dataset Generation](#5-dataset-generation)
6.  [Implementation Details](#6-implementation-details)
7.  [Results & Analysis](#7-results--analysis)
8.  [Challenges Encountered](#8-challenges-encountered)
9.  [Limitations](#9-limitations)
10. [Future Improvements](#10-future-improvements)
11. [How to Run the Project](#11-how-to-run-the-project)
12. [File Structure](#12-file-structure)
13. [License](#13-license)
14. [Contact](#14-contact)

---

## 1. Project Overview

This project explores the application of **Physics-Informed Neural Networks (PINNs)** to solve the second-order Cauchy-Euler differential equation. The Cauchy-Euler equation is a common differential equation encountered in various scientific and engineering fields, often posing challenges for traditional numerical methods, especially with complex boundary conditions or non-linear variations.

PINNs offer a promising alternative by directly incorporating the underlying physics of the differential equation into the neural network's learning process. This approach can lead to more accurate and robust solutions, particularly in scenarios where traditional methods might struggle.

The project demonstrates two primary approaches:
*   **Standard Neural Network (NN) Regression:** Training a neural network purely on a dataset of `(x, y)` pairs generated from a numerical solver (Assignment 1).
*   **Physics-Informed Neural Network (PINN):** Training a neural network by combining a data-driven loss (regression) with a physics-informed loss (derived from the differential equation's residual), using analytically generated data (Assignment 2).

The performance of both models is evaluated and compared against the true analytical solution.

---

## 2. Mathematical Formulation

The Cauchy-Euler equation is a second-order linear homogeneous differential equation with variable coefficients, having the general form:

$$
x^2 \frac{d^2 y}{dx^2} + \alpha x \frac{dy}{dx} + \beta y = 0
$$

Here, $\alpha$ and $\beta$ are constants.

For the specific form used in this project, which results in complex roots, the analytical solution is:

$$
y = x^{-0.5} (c_1 \cdot \cos(b \cdot \ln(x))) + (c_2 \cdot \sin(b \cdot \ln(x)))
$$

This analytical solution serves as the ground truth for generating synthetic training data and as a benchmark for evaluating model performance. Traditional methods often involve transforming this equation into a constant-coefficient form, which can then be solved using standard techniques.

---

## 3. PINN Design & Architecture

A fully connected feedforward neural network (`NeuralNet` class) is employed for both the standard Neural Network and the PINN. The architecture consists of:

*   **Input Layer:** 1 neuron (for the input variable `x`)
*   **Hidden Layer 1:** 64 neurons with Tanh activation function
*   **Hidden Layer 2:** 64 neurons with Tanh activation function
*   **Output Layer:** 1 neuron (for the predicted output `y`)

The Tanh activation function is chosen for its smooth, non-linear properties, which are beneficial for learning complex relationships in differential equations.

---

## 4. Training Process

The training process differs significantly between the standard NN and the PINN.

### Standard Neural Network (NN) Training (Assignment 1)
*   **Optimizer:** Adam optimizer
*   **Learning Rate:** `0.001`
*   **Loss Function:** Mean Squared Error (MSE) between predicted `y` values and actual `y` values.
*   **Epochs:** 5000 epochs.

### Physics-Informed Neural Network (PINN) Training (Assignment 2)
The PINN combines two types of loss:
1.  **Data Loss (`data_loss`):** Mean Squared Error (MSE) between the network's predicted `y` values and the ground truth `y` values from the dataset.
2.  **Physics Loss (`physics_loss`):** Mean Squared Error (MSE) of the Cauchy-Euler equation's residual. This term ensures the network's predictions satisfy the differential equation. The derivatives `dy/dx` and `d²y/dx²` are computed using PyTorch's `autograd` capabilities.

The total loss for the PINN is a weighted sum of these two losses:
$$
\text{Total Loss} = \text{Data Loss} + \lambda_{\text{physics}} \cdot \text{Physics Loss}
$$

*   **Optimizer:** Adam optimizer
*   **Learning Rate:** `3e-4`
*   **Epochs:** 4000 epochs.
*   **Weighting Factor (`lambda_physics`):**
    *   Starts at `0.0001`.
    *   Gradually increases by `0.0001` every `500` epochs (starting after epoch 0). This strategy allows the model to initially focus on fitting the data and then gradually emphasizes adherence to the underlying physical laws as training progresses.

---

## 5. Dataset Generation

Two distinct datasets were used for the two assignments:

### Dataset 1 (for `Coding_Assignment1_PINNs.ipynb`)
*   **Source:** Generated using MATLAB's `ode45` numerical solver.
*   **Cauchy-Euler Constants:** $\alpha = 0.01$ and $\beta = 0.44$.
*   **Initial Conditions (for `ode45`):** $y(0) = 1$ and $dy/dx(0) = 0$.
*   **Data Points:** 1000 data points for the input variable `x`.
*   **Normalization:** `y` values are normalized using min-max scaling (0 to 1) within the `CauchyEulerDataset` class.

### Dataset 2 (for `Coding_Assignment_2_Atharv_Priyadarshi_23117034.ipynb`)
*   **Source:** Synthetic data generated directly within the Jupyter Notebook environment using the analytical solution of the Cauchy-Euler equation.
*   **Analytical Solution Parameters:**
    *   $a = -0.5$
    *   $b = 0.5$
    *   $c_1 = 1.0$
    *   $c_2 = 0.5$
*   **Cauchy-Euler Equation Constants (derived from analytical solution):** $\alpha = 2.0$ and $\beta = 1.0$.
*   **Input Range:** `x` values are linearly spaced from `0.1` to `2.0`.
*   **Data Points:** 1000 data points.
*   **Noise:** A Gaussian noise factor with a standard deviation of `0.01` was added to the generated `y` values to simulate real-world measurement errors.

The primary assumption is that the analytical solution accurately represents the underlying physics of the system being modeled. The addition of noise to the training data simulates real-world scenarios where measurements are subject to error.

---

## 6. Implementation Details

### Libraries Used
The project utilizes the following Python libraries:
*   **`torch`** (PyTorch): For building and training neural networks.
*   **`torch.nn`**: Neural network modules.
*   **`torch.optim`**: Optimization algorithms (Adam).
*   **`torch.autograd`**: Automatic differentiation (essential for PINNs).
*   **`torch.utils.data`**: Utilities for dataset and dataloader management (used in Assignment 1).
*   **`numpy`**: For numerical operations, especially data handling.
*   **`matplotlib.pyplot`**: For plotting and visualization.
*   **`scipy.io`**: For loading MATLAB `.mat` files (used in Assignment 1).
*   **`sklearn.metrics`**: For calculating performance metrics like MSE, MAE, R-squared (used in Assignment 1).

### Code Structure
The code is organized into two main Jupyter Notebooks, each representing a separate assignment, but building upon similar concepts:

*   **`Coding_Assignment1_PINNs.ipynb`:**
    *   Loads data from `cauchy_euler_dataset.mat`.
    *   Defines `CauchyEulerDataset` and `CauchyEulerNN` (a standard feedforward network).
    *   Trains the `CauchyEulerNN` using MSE loss on the MATLAB-generated data.
    *   Evaluates the model and plots the loss history.
    *   Calculates accuracy metrics (MSE, MAE, R-squared).
*   **`Coding_Assignment_2_Atharv_Priyadarshi_23117034.ipynb`:**
    *   Generates synthetic data using the analytical solution with added noise.
    *   Defines `NeuralNet` (the same network architecture as Assignment 1).
    *   Trains the `NeuralNet` as a standard NN on the generated data.
    *   Trains another instance of `NeuralNet` as a PINN, combining data loss and physics loss.
    *   Plots the loss histories for both the standard NN and PINN.
    *   Compares the predictions of both models against the actual data visually.

---

## 7. Results & Analysis

The performance of both the standard Neural Network (NN) and the Physics-Informed Neural Network (PINN) was evaluated by comparing their predictions against the actual data generated from the analytical solution.

### Loss Convergence
*   Both the Regular NN and PINN models show a rapid decrease in loss during the initial epochs, indicating successful learning.
*   The standard NN's loss converges to a slightly lower value than the PINN's total loss. This is expected because the standard NN only optimizes for data fitting, whereas the PINN must balance both data matching and adherence to the underlying physics, which can introduce a higher overall loss value but potentially a more physically consistent solution.

### Prediction Accuracy
*   Both models are able to capture the overall decreasing trend of the solution.
*   The **standard NN's predictions** tend to deviate more significantly from the actual noisy data, especially in regions with higher fluctuations. While it attempts to fit the noise, it might not capture the true underlying trend robustly.
*   The **PINN's prediction curve** exhibits a smoother prediction curve that aligns more closely with the actual data, particularly in capturing the underlying trend of the differential equation. This highlights how the physics-informed component of the loss function helps to regularize the PINN's predictions, leading to a more physically consistent solution and preventing overfitting to the noise in the training data.

---

## 8. Challenges Encountered

*   **Dataset Generation and Noise:** Creating a sufficiently complex dataset that accurately represents the physics while also being suitable for training both models was challenging. Adding realistic Gaussian noise further complicated the process.
*   **Model Accuracy for Nonlinear Equations:** Ensuring the neural networks accurately approximate complex differential equations (like the Cauchy-Euler equation) required extensive experimentation with network architecture, learning rates, and optimizer settings.
*   **Training Stability and Loss Convergence:** Maintaining stable training and minimizing fluctuations in the loss functions, particularly the physics loss, was a significant challenge. Initial attempts often led to increasing total loss over epochs, hindering convergence.
*   **Balancing Data and Physics Loss:** In PINNs, finding the right balance between the `data_loss` and `physics_loss` (managed by `lambda_physics`) is crucial. An improper balance can lead to overfitting to the data (ignoring physics) or overfitting to the physics (ignoring data consistency), affecting overall accuracy and generalization. The adaptive weighting strategy was implemented to address this.
*   **Visualization and Interpretation:** Generating clear, interpretable plots for actual vs. predicted data and loss convergence was essential for understanding and communicating model performance effectively.

---

## 9. Limitations

While the PINN demonstrated promising results in this project, some limitations should be acknowledged:
*   **Reliance on Analytical Solution:** The model's performance relies heavily on the accuracy of the analytical solution used for data generation. In real-world scenarios where analytical solutions are often unavailable, obtaining accurate training data can be a major challenge.
*   **Specific Equation Form:** The current implementation focuses on a specific form of the Cauchy-Euler equation. Generalizing this approach to other forms of differential equations or more complex boundary conditions requires further investigation and adaptation.
*   **Noise Handling:** While noise was added, the specific handling of noise (e.g., its distribution or magnitude) can significantly impact PINN performance. More sophisticated noise models or robust loss functions might be needed for highly noisy real-world data.

---

## 10. Future Improvements

*   **Exploring Different Network Architectures:** Experimenting with deeper networks, wider layers, or different activation functions (e.g., Swish, GELU) could potentially improve model performance and convergence speed.
*   **Alternative Optimizers or Learning Rate Schedules:** Investigating other optimizers (e.g., L-BFGS for PINNs) or advanced learning rate schedules (e.g., cosine annealing, learning rate finders) might enhance training stability and convergence.
*   **Adaptive Weighting Strategies:** Further research into more dynamic and adaptive weighting strategies for the `data_loss` and `physics_loss` could lead to more robust and automatic balance tuning, reducing the need for manual hyperparameter search.
*   **Applying to More Complex Equations:** Extending the PINN approach to solve more complex Cauchy-Euler equations with different boundary conditions, non-homogeneous terms, or non-linear variations would be a valuable next step.
*   **Incorporating Boundary Conditions:** While not explicitly enforced in the loss for this particular problem's setup (as the analytical solution already implicitly handles them through data generation), explicitly adding boundary condition terms to the loss function would be crucial for solving ODEs/PDEs where such conditions are fundamental.

---

## 11. How to Run the Project

To run this project on your local machine, follow these steps:

### Prerequisites
*   **Python 3.x:** Ensure you have Python installed.
*   **Git:** Make sure Git is installed on your system.

### 1. Clone the Repository
Open your Git Bash terminal (or any terminal with Git configured) and clone the repository to your local machine:
```bash
git clone https://github.com/YourUsername/Cauchy-Euler-PINN-Project.git
# Replace 'YourUsername' and 'Cauchy-Euler-PINN-Project' with your actual GitHub username and repository name.
