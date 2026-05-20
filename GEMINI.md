# Project: Gradient Optimization Methods for Neural Networks (Kursova)

This project is a university coursework (Kursova) titled **"Gradient Optimization Methods for Training a Neural Network"**. It compares three different optimization algorithms for training a Multi-Layer Perceptron (MLP) on the classic XOR logical problem.

## Project Overview
The project implements a 2→4→1 MLP architecture with Sigmoid activation functions and Mean Squared Error (MSE) loss. It specifically explores and compares the efficiency of the following optimization techniques:
1.  **Gradient Descent (GD):** Classic first-order optimization with a fixed learning rate.
2.  **Steepest Descent (SD):** A variation of GD that uses an exact line search (Golden-Section Search) to find the optimal step size at each iteration.
3.  **Newton's Method:** A second-order optimization method that uses the Hessian matrix (calculated numerically) and Levenberg-Marquardt damping for numerical stability.

## Key Technologies
- **Python 3**
- **NumPy:** Core library for numerical computations and matrix operations.
- **SciPy:** Used for line search optimization in the Steepest Descent method.
- **Matplotlib:** For generating convergence plots and performance visualizations.
- **Typst:** Used for writing the final coursework report (`Kursova.typ`).

## Directory Structure
- `neural_network.py`: Definition of the `MLP` class (forward/backward passes, loss calculation).
- `optimizers.py`: Implementation of the `Optimizers` class containing GD, SD, and Newton methods.
- `run_experiments.py`: Main entry point for running comparison experiments and generating plots.
- `xor_data.py`: Training data for the XOR problem.
- `test_optimizers.py` / `test_forward.py`: Scripts for verifying the implementation.
- `Kursova.typ`: The final report source in Typst.
- `images/`: Directory for generated plots and report illustrations.
- `lib.bib`: Bibliography for the report.

## Building and Running
### Running Experiments
To execute the comparison between the three optimization methods and generate convergence plots:
```bash
python run_experiments.py
```

### Running Tests
To verify the functionality of the optimizers or the forward pass:
```bash
python test_optimizers.py
python test_forward.py
```

### Generating the Report (Requires Typst)
If you have Typst installed, you can compile the report:
```bash
typst compile Kursova.typ
```

## Development Conventions
- **Matrix Operations:** Use NumPy's vectorized operations whenever possible for performance.
- **Optimization Interface:** The `Optimizers` class methods take a `net` (MLP instance), training data `X`, and labels `y`.
- **Reproducibility:** A fixed seed is used in `neural_network.py` and experiment scripts to ensure consistent results.
- **Documentation:** Code includes comments in both English and Ukrainian (especially in the report-related files).
