# Optimization Algorithms and SDE Approximations

This repository contains Python implementations for comparing discrete optimization algorithms (Adam, RMSProp) with their continuous SDE (Stochastic Differential Equation) approximations. The code supports experiments in both ballistic and batch equivalent regimes, and includes modules for neural networks, polynomial and quadratic function approximation, and result visualization.

## Folder Structure
- **Algorithms/**: Core implementations of Adam and RMSProp in different regimes.
- **NeuralNetwork/**: Deep learning models, training scripts, and utilities.
- **Poly/**: Polynomial function experiments and plotting.
- **QuadraticFunction/**: Quadratic function experiments, SDE/ODE simulations, and utilities.

## Requirements
See `requirements.txt` for dependencies. All main packages are installed via pip:

## Usage
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run experiments:
   - For quadratic function experiments:
     ```
     bash QuadraticFunction/runs.sh      # Linux/macOS
     QuadraticFunction\runs.bat          # Windows
     ```
   - For polynomial experiments:
     ```
     bash Poly/runs.sh                   # Linux/macOS
     Poly\runs.bat                       # Windows
     ```
   - For neural network experiments:
     ```
     bash NeuralNetwork/runs.sh          # Linux/macOS
     NeuralNetwork\runs.bat              # Windows
     ```

## License
MIT License

## Contact
For questions or contributions, open an issue or contact the repository maintainer.
