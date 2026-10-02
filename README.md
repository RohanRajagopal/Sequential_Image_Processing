# Neural Analog Circuit Flow

**A Physics-Informed Neural ODE for Computer Vision**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![JAX](https://img.shields.io/badge/JAX-0.4-8A2BE2)
![Equinox](https://img.shields.io/badge/Framework-Equinox-red)

This repository implements a **Deep Circuit Graph**: a continuous-depth neural network that treats image recognition as a signal propagation problem through a simulated analog circuit.

Instead of standard ReLU layers, we model non-linear oscillatory dynamics using **Kirchhoff's laws** and MOSFET-like equation sets, solved via `diffrax` with high-precision (64-bit) integration.

### 🔬 Key Features
* **Physics-Informed:** Layers modeled as coupled oscillators with resistance, capacitance, and current sources.
* **Neural ODE Solver:** Uses `diffrax.Tsit5` with PID controller for adaptive step-size integration.
* **Parallel Training:** `jax.pmap` implementation for multi-GPU/TPU data parallelism.
* **High Precision:** Full `float64` pipeline to maintain numerical stability in stiff ODE regions.

### 📊 Visualization
*(Insert a screenshot of your `visualize_dynamics` dashboard here)*

### 🚀 Quick Start

1. **Install Dependencies**
   (Ensure you install the correct JAX version for your CUDA driver first)
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training**
   The script automatically downloads the GTSRB dataset (German Traffic Signs).
   ```bash
   python main.py
   ```
   *Note: On the first run, this will download ~600MB of data.*

### 🧠 Architecture details

The model treats the image height (H) as the **Time** domain ($t$) and the image width + channels as the input signal $u(t)$.

$$
\frac{du}{dt} = f(u(t), \theta_{circuit})
$$

The dynamics are governed by a custom `CircuitOscillatorLayer` which simulates charge accumulation ($C_{eff}$) and leakage ($R_{val}$) across nodes.

### 📁 Project Structure

* `src/model.py`: Contains the `DeepCircuitGraph` and physics equations.
* `src/trainer.py`: JAX `pmap` training loops and Equinox serialization.
* `src/config.py`: Hyperparameters (Time steps, LR, ODE tolerances).

### 📄 License
MIT License
