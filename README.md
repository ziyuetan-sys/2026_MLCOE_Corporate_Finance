# Corporate Finance
## Part1: Deep learning for solving Dynamic Model
### Overview

This project provides a **complete deep learning demonstration** for solving dynamic models in corporate finance.  
It implements a unified training framework that combines **economic model definitions**, **neural network approximation**, and **Bellman equation–based optimization**.

---
###  Project Structure
```text

dynamic_model/

├── Basic.py
 # Defines basic model with first-order condition

├── Debt.py 
# Defines dynamic model with risk-free and risky debt

├── Networks.py 
# Defines neural networks for Bellman equation approximation

└── Trainer.py 
# Provides unified training framework with TensorFlow

simulation/
├── basic_model.ipynb   # Tutorial for training and evaluating the basic model
└── risky_debt.ipynb    # Tutorial for risky debt dynamic model simulation

```
### Training Demo

```python
import sys, os
sys.path.append(os.path.abspath(".."))

from dynamic_model.Basic import BasicModel
from dynamic_model.Networks import BellmanNet_FOC
from dynamic_model.Trainer import BellmanTrainer

# Define a basic dynamic model
model = BasicModel(cost_type="None")

# Define neural network architecture
net = BellmanNet_FOC(model)

# Create trainer with unified framework
trainer = BellmanTrainer(model, net, hidden_dim=[32, 32], nu=10, lr=1e-4)

# Train model 
trainer.fit(
    training_steps=10000,
    display_step=2000,
    eval=True,
    early_stop=False,
    eval_interval=10,
    n_eval=5
)
```

### Tutorials

* [**basic_model.ipynb**](https://github.com/ziyuetan-sys/2026_MLCOE_Corporate_Finance/blob/main/simulation/basic_model.ipynb)
Demonstrates how to define and train the basic dynamic model using the BasicModel and unified training framework.

* [**risky_debt.ipynb**](https://github.com/ziyuetan-sys/2026_MLCOE_Corporate_Finance/blob/main/simulation/risky_debt.ipynb)
Shows an dynamic model with risky debt.