import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from domain.model.modules import MLP

class Objective:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __call__(self, trial):
        
        # Define the hyperparameters to be optimized by Optuna
        hidden_factor = trial.suggest_uniform('hidden_factor', 0.5, 2.0)
        depth = trial.suggest_int('depth', 1, 5)
        
        model = MLP(
            input_shape=(784,),  # Example for MNIST
            num_classes=10,
            hidden_factor=hidden_factor,
            depth=depth
        )

        # Training specifics
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(self.data, self.target), batch_size=64, shuffle=True)

        # Training loop
        for epoch in range(3):  # Using fewer epochs for demonstration
            for batch, (x, y) in enumerate(loader):
                optimizer.zero_grad()
                output = model(x.float())
                loss = criterion(output, y.long())
                loss.backward()
                optimizer.step()

        # Evaluate the model (simplistic evaluation: just return the last loss)
        return loss.item()

# Dummy dataset
x_dummy = torch.randn(1000, 784)  # 1000 samples, 784 features (e.g., flattened 28x28 images)
y_dummy = torch.randint(0, 10, (1000,))  # 1000 target class labels

# Create an objective function instance with the dataset
objective = Objective(x_dummy, y_dummy)

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

print('Best trial:', study.best_trial.params)


"""
base) (base) riley@Rileys-MBP domain-exploration % /Users/riley/miniconda3/bin/python optuna_testing.py
[I 2024-05-17 14:29:28,883] A new study created in memory with name: no-name-fbcb25cf-1bf7-4d6d-bb63-caf58a42587a
/Users/riley/Desktop/DomainResearch/domain-exploration/optuna_testing.py:17: FutureWarning: suggest_uniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float instead.
  hidden_factor = trial.suggest_uniform('hidden_factor', 0.5, 2.0)
/Users/riley/miniconda3/lib/python3.11/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  _torch_pytree._register_pytree_node(
[I 2024-05-17 14:29:30,140] Trial 0 finished with value: 1.7674814462661743 and parameters: {'hidden_factor': 1.1675430883678002, 'depth': 4}. Best is trial 0 with value: 1.7674814462661743.
/Users/riley/Desktop/DomainResearch/domain-exploration/optuna_testing.py:17: FutureWarning: suggest_uniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float instead.
  hidden_factor = trial.suggest_uniform('hidden_factor', 0.5, 2.0)
[I 2024-05-17 14:29:30,402] Trial 1 finished with value: 0.5076004266738892 and parameters: {'hidden_factor': 0.7484156696703561, 'depth': 3}. Best is trial 1 with value: 0.5076004266738892.
[I 2024-05-17 14:29:31,067] Trial 2 finished with value: 2.1297082901000977 and parameters: {'hidden_factor': 1.1684242548117951, 'depth': 5}. Best is trial 1 with value: 0.5076004266738892.
[I 2024-05-17 14:29:31,551] Trial 3 finished with value: 0.995756983757019 and parameters: {'hidden_factor': 1.1099819602518086, 'depth': 3}. Best is trial 1 with value: 0.5076004266738892.
[I 2024-05-17 14:29:32,404] Trial 4 finished with value: 2.202216386795044 and parameters: {'hidden_factor': 1.2517278060109018, 'depth': 5}. Best is trial 1 with value: 0.5076004266738892.
[I 2024-05-17 14:29:32,889] Trial 5 finished with value: 0.1474187970161438 and parameters: {'hidden_factor': 1.6761155220250785, 'depth': 1}. Best is trial 5 with value: 0.1474187970161438.
[I 2024-05-17 14:29:33,215] Trial 6 finished with value: 1.6505584716796875 and parameters: {'hidden_factor': 0.5553950553044246, 'depth': 4}. Best is trial 5 with value: 0.1474187970161438.
[I 2024-05-17 14:29:33,869] Trial 7 finished with value: 2.084409475326538 and parameters: {'hidden_factor': 0.9288305019242002, 'depth': 5}. Best is trial 5 with value: 0.1474187970161438.
[I 2024-05-17 14:29:34,198] Trial 8 finished with value: 1.1018269062042236 and parameters: {'hidden_factor': 0.7673741392884553, 'depth': 3}. Best is trial 5 with value: 0.1474187970161438.
[I 2024-05-17 14:29:34,631] Trial 9 finished with value: 0.9164503216743469 and parameters: {'hidden_factor': 0.8717957820839919, 'depth': 3}. Best is trial 5 with value: 0.1474187970161438.
Best trial: {'hidden_factor': 1.6761155220250785, 'depth': 1} 
"""
