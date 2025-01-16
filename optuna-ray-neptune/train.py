import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.wandb import WandbLogger
import wandb
import neptune.new as neptune
from neptune.new.integrations.optuna import NeptuneCallback
import optuna

# Define the neural network architecture
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(MyModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 else 
            nn.Linear(hidden_dim, hidden_dim) 
            for i in range(num_layers)
        ])
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x, _ = self.attention(x, x, x)  # Self-attention
        x = self.dropout(x)
        return x

# Define the training loop
def train_model(config):
    # Create the model
    model = MyModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"]
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Load your dataset (replace with your actual data loading logic)
    train_loader = DataLoader(your_dataset, batch_size=config["batch_size"])

    # Training loop
    for epoch in range(config["epochs"]):
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Log metrics to Weights & Biases
        wandb.log({"loss": loss.item()})

        # Report results to Ray Tune
        tune.report(loss=loss.item())

# Ray Tune with Weights & Biases
if __name__ == "__main__":
    ray.init()

    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "dropout": tune.uniform(0, 0.5),
        "hidden_dim": tune.choice([64, 128, 256]),
        "num_layers": tune.choice([2, 3, 4]),
        "num_heads": tune.choice([2, 4, 8]),
        "epochs": 10,  # Example: Adjust epochs as needed
        "input_dim": your_input_dim,  # Replace with your input dimension
    }

    # Create a scheduler (e.g., ASHAScheduler)
    scheduler = ASHAScheduler(metric="loss", mode="min", grace_period=10, max_t=100)

    # Create a Weights & Biases logger
    wandb_logger = WandbLogger()

    # Run the experiment
    tune.run(
        train_model,
        config=search_space,
        num_samples=10,
        scheduler=scheduler,
        loggers=[wandb_logger]
    )

# Optuna with Weights & Biases and Neptune
def objective(trial):
    # Get hyperparameters from trial
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0, 0.5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_categorical("num_layers", [2, 3, 4])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])

    # Create the model
    model = MyModel(
        input_dim=your_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    )

    # ... (Rest of the training loop as in the Ray Tune version) ...

    # Log metrics to Weights & Biases
    wandb.log({"loss": loss.item()})

    # Log metrics to Neptune
    run["metrics/loss"].log(loss.item())

    # Return the objective value
    return loss.item()

if __name__ == "__main__":
    # Initialize Neptune run
    run = neptune.init_run(project="your-project", api_token="your-api-token")

    # Create a study
    study = optuna.create_study(direction="minimize")

    # Add Neptune callback to the study
    study.optimize(objective, n_trials=10, callbacks=[NeptuneCallback(run)])