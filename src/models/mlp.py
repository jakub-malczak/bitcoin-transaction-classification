import torch
import torch.nn as nn
from src.config import SEED

from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

def build_mlp(in_features, hidden_size):
    return MLP(
        in_features=in_features,
        hidden_size=hidden_size
    )

class MLP(nn.Module):
    def __init__(self, in_features, hidden_size):
        torch.manual_seed(SEED)

        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size, device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, device=self.device),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    def fit(self, X, y, batch_size=128, epochs=200, lr=0.001):
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        self.to(self.device)

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.float())

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                avg_loss = train_loss / len(train_loader)
                print(f"PyTorch MLP Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f}")

        return self

    def predict_proba(self, X):
        self.eval()

        if isinstance(X, pd.DataFrame): X = X.values
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            probs_class_1 = self(X_tensor).cpu().numpy()

        probs_class_0 = 1.0 - probs_class_1

        return np.hstack((probs_class_0, probs_class_1))

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)