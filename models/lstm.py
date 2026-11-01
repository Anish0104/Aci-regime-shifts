"""Small LSTM one-step-ahead forecaster."""
import numpy as np
import torch
import torch.nn as nn


class _LSTM(nn.Module):
    def __init__(self, hidden=32, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                  # x: (batch, L)
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1]).squeeze(-1)


class LSTMForecaster:
    def __init__(self, hidden=32, layers=1, epochs=15, lr=1e-3, batch=128, seed=0):
        self.hidden, self.layers, self.epochs = hidden, layers, epochs
        self.lr, self.batch, self.seed = lr, batch, seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        # scaler fit on TRAINING data only (no leakage)
        self.mu, self.sd = float(y.mean()), float(y.std() + 1e-8)
        Xt = torch.tensor((X - self.mu) / self.sd)
        yt = torch.tensor((y - self.mu) / self.sd)
        self.net = _LSTM(self.hidden, self.layers)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        Xt = torch.tensor((X - self.mu) / self.sd)
        return self.net(Xt).numpy() * self.sd + self.mu