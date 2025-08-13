import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

LOG_PATH = Path('event_log.parquet')

class ToyHRM(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=32):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

def prepare_data(log_path):
    df = pd.read_parquet(log_path)
    sequences = []
    targets = []
    for i in range(len(df) - 5):
        seq = df.iloc[i:i+5]
        sequences.append(seq['vision_latent'].apply(lambda x: torch.tensor(x)).tolist() +
                         seq['imu_latent'].apply(lambda x: torch.tensor(x)).tolist() +
                         seq['coherence_score'].apply(lambda x: torch.tensor([x])).tolist())
        targets.append([df.iloc[i+5]['coherence_score']])
    X = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    return X, y

def train_model(X, y):
    model = ToyHRM(input_dim=X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return model

if __name__ == '__main__':
    X, y = prepare_data(LOG_PATH)
    train_model(X, y)
