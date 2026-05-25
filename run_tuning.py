"""
Hyperparameter tuning experiments for LSTM and TCN.
Logs results to outputs/tables/tuning_log.csv
"""
import json, time, warnings, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cpu')

pivot = pd.read_parquet('data/processed/milan_traffic_pivot.parquet')
with open('data/processed/target_areas.json') as f:
    info = json.load(f)

TARGET_AREAS = info['target_areas']
TEST_START = pd.Timestamp('2013-12-16', tz='UTC')
TRAIN_END  = TEST_START - pd.Timedelta(minutes=10)

# Use area 4159 as the tuning area (mid-range traffic, representative)
TUNE_AREA = 4159

ts_full = pivot[TUNE_AREA].fillna(0)
train_vals = ts_full[ts_full.index <= TRAIN_END].values.reshape(-1,1)
test_vals  = ts_full[(ts_full.index >= TEST_START) &
                     (ts_full.index <= pd.Timestamp('2013-12-22 23:59', tz='UTC'))].values

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_vals).flatten()


class TrafficDataset(Dataset):
    def __init__(self, series, seq_len):
        self.X, self.y = [], []
        for i in range(len(series) - seq_len):
            self.X.append(series[i:i+seq_len])
            self.y.append(series[i+seq_len])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LSTMModel(nn.Module):
    def __init__(self, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)[..., :-self.conv1.padding[0]]))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)[..., :-self.conv2.padding[0]]))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, num_channels=64, kernel_size=3, num_blocks=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = 1
        for i in range(num_blocks):
            layers.append(TCNBlock(in_ch, num_channels, kernel_size, 2**i, dropout))
            in_ch = num_channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, 1)
    def forward(self, x):
        out = self.network(x.permute(0, 2, 1))
        return self.fc(out[:, :, -1]).squeeze(-1)


def train_eval(model_fn, seq_len, epochs=40, batch_size=64, lr=1e-3, val_frac=0.1, patience=7):
    split = int(len(train_scaled) * (1 - val_frac))
    train_ds = TrafficDataset(train_scaled[:split], seq_len)
    val_ds   = TrafficDataset(train_scaled[split:], seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    model = model_fn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val, best_state, pat_count = np.inf, None, 0

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        vl = np.mean([loss_fn(model(xb), yb).item()
                      for xb, yb in val_dl])
        if vl < best_val:
            best_val, best_state, pat_count = vl, model.state_dict(), 0
        else:
            pat_count += 1
            if pat_count >= patience:
                break
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)

    # Auto-regressive prediction
    model.eval()
    buf = list(train_scaled[-seq_len:])
    preds = []
    with torch.no_grad():
        for _ in range(len(test_vals)):
            x = torch.tensor(buf[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            p = model(x).item()
            preds.append(p)
            buf.append(p)
    preds = np.maximum(scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten(), 0)
    mae  = mean_absolute_error(test_vals, preds)
    rmse = np.sqrt(mean_squared_error(test_vals, preds))
    mape = np.mean(np.abs((test_vals - preds) / (np.abs(test_vals) + 1e-8))) * 100
    return mae, mape, rmse, train_time, epoch+1


results = []

# ── LSTM experiments ────────────────────────────────────────────────────────
lstm_configs = [
    # seq_len, hidden, layers, lr, label, reasoning
    (72,  64,  2, 1e-3, 'LSTM-A', 'Baseline: 12h window, 2-layer 64 units'),
    (144, 64,  2, 1e-3, 'LSTM-B', 'Increase window to 1 day: captures full daily cycle'),
    (144, 128, 2, 1e-3, 'LSTM-C', 'Larger hidden (128): more capacity for complex patterns'),
    (144, 64,  1, 1e-3, 'LSTM-D', 'Shallower (1 layer): test if depth helps or hurts'),
    (288, 64,  2, 5e-4, 'LSTM-E', 'Longer window (2 days) with lower LR for stability'),
]

print('=== LSTM Tuning ===')
for seq_len, hidden, layers, lr, label, reason in lstm_configs:
    def make_lstm(h=hidden, l=layers): return LSTMModel(h, l)
    mae, mape, rmse, tt, ep = train_eval(make_lstm, seq_len, lr=lr)
    print(f'{label}: seq={seq_len} hidden={hidden} layers={layers} lr={lr} | '
          f'MAE={mae:.2f} MAPE={mape:.1f}% RMSE={rmse:.2f} | train={tt:.0f}s ep={ep}')
    results.append({'Model': 'LSTM', 'Config': label, 'seq_len': seq_len,
                    'param1': f'hidden={hidden}', 'param2': f'layers={layers}',
                    'lr': lr, 'MAE': round(mae,2), 'MAPE': round(mape,1),
                    'RMSE': round(rmse,2), 'Train_s': round(tt,1),
                    'Epochs': ep, 'Reasoning': reason})

# ── TCN experiments ─────────────────────────────────────────────────────────
tcn_configs = [
    (144, 32,  3, 3, 1e-3, 'TCN-A', 'Baseline: 32 channels, 3 blocks, kernel=3'),
    (144, 64,  3, 3, 1e-3, 'TCN-B', 'Double channels (64): richer feature maps'),
    (144, 64,  4, 3, 1e-3, 'TCN-C', 'Add block (4 blocks): wider receptive field'),
    (144, 64,  3, 5, 1e-3, 'TCN-D', 'Larger kernel (5): smoother temporal patterns'),
    (288, 64,  3, 3, 5e-4, 'TCN-E', 'Longer window (2 days) with lower LR'),
]

print('\n=== TCN Tuning ===')
for seq_len, ch, blocks, ks, lr, label, reason in tcn_configs:
    def make_tcn(c=ch, b=blocks, k=ks): return TCNModel(c, k, b)
    mae, mape, rmse, tt, ep = train_eval(make_tcn, seq_len, lr=lr)
    print(f'{label}: seq={seq_len} ch={ch} blocks={blocks} kernel={ks} lr={lr} | '
          f'MAE={mae:.2f} MAPE={mape:.1f}% RMSE={rmse:.2f} | train={tt:.0f}s ep={ep}')
    results.append({'Model': 'TCN', 'Config': label, 'seq_len': seq_len,
                    'param1': f'channels={ch}', 'param2': f'blocks={blocks}',
                    'lr': lr, 'MAE': round(mae,2), 'MAPE': round(mape,1),
                    'RMSE': round(rmse,2), 'Train_s': round(tt,1),
                    'Epochs': ep, 'Reasoning': reason})

os.makedirs('outputs/tables', exist_ok=True)
df = pd.DataFrame(results)
df.to_csv('outputs/tables/tuning_log.csv', index=False)
print('\nSaved outputs/tables/tuning_log.csv')
print(df[['Model','Config','seq_len','param1','param2','lr','MAE','MAPE','RMSE','Train_s']].to_string())
