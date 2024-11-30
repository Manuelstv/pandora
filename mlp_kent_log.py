import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import logging
import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class KentMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=[1024, 768, 512, 384, 256]):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU()
            ])
        
        self.shared_layers = nn.Sequential(*layers)
        
        self.kappa_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Softplus()
            ) for _ in range(3)
        ])
        
        self.last_param_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 512),
                nn.ReLU(),
                nn.Linear(512, 384),
                nn.ReLU(),
                nn.Linear(384, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.ReLU()
            ) for _ in range(5)
        ])
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        kappa_preds = torch.cat([head(shared_features) for head in self.kappa_head], dim=1)
        last_param_preds = torch.cat([head(shared_features) for head in self.last_param_head], dim=1)
        kappa = torch.mean(kappa_preds, dim=1, keepdim=True)
        last_param = torch.mean(last_param_preds, dim=1, keepdim=True)
        return torch.cat([kappa, last_param], dim=1)

def train_kent_mlp(X, y, batch_size=128, epochs=300, learning_rate=0.0001, hidden_dims=[1024, 768, 512, 384, 256]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Log transform kappa values
    y_transformed = y.copy()
    y_transformed[:, 0] = np.log1p(y_transformed[:, 0])
    
    # Scale inputs and outputs with MinMaxScaler
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_transformed)
    
    dataset = KentDataset(X_scaled, y_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = KentMLP(input_dim=X.shape[1], hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    return model, scaler_X, scaler_y

def predict_kent_params_batch(model, scaler_X, scaler_y, input_tensor, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    input_np = input_tensor.detach().numpy()
    input_scaled = scaler_X.transform(input_np)
    
    input_scaled = torch.FloatTensor(input_scaled)
    predictions = []
    model.eval()
    
    for i in range(0, len(input_scaled), batch_size):
        batch = input_scaled[i:i + batch_size].to(device)
        with torch.no_grad():
            batch_predictions = model(batch)
            predictions.append(batch_predictions.cpu().numpy())
    
    predictions = np.vstack(predictions)
    predictions_original = scaler_y.inverse_transform(predictions)
    
    # Inverse transform kappa
    predictions_original[:, 0] = np.expm1(predictions_original[:, 0])
    
    return predictions_original

def load_samples(h5_file):
    """Load samples from H5 file containing original and Kent annotations"""
    with h5py.File(h5_file, 'r') as f:
        X = f['original_boxes'][:]
        y = f['transformed_boxes'][:]
    return X, y

if __name__ == "__main__":
    np.random.seed(42)
    X, y = load_samples('bbox_comparison_easy.h5')
    
    model, scaler_X, scaler_y = train_kent_mlp(X, y)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }, 'kent_mlp_model.pth')
    
    test_tensor = torch.tensor([
        [180.28125, 133.03125, 5.0, 5.0],
        [35.0, 0.0, 23.0, 50.0],
        [35.0, 10.0, 23.0, 20.0]
    ], dtype=torch.float32)
    
    predictions = predict_kent_params_batch(model, scaler_X, scaler_y, test_tensor)
    print("\nPredictions (kappa, last_param):")
    for i, pred in enumerate(predictions):
        print(f"Box {i+1}: {pred}")