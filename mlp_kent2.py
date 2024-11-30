import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import h5py
import logging
from collections import defaultdict

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
                
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU()
        ]
        
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

class KentLoss(nn.Module):
    def __init__(self, last_param_weight=0.5):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='none', delta=1.0)
        self.last_param_weight = last_param_weight
        
    def forward(self, pred, target):
        kappa_loss = torch.mean(self.huber(pred[:, 0:1], target[:, 0:1]))
        last_param_loss = torch.mean(self.huber(pred[:, 1:2], target[:, 1:2]))
        last_param_loss = self.last_param_weight * last_param_loss
        
        total_loss = kappa_loss + last_param_loss
        
        return total_loss, {
            'kappa_loss': kappa_loss.item(),
            'last_param_loss': last_param_loss.item()
        }

def train_kent_mlp(X, y, batch_size=128, epochs=300, learning_rate=0.0001, 
                   hidden_dims=[1024, 768, 512, 384, 256], weight_decay=0.01,
                   scheduler_factor=0.5, scheduler_patience=10,
                   early_stopping_patience=20, train_split=0.8,
                   val_split=0.1, last_param_weight=2.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    X = np.array(X)
    y = np.array(y)
    y_target = y[:, [3, 4]]  # Select only required columns
    
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_X.fit_transform(X)
    
    y_log = np.log1p(np.abs(y_target))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_scaled = scaler_y.fit_transform(y_log)
    
    train_size = int(train_split * len(X))
    val_size = int(val_split * len(X))
    indices = np.random.permutation(len(X))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_dataset = KentDataset(X_scaled[train_indices], y_scaled[train_indices])
    val_dataset = KentDataset(X_scaled[val_indices], y_scaled[val_indices])
    test_dataset = KentDataset(X_scaled[test_indices], y_scaled[test_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = KentMLP(input_dim=X.shape[1], hidden_dims=hidden_dims).to(device)
    criterion = KentLoss(last_param_weight=last_param_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, 
                                 patience=scheduler_patience, verbose=True)
    
    best_val_loss = float('inf')
    best_model = None
    train_losses = []
    val_losses = []
    patience_counter = 0
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_metrics = defaultdict(float)
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss, metrics = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            for k, v in metrics.items():
                epoch_train_metrics[k] += v
        
        model.eval()
        epoch_val_loss = 0
        epoch_val_metrics = defaultdict(float)
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss, metrics = criterion(outputs, y_batch)
                
                epoch_val_loss += loss.item()
                for k, v in metrics.items():
                    epoch_val_metrics[k] += v

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
            
        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, "
                f"Val Loss = {avg_val_loss:.6f}, "
                f"LR = {optimizer.param_groups[0]['lr']:.2e}"
            )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    model.load_state_dict(best_model)
    
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            pred_numpy = outputs.cpu().numpy()
            pred_original = np.expm1(scaler_y.inverse_transform(pred_numpy))
            test_predictions.extend(pred_original)
            test_targets.extend(scaler_y.inverse_transform(y_batch.cpu().numpy()))
    
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    mse_kappa = np.mean((test_predictions[:, 0] - test_targets[:, 0])**2)
    rmse_kappa = np.sqrt(mse_kappa)
    rel_errors_kappa = np.abs(test_predictions[:, 0] - test_targets[:, 0]) / (test_targets[:, 0] + 1e-8)
    mean_rel_error_kappa = np.mean(rel_errors_kappa)
    
    mse_last = np.mean((test_predictions[:, 1] - test_targets[:, 1])**2)
    rmse_last = np.sqrt(mse_last)
    rel_errors_last = np.abs(test_predictions[:, 1] - test_targets[:, 1]) / (test_targets[:, 1] + 1e-8)
    mean_rel_error_last = np.mean(rel_errors_last)
    
    results = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'training_time': training_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'rmse_kappa': rmse_kappa,
        'mean_relative_error_kappa': mean_rel_error_kappa,
        'rmse_last': rmse_last,
        'mean_relative_error_last': mean_rel_error_last,
        'best_val_loss': best_val_loss
    }
    
    return results

def load_samples(h5_file):
    """Load samples from H5 file containing original and Kent annotations"""
    with h5py.File(h5_file, 'r') as f:
        X = f['original_boxes'][:]
        y = f['transformed_boxes'][:]
    return X, y

if __name__ == "__main__":
    hyperparameters = {
        'batch_size': 128,
        'epochs': 300,
        'learning_rate': 0.001,
        'hidden_dims': [1024, 768, 512, 384, 256],
        'weight_decay': 0.01,
        'scheduler_factor': 0.5,
        'scheduler_patience': 10,
        'early_stopping_patience': 20,
        'train_split': 0.8,
        'val_split': 0.1,
        'last_param_weight': 2.0
    }
    
    X, y = load_samples('bbox_comparison_easy.h5')
    
    results = train_kent_mlp(X, y, **hyperparameters)
    
    logger.info("\nResults:")
    logger.info(f"Kappa RMSE: {results['rmse_kappa']:.6f}")
    logger.info(f"Kappa Mean Relative Error: {results['mean_relative_error_kappa']:.6f}")
    logger.info(f"Last Parameter RMSE: {results['rmse_last']:.6f}")
    logger.info(f"Last Parameter Mean Relative Error: {results['mean_relative_error_last']:.6f}")
    
    torch.save({
        'model_state_dict': results['model'].state_dict(),
        'scaler_X': results['scaler_X'],
        'scaler_y': results['scaler_y'],
    }, 'kent_mlp_model.pth')