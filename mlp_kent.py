import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import h5py
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TorchStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.device = None
    
    def to(self, device):
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        self.device = device
        return self
    
    def fit(self, x):
        self.device = x.device
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)
        return self
    
    def transform(self, x):
        if self.device != x.device:
            self.to(x.device)
        return (x - self.mean) / (self.std + 1e-8)
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def state_dict(self):
        return {
            'mean': self.mean.cpu(),
            'std': self.std.cpu()
        }
    
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self.device = self.mean.device

class KentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X if isinstance(X, torch.Tensor) else torch.FloatTensor(X)
        self.y = y[:, [3, 4]] if isinstance(y, torch.Tensor) else torch.FloatTensor(y[:, [3, 4]])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class KentMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=[1024, 512, 256]):
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
                nn.Linear(hidden_dims[-1], 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()
            ) for _ in range(3)
        ])

        self.beta_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()
            ) for _ in range(3)
        ])
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        
        kappa_preds = torch.cat([head(shared_features) for head in self.kappa_head], dim=1)
        beta_preds = torch.cat([head(shared_features) for head in self.beta_head], dim=1)
        
        kappa = torch.mean(kappa_preds, dim=1, keepdim=True)
        beta = torch.mean(beta_preds, dim=1, keepdim=True)
        
        return torch.cat([kappa, beta], dim=1)

class KentLoss(nn.Module):
    def __init__(self, beta_weight=1):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='none', delta=1.0)
        self.beta_weight = beta_weight
        
    def forward(self, pred, target):
        kappa_loss = torch.mean(self.huber(pred[:, 0:1], target[:, 0:1]) / (target[:, 0:1] + 1e-8))
        beta_loss = torch.mean(self.huber(pred[:, 1:2], target[:, 1:2]) / (target[:, 1:2] + 1e-8))
        beta_loss = self.beta_weight * beta_loss
        
        total_loss = kappa_loss + beta_loss
        
        return total_loss, {
            'kappa_loss': kappa_loss.item(),
            'beta_loss': beta_loss.item()
        }

def train_kent_mlp(X, y, batch_size=128, epochs=300, learning_rate=0.0001, 
                   hidden_dims=[1024, 512, 256], weight_decay=0.01,
                   scheduler_factor=0.5, scheduler_patience=10,
                   early_stopping_patience=20, train_split=0.8,
                   val_split=0.1, beta_weight=2.0):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Convert inputs to PyTorch tensors
    X = torch.as_tensor(X, dtype=torch.float32).to(device)
    y = torch.as_tensor(y, dtype=torch.float32).to(device)
    
    # Scale the data
    scaler = TorchStandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create train/val/test splits
    total_size = X.size(0)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    
    indices = torch.randperm(total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_dataset = KentDataset(X_scaled[train_indices], y[train_indices])
    val_dataset = KentDataset(X_scaled[val_indices], y[val_indices])
    test_dataset = KentDataset(X_scaled[test_indices], y[test_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = KentMLP(input_dim=X.size(1), hidden_dims=hidden_dims).to(device)
    criterion = KentLoss(beta_weight=beta_weight)
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
    
    try:
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
                best_model = model.state_dict().copy()
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
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    model.load_state_dict(best_model)
    
    # Test set evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_predictions.append(outputs)
            test_targets.append(y_batch)
    
    test_predictions = torch.cat(test_predictions)
    test_targets = torch.cat(test_targets)
    
    # Calculate metrics
    mse_kappa = torch.mean((test_predictions[:, 0] - test_targets[:, 0])**2)
    rmse_kappa = torch.sqrt(mse_kappa)
    rel_errors_kappa = torch.abs(test_predictions[:, 0] - test_targets[:, 0]) / (test_targets[:, 0] + 1e-8)
    mean_rel_error_kappa = torch.mean(rel_errors_kappa)
    
    mse_beta = torch.mean((test_predictions[:, 1] - test_targets[:, 1])**2)
    rmse_beta = torch.sqrt(mse_beta)
    rel_errors_beta = torch.abs(test_predictions[:, 1] - test_targets[:, 1]) / (test_targets[:, 1] + 1e-8)
    mean_rel_error_beta = torch.mean(rel_errors_beta)
    
    results = {
        'model': model.cpu(),  # Move model to CPU before returning
        'scaler': scaler.to('cpu'),  # Move scaler to CPU before returning
        'training_time': training_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'rmse_kappa': rmse_kappa.item(),
        'mean_relative_error_kappa': mean_rel_error_kappa.item(),
        'rmse_beta': rmse_beta.item(),
        'mean_relative_error_beta': mean_rel_error_beta.item(),
        'best_val_loss': best_val_loss
    }
    
    return results

def predict_kappa(model, scaler, input_tensor):
    """Predict kappa values while maintaining gradients and handling devices"""
    device = input_tensor.device
    model = model.to(device)
    scaler = scaler.to(device)
    
    model.eval()
    scaled_input = scaler.transform(input_tensor)
    predictions = model(scaled_input)
    
    return predictions

def save_kent_model(model, scaler, path):
    """Save model and scaler states"""
    # Move to CPU before saving
    model = model.cpu()
    scaler = scaler.to('cpu')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state_dict': scaler.state_dict()
    }, path)

def load_kent_model(model_path):
    """Load model and scaler states"""
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        hidden_dims = [512, 256, 128, 64]
        model = KentMLP(input_dim=4, hidden_dims=hidden_dims)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        scaler = TorchStandardScaler()
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise

def load_samples(h5_file):
    """Load samples from H5 file"""
    try:
        with h5py.File(h5_file, 'r') as f:
            X = f['original_boxes'][:]
            y = f['transformed_boxes'][:]
        return X, y
    except Exception as e:
        logger.error(f"Error loading samples from {h5_file}: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Define hyperparameters
        hyperparameters = {
            'batch_size': 128,
            'epochs': 300,
            'learning_rate': 0.0001,
            'hidden_dims': [512, 256, 128, 64],
            'weight_decay': 0.01,
            'scheduler_factor': 0.5,
            'scheduler_patience': 10,
            'early_stopping_patience': 20,
            'train_split': 0.8,
            'val_split': 0.1,
            'beta_weight': 2.0
        }
        
        # Load data
        logger.info("Loading data from H5 file...")
        X, y = load_samples('bbox_comparison.h5')
        
        # Train model
        logger.info("Starting model training...")
        results = train_kent_mlp(X, y, **hyperparameters)
        
        # Log results
        logger.info("\nTraining Results:")
        logger.info(f"Training time: {results['training_time']:.2f} seconds")
        logger.info(f"Kappa RMSE: {results['rmse_kappa']:.6f}")
        logger.info(f"Kappa Mean Relative Error: {results['mean_relative_error_kappa']:.6f}")
        logger.info(f"Beta RMSE: {results['rmse_beta']:.6f}")
        logger.info(f"Beta Mean Relative Error: {results['mean_relative_error_beta']:.6f}")
        
        # Save model
        model_path = 'kent_mlp_model.pth'
        logger.info(f"\nSaving model to {model_path}")
        save_kent_model(results['model'], results['scaler'], model_path)
        
        # Test prediction with gradients
        logger.info("\nTesting model prediction with gradients...")
        model, scaler = load_kent_model(model_path)
        
        # Example input tensor with gradients enabled
        test_input = torch.tensor([
            [180.28125, 133.03125, 5.0, 5.0],
            [35.0, 0.0, 23.0, 50.0],
            [35.0, 10.0, 23.0, 20.0]
        ], dtype=torch.float32, requires_grad=True)
        
        # Get predictions with gradients
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_input = test_input.to(device)
        predictions = predict_kappa(model, scaler, test_input)
        
        logger.info("Predicted values:")
        logger.info(predictions.cpu().detach().numpy())
        
        # Test backpropagation
        loss = predictions.sum()
        loss.backward()
        
        logger.info("\nInput gradients:")
        logger.info(test_input.grad.cpu().numpy())
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
