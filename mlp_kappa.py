import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import time
import h5py
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        # Transform kappa and beta to log space during dataset creation
        kappa = y[:, 0]
        beta = y[:, 1]
        log_kappa = np.log(kappa + 1e-8)  # Add small epsilon to avoid log(0)
        r = 2 * beta / (kappa + 1e-8)  # Convert beta to ratio
        r = np.clip(r, 0, 1)  # Ensure ratio is in [0,1]
        
        self.y = torch.FloatTensor(np.column_stack([log_kappa, r]))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class KentMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        self.input_batch_norm = nn.BatchNorm1d(input_dim)
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with residual connections
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),  # GELU instead of ReLU for better gradient flow
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        # Separate heads for log_kappa and beta_ratio
        self.shared_layers = nn.Sequential(*layers)
        self.log_kappa_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.beta_ratio_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Ensure ratio is between 0 and 1
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x = self.input_batch_norm(x)
        shared_features = self.shared_layers(x)
        log_kappa = self.log_kappa_head(shared_features)
        beta_ratio = self.beta_ratio_head(shared_features)
        return torch.cat([log_kappa, beta_ratio], dim=1)

class KentLoss(nn.Module):
    def __init__(self, kappa_weight=1.0, beta_weight=1.0):
        super().__init__()
        self.kappa_weight = kappa_weight
        self.beta_weight = beta_weight
        
    def forward(self, pred, target):
        # Unpack predictions and targets
        pred_log_kappa, pred_ratio = pred[:, 0], pred[:, 1]
        target_log_kappa, target_ratio = target[:, 0], target[:, 1]
        
        # Compute losses
        kappa_loss = torch.mean((pred_log_kappa - target_log_kappa) ** 2)
        ratio_loss = torch.mean((pred_ratio - target_ratio) ** 2)
        
        # Add regularization to prevent extreme values
        kappa_reg = 0.01 * torch.mean(torch.abs(pred_log_kappa))
        
        # Combine losses
        total_loss = (
            self.kappa_weight * kappa_loss +
            self.beta_weight * ratio_loss +
            kappa_reg
        )
        
        return total_loss, {
            'kappa_loss': kappa_loss.item(),
            'ratio_loss': ratio_loss.item(),
            'kappa_reg': kappa_reg.item()
        }

def convert_predictions(log_kappa, beta_ratio):
    """Convert network outputs back to original parameter space"""
    kappa = torch.exp(log_kappa)
    beta = (beta_ratio * kappa) / 2
    return kappa, beta

def train_kent_mlp(X, y, batch_size=64, epochs=200, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Extract only kappa and beta from y
    y = y[:, [3, 4]]  # Assuming kappa and beta are at indices 3 and 4
    
    # Prepare data
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Log statistics about the parameters
    logger.info(f"Kappa range: [{y[:,0].min():.2e}, {y[:,0].max():.2e}]")
    logger.info(f"Beta range: [{y[:,1].min():.2e}, {y[:,1].max():.2e}]")
    
    # Split data
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))
    indices = np.random.permutation(len(X))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create datasets
    train_dataset = KentDataset(X_scaled[train_indices], y[train_indices])
    val_dataset = KentDataset(X_scaled[val_indices], y[val_indices])
    test_dataset = KentDataset(X_scaled[test_indices], y[test_indices])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model and training components
    model = KentMLP().to(device)
    criterion = KentLoss(kappa_weight=1.0, beta_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=10, factor=0.5, min_lr=1e-6
    )
    
    # Training
    best_val_loss = float('inf')
    best_model = None
    train_losses = []
    val_losses = []
    early_stopping_counter = 0
    early_stopping_patience = 20
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        model.train()
        epoch_train_loss = 0
        epoch_train_metrics = defaultdict(float)
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss, metrics = criterion(outputs, y_batch)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            for k, v in metrics.items():
                epoch_train_metrics[k] += v
        
        # Validate
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
        
        # Calculate average losses and metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping and model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            logger.info("Early stopping triggered")
            break
        
        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, "
                f"Val Loss = {avg_val_loss:.6f}, "
                f"LR = {optimizer.param_groups[0]['lr']:.2e}"
            )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Load best model for evaluation
    model.load_state_dict(best_model)
    
    # Test evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            # Convert predictions back to original space
            log_kappa_pred, ratio_pred = outputs[:, 0], outputs[:, 1]
            kappa_pred, beta_pred = convert_predictions(log_kappa_pred, ratio_pred)
            
            test_predictions.extend(torch.stack([kappa_pred, beta_pred], dim=1).cpu().numpy())
            
            # Convert targets back to original space
            log_kappa_target, ratio_target = y_batch[:, 0], y_batch[:, 1]
            kappa_target = torch.exp(log_kappa_target)
            beta_target = (ratio_target * kappa_target) / 2
            
            test_targets.extend(torch.stack([kappa_target, beta_target], dim=1).cpu().numpy())
    
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    # Calculate metrics
    mse = np.mean((test_predictions - test_targets)**2, axis=0)
    rmse = np.sqrt(mse)
    
    # Calculate relative errors
    rel_errors = np.abs(test_predictions - test_targets) / (test_targets + 1e-8)
    mean_rel_errors = np.mean(rel_errors, axis=0)
    
    results = {
        'model': model,
        'scaler': scaler_X,
        'training_time': training_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'rmse': rmse,
        'mean_relative_errors': mean_rel_errors,
        'best_val_loss': best_val_loss
    }
    
    return results

def load_samples(h5_file):
    """Load samples from H5 file containing original and Kent annotations"""
    with h5py.File(h5_file, 'r') as f:
        X = f['original_annotations'][:]  # Original COCO annotations
        y = f['new_annotations'][:]       # Kent parameters
    return X, y

if __name__ == "__main__":
    # Load data from the H5 file generated by coco_to_kent.py
    X, y = load_samples('kent_samples.h5')
    results = train_kent_mlp(X, y)

    # Print results
    logger.info("\nResults for Kent parameters:")
    param_names = ['κ (kappa)', 'β (beta)']
    for i, param in enumerate(param_names):
        logger.info(f"\n{param}:")
        logger.info(f"RMSE: {results['rmse'][i]:.6f}")
        logger.info(f"Mean Relative Error: {results['mean_relative_errors'][i]:.6f}")