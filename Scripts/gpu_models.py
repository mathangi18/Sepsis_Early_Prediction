import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class PyTorchLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, batch_size=None, device=None, verbose=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        # Convert to numpy arrays if they are pandas objects
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        # Define model
        self.model = nn.Linear(n_features, 1).to(self.device)
        
        # Loss and Optimizer
        # Using BCEWithLogitsLoss which combines Sigmoid and BCELoss for numerical stability
        criterion = nn.BCEWithLogitsLoss() 
        
        # LBFGS is often faster for logistic regression on full batch
        if self.batch_size is None:
             optimizer = optim.LBFGS(self.model.parameters(), lr=1, max_iter=20, tolerance_grad=self.tol, line_search_fn='strong_wolfe')
        else:
             optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        prev_loss = float('inf')
        
        for epoch in range(self.max_iter):
            def closure():
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                return loss
            
            if self.batch_size is None:
                loss = optimizer.step(closure)
            else:
                 # Mini-batch implementation (simplified for now, mostly full batch is fine for this size on GPU)
                 # For now, replicate full batch behavior if batch_size is None
                 loss = closure()
                 optimizer.step()

            current_loss = loss.item()
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss {current_loss:.6f}")

            if abs(prev_loss - current_loss) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch}")
                break
            prev_loss = current_loss

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        if hasattr(X, 'values'):
            X = X.values
        X = check_array(X)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        # Return (n_samples, 2) array as sklearn expects
        return np.hstack([1 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
