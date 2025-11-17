"""
Machine Learning Model Example
Author: Bahaa Al-Deen Walid Ahmad Al-Eid
Description: Simple ML model demonstration using scikit-learn
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class SimpleMLModel:
    """
    A simple machine learning model wrapper for linear regression
    """
    
    def __init__(self):
        """Initialize the model"""
        self.model = LinearRegression()
        self.is_trained = False
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model on provided data
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            dict: Training metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
    
    # Create and train model
    ml_model = SimpleMLModel()
    metrics = ml_model.train(X, y)
    
    print("Machine Learning Model Training Results:")
    print("=" * 50)
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
