import numpy as np

# ---------------------------------------------------------
# BLOCK 1: The Core MULTI-LINEAR logic (Model Definition)
# ---------------------------------------------------------
class MyLinearRegression:
     # new data, so we reduced learning rate to make sure we don't overshoot the optimal weights
    def __init__(self, learning_rate=0.001):
        # stores the importance of each feature (distance, traffic, weather) called as weights
        self.weights = None 
        # intercept (Base fare)
        self.c = 0         
        self.lr = learning_rate
    
    def fit(self, X, y, epochs=1000):
        n_samples, n_features = X.shape 
        self.weights = np.zeros(n_features) # 3 features so 3 zeros to start with
        
        for _ in range(epochs):
            # np.dot (w1*x1 + w2*x2 + w3*x3) calculate at once for all samples
            y_pred = np.dot(X, self.weights) + self.c
            
            # calculating gradients (how much each weight is responsible for the error)
            D_w = (-2/n_samples) * np.dot(X.T, (y - y_pred))
            D_c = (-2/n_samples) * np.sum(y - y_pred)
            
            # updating weights (correcting the weights based on the error)
            self.weights = self.weights - (self.lr * D_w)
            self.c = self.c - (self.lr * D_c)
            
    def predict(self, X_new):
        return np.dot(X_new, self.weights) + self.c