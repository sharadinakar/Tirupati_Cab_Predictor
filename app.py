import numpy as np
import gradio as gr

# ---------------------------------------------------------
# BLOCK 1: The Core Engine (MATHS & ALGORITHMS)
# ---------------------------------------------------------
class MyLinearRegression:
    def __init__(self, learning_rate=0.01):
        self.m = 0
        self.c = 0
        self.lr = learning_rate
        
    def fit(self, X, y, epochs=1000):
        n = len(X) 
        for _ in range(epochs):
            y_pred = self.m * X + self.c
            D_m = (-2/n) * np.sum(X * (y - y_pred))
            D_c = (-2/n) * np.sum(y - y_pred)
            self.m = self.m - (self.lr * D_m)
            self.c = self.c - (self.lr * D_c)
            
    def predict(self, X_new):
        return self.m * X_new + self.c

# ---------------------------------------------------------
# BLOCK 2: Data & Training (Data Preparation and Model Training)
# ---------------------------------------------------------

X = np.array([2, 4, 5, 8, 10]) 
y = np.array([50, 200, 250, 400,500])




model = MyLinearRegression()
model.fit(X, y, epochs=1000)

# ---------------------------------------------------------
# BLOCK 3: The Front-End UI (Customer-facing Web App)
# ---------------------------------------------------------
def get_cab_price(distance):
    price = model.predict(distance) 
    return f"₹ {round(price, 2)}"


interface = gr.Interface(
    fn=get_cab_price, 
    inputs=gr.Slider(minimum=1, maximum=50, label="Distance (in KM)"), 
    outputs=gr.Text(label="Predicted Cab Price"),
    title="Cab Price Predictor - Core AI Engine",
    description="Engine Built from Scratch without Sklearn."
)

if __name__ == "__main__":
    interface.launch()