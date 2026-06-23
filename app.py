import numpy as np
import pandas as pd
import gradio as gr

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

# ---------------------------------------------------------
# BLOCK 2: Data Loading & Train-Test Split
# ---------------------------------------------------------
print("Loading data from tirupati_cab_data.csv...")

# 1. Reading the CSV file into a DataFrame
df = pd.read_csv("tirupati_cab_data.csv")

# 2. x (features) and y (target) data preparation
X_data = df[['Distance_KM', 'Traffic_Level', 'Weather_Rainy']].values
y_data = df['Cab_Price_INR'].values

# 3. Train-Test Split (80% Training, 20% Testing)
split_index = int(0.8 * len(df))

X_train = X_data[:split_index]
y_train = y_data[:split_index]

X_test = X_data[split_index:]
y_test = y_data[split_index:]

# 4. మోడల్ ని ట్రైన్ చేయడం (కేవలం 80% డేటా తోనే)
print("Training the Multi-Linear AI Engine...")
model = MyLinearRegression(learning_rate=0.001)
model.fit(X_train, y_train, epochs=20000)

# 5. The Actual Exam (Testing the model on unseen 20% data)
y_test_pred = model.predict(X_test)

# ప్రతి రికార్డ్ కి ఎంత తప్పు చేసిందో లెక్కించి, దాని ఆవరేజ్ తీయడం (MAE)
test_error = np.mean(np.abs(y_test - y_test_pred))
print(f"Test Exam Result: On average, the model's price prediction is off by ₹ {test_error:.2f}")

print(f"Engine Trained! \nFound Weights: {model.weights} \nBase Fare (c): {model.c}")

# ---------------------------------------------------------
# BLOCK 3 & 4: The Front-End UI (Using Gradio Blocks for a clean layout)
# ---------------------------------------------------------
def get_cab_price(distance, traffic, weather):
    input_features = np.array([distance, traffic, weather])
    price = model.predict(input_features) 
    return f"₹ {round(price, 2)}"

# gr.Blocks వాడితే లేఅవుట్ మన కంట్రోల్ లో ఉంటుంది
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🚖 Tirupati Cab Price Predictor")
    gr.Markdown("### Core AI Engine:")
    
    with gr.Row(): # ఈ మూడు పక్కపక్కన వస్తాయి
        dist_input = gr.Slider(minimum=1, maximum=50, value=5, label="Distance (in KM)")
        traf_input = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Traffic Level (1=Free, 5=Jam)")
        weat_input = gr.Dropdown(choices=[0, 1], value=0, label="Weather (0=Clear, 1=Rain)")
        
    predict_btn = gr.Button("Calculate Price 🚀", variant="primary")
    output_price = gr.Text(label="Predicted Cab Price")
    
    # బటన్ నొక్కినప్పుడు ఫంక్షన్ రన్ అవుతుంది
    predict_btn.click(fn=get_cab_price, inputs=[dist_input, traf_input, weat_input], outputs=output_price)

if __name__ == "__main__":
    interface.launch()