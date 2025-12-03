import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

# --- 1. Model Definitions (Must match the training notebook exactly) ---

class ImprovedCropCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ImprovedCropCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ImprovedCropLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ImprovedCropLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ImprovedCropGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ImprovedCropGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        x = self.relu(self.batchnorm1(self.fc1(context)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CropTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes):
        super(CropTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 10, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x) * np.sqrt(128)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ResidualMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.res_block1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size))
        self.res_block2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(x + self.res_block1(x))
        x = self.dropout(x)
        x = self.relu(x + self.res_block2(x))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

class HybridCNNLSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HybridCNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cnn_bn1 = nn.BatchNorm1d(64)
        self.cnn_bn2 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(input_size, 128, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(128 + 256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        cnn_x = x.unsqueeze(1)
        cnn_x = self.relu(self.cnn_bn1(self.conv1(cnn_x)))
        cnn_x = self.relu(self.cnn_bn2(self.conv2(cnn_x)))
        cnn_x = self.pool(cnn_x).view(cnn_x.size(0), -1)
        
        lstm_x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_x)
        lstm_features = lstm_out[:, -1, :]
        
        combined = torch.cat((cnn_x, lstm_features), dim=1)
        x = self.relu(self.batchnorm1(self.fc1(combined)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

# --- 2. Helper Functions ---
@st.cache_resource
def load_system():
    try:
        # FIX: We set weights_only=False because the file contains Scikit-learn objects (Scaler, Encoder)
        # which are not allowed by default in PyTorch 2.6+
        checkpoint = torch.load('india_crop_system.pth', map_location=torch.device('cpu'), weights_only=False)
        return checkpoint
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return None

# --- 3. UI Layout ---
st.set_page_config(page_title="Indian Crop Recommendation", layout="wide")

st.title("🌱 Indian Crop Recommendation System")
st.markdown("Analyze soil and weather conditions using multiple Deep Learning models to predict the best crop.")

# Load Data
system_data = load_system()

if system_data is None:
    st.error("⚠️ `india_crop_system.pth` not found. Please run the training notebook first to generate the system file.")
    st.stop()

# Extract components
metadata = system_data['metadata']
scaler = metadata['scaler']
label_encoder = metadata['label_encoder']
saved_models = system_data['models_state_dict']
test_accuracies = system_data['accuracies']
best_model_name = system_data['best_model_name']

input_size = metadata['input_size']   # Should be 7
num_classes = metadata['num_classes'] # Should be 22

# --- Sidebar Inputs ---
st.sidebar.header("🌍 Soil & Weather Conditions")

# Inputs matching the dataset features: N, P, K, temperature, humidity, ph, rainfall
N = st.sidebar.slider("Nitrogen (N)", 0, 140, 40)
P = st.sidebar.slider("Phosphorus (P)", 5, 145, 50)
K = st.sidebar.slider("Potassium (K)", 5, 205, 50)
temperature = st.sidebar.slider("Temperature (°C)", 8.0, 45.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 14.0, 100.0, 70.0)
ph = st.sidebar.slider("pH Level", 3.5, 10.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

# --- Main Inference Block ---
if st.button("🚀 Analyze & Predict", use_container_width=True):
    
    # 1. Prepare Input
    try:
        # Feature vector: [N, P, K, temperature, humidity, ph, rainfall]
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        features_scaled = scaler.transform(features)
        input_tensor = torch.FloatTensor(features_scaled)
        
        # 2. Run All Models
        # Dictionary mapping name to Class Instance with specific hyperparameters from notebook
        model_instances = {
            'CNN': ImprovedCropCNN(input_size, num_classes),
            'LSTM': ImprovedCropLSTM(input_size, 128, 2, num_classes),
            'GRU': ImprovedCropGRU(input_size, 128, 2, num_classes),
            'Transformer': CropTransformer(input_size, 128, 4, 3, num_classes),
            'ResidualMLP': ResidualMLP(input_size, 256, num_classes),
            'Hybrid': HybridCNNLSTM(input_size, num_classes)
        }
        
        results = []
        best_confidence = -1
        best_probs = None
        
        progress_bar = st.progress(0)
        model_names = list(model_instances.keys())
        
        for idx, name in enumerate(model_names):
            model = model_instances[name]
            
            # Load Weights
            if name in saved_models:
                model.load_state_dict(saved_models[name])
                model.eval()
                
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1)
                    confidence, predicted_idx = torch.max(probs, 1)
                    
                    pred_class = label_encoder.inverse_transform([predicted_idx.item()])[0]
                    conf_score = confidence.item() * 100
                    
                    # Formatting accuracy
                    acc_val = test_accuracies.get(name, 0.0)
                    if isinstance(acc_val, float):
                        acc_str = f"{acc_val*100:.2f}%"
                    else:
                        acc_str = str(acc_val)

                    results.append({
                        "Algorithm": name,
                        "Predicted Crop": pred_class,
                        "Confidence": f"{conf_score:.2f}%",
                        "Test Set Accuracy": acc_str,
                        "_raw_conf": conf_score
                    })
                    
                    # Track best model based on confidence
                    if conf_score > best_confidence:
                        best_confidence = conf_score
                        best_probs = probs[0]
            else:
                 results.append({
                    "Algorithm": name,
                    "Predicted Crop": "Error (Weights Missing)",
                    "Confidence": "0%",
                    "Test Set Accuracy": "N/A",
                    "_raw_conf": 0
                })
            
            progress_bar.progress((idx + 1) / len(model_names))
            
        progress_bar.empty()
        
        # 3. Display Results
        st.divider()
        
        # Create DataFrame
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values(by="_raw_conf", ascending=False).drop(columns=["_raw_conf"])
        
        # Highlight Top Result
        top_row = res_df.iloc[0]
        st.subheader(f"🏆 Top Prediction: {top_row['Predicted Crop']}")
        st.success(f"**Recommended Crop: {top_row['Predicted Crop']}** (Confidence: {top_row['Confidence']} using {top_row['Algorithm']})")
        
        # Show Comparison Table
        st.write("### 📊 Model Comparison")
        
        def highlight_best_row(row):
            is_best = row['Algorithm'] == top_row['Algorithm']
            # Green background with White text and Bold font for visibility
            return ['background-color: #2E7D32; color: white; font-weight: bold' if is_best else '' for _ in row]

        st.dataframe(res_df.style.apply(highlight_best_row, axis=1), use_container_width=True)
        
        # 4. Top 3 Alternatives
        st.divider()
        st.subheader(f"🥇 Top 3 Alternatives ({top_row['Algorithm']})")
        
        if best_probs is not None:
            top3_prob, top3_idx = torch.topk(best_probs, 3)
            
            cols = st.columns(3)
            for i in range(3):
                crop_name = label_encoder.inverse_transform([top3_idx[i].item()])[0]
                prob_val = top3_prob[i].item() * 100
                
                with cols[i]:
                    st.metric(label=f"Rank #{i+1}", value=crop_name, delta=f"{prob_val:.1f}% Probability")
                    st.progress(prob_val / 100)
                    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")