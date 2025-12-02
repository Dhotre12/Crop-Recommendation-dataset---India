import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import joblib
import sklearn  # Added to ensure context for unpickling

# ==============================================================================
# 1. MODEL ARCHITECTURES (Must match the Notebook exactly)
# ==============================================================================

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

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

@st.cache_resource
def load_system_data(filename='india_crop_system.pth'):
    if not os.path.exists(filename):
        return None
    
    # Load checkpoint
    # FIXED: Added weights_only=False to allow loading sklearn objects in PyTorch 2.6+
    checkpoint = torch.load(filename, map_location=torch.device('cpu'), weights_only=False)
    
    # Extract Metadata
    metadata = checkpoint['metadata']
    input_size = metadata['input_size']
    num_classes = metadata['num_classes']
    
    # Reconstruct Models
    models = {}
    state_dicts = checkpoint['models_state_dict']
    
    # Init blank models
    model_configs = {
        'CNN': ImprovedCropCNN(input_size, num_classes),
        'LSTM': ImprovedCropLSTM(input_size, 128, 2, num_classes),
        'GRU': ImprovedCropGRU(input_size, 128, 2, num_classes),
        'Transformer': CropTransformer(input_size, 128, 4, 3, num_classes),
        'ResidualMLP': ResidualMLP(input_size, 256, num_classes),
        'Hybrid': HybridCNNLSTM(input_size, num_classes)
    }
    
    # Load weights
    for name, model in model_configs.items():
        if name in state_dicts:
            model.load_state_dict(state_dicts[name])
            model.eval()
            models[name] = model
            
    return {
        'models': models,
        'accuracies': checkpoint['accuracies'],
        'best_name': checkpoint['best_model_name'],
        'scaler': metadata['scaler'],
        'encoder': metadata['label_encoder'],
        'feature_names': metadata['feature_names'],
        'class_names': metadata['class_names']
    }

def predict_crop(system_data, inputs):
    """
    Predicts using the BEST model.
    """
    # 1. Preprocess
    scaler = system_data['scaler']
    le = system_data['encoder']
    best_model_name = system_data['best_name']
    best_model = system_data['models'][best_model_name]
    
    input_array = np.array(inputs).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    tensor_input = torch.FloatTensor(scaled_input)
    
    # 2. Predict
    with torch.no_grad():
        outputs = best_model(tensor_input)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, 1)
        
        # Get top 3 for detail
        top3_probs, top3_idxs = torch.topk(probs, 3)
    
    # 3. Format Results
    top_pred = le.inverse_transform([top_idx.item()])[0]
    
    top3_results = []
    for i in range(3):
        idx = top3_idxs[0][i].item()
        p = top3_probs[0][i].item()
        name = le.inverse_transform([idx])[0]
        top3_results.append((name, p))
        
    return top_pred, top_prob.item(), top3_results

def get_all_model_predictions(system_data, inputs):
    """
    Gets predictions from ALL models for comparison.
    """
    scaler = system_data['scaler']
    le = system_data['encoder']
    
    input_array = np.array(inputs).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    tensor_input = torch.FloatTensor(scaled_input)
    
    results = []
    
    for name, model in system_data['models'].items():
        with torch.no_grad():
            out = model(tensor_input)
            prob, idx = torch.max(torch.softmax(out, dim=1), 1)
            pred_class = le.inverse_transform([idx.item()])[0]
            
            # Get test accuracy for this model
            test_acc = system_data['accuracies'].get(name, 0.0)
            
            results.append({
                'Algorithm': name,
                'Prediction': pred_class.title(),
                'Confidence': f"{prob.item():.2%}",
                'Test Accuracy': test_acc
            })
            
    return pd.DataFrame(results)

# ==============================================================================
# 3. STREAMLIT UI
# ==============================================================================

st.set_page_config(page_title="India Crop Recommender", page_icon="🌾", layout="wide")

st.title("🌾 Intelligent Crop Recommendation System (India)")
st.markdown("Using Advanced Deep Learning Architectures (CNN, LSTM, Transformer, etc.)")

# Load System
system_path = 'india_crop_system.pth'
system_data = load_system_data(system_path)

if system_data is None:
    st.error(f"❌ Model file '{system_path}' not found!")
    st.info("⚠️ Please run the 'System Export Script' in your notebook to generate this file.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("📝 Soil & Climate Data")

def user_input_features():
    # Ranges based on typical India dataset values
    N = st.sidebar.slider('Nitrogen (N)', 0, 140, 90)
    P = st.sidebar.slider('Phosphorous (P)', 5, 145, 42)
    K = st.sidebar.slider('Potassium (K)', 5, 205, 43)
    temp = st.sidebar.slider('Temperature (°C)', 8.0, 45.0, 20.8)
    humidity = st.sidebar.slider('Humidity (%)', 10.0, 100.0, 82.0)
    ph = st.sidebar.slider('pH Level', 3.5, 10.0, 6.5)
    rainfall = st.sidebar.slider('Rainfall (mm)', 20.0, 300.0, 202.9)
    
    data = [N, P, K, temp, humidity, ph, rainfall]
    return data

input_data = user_input_features()

# --- Main Page ---

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Current Input")
    input_df = pd.DataFrame([input_data], columns=['N', 'P', 'K', 'Temp', 'Humid', 'pH', 'Rain'])
    st.dataframe(input_df, hide_index=True)
    
    if st.button("🔍 Recommend Crop", type="primary", use_container_width=True):
        with st.spinner('Analyzing soil patterns...'):
            best_crop, confidence, top3 = predict_crop(system_data, input_data)
            
            # Display Primary Result
            st.success(f"### 🏆 Recommended: **{best_crop.title()}**")
            st.metric(label="Confidence Score", value=f"{confidence:.2%}")
            
            st.markdown("---")
            st.caption(f"Based on **{system_data['best_name']}** model (Highest Accuracy)")
            
            # Top 3 probabilities chart
            st.subheader("Alternative Options")
            probs_df = pd.DataFrame(top3, columns=['Crop', 'Probability'])
            probs_df['Probability'] = probs_df['Probability'] * 100
            st.bar_chart(probs_df.set_index('Crop'))

with col2:
    if st.session_state.get('button_clicked') or True: # Show by default or on click
        st.subheader("🤖 Multi-Model Consensus")
        
        comparison_df = get_all_model_predictions(system_data, input_data)
        
        # Formatting for display
        comparison_df['Test Accuracy'] = comparison_df['Test Accuracy'].apply(lambda x: f"{x:.2%}")
        
        # Highlight best model row
        best_name = system_data['best_name']
        def highlight_best(row):
            return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(row) if row['Algorithm'] == best_name else [''] * len(row)

        st.dataframe(
            comparison_df.style.apply(highlight_best, axis=1), 
            use_container_width=True,
            hide_index=True
        )
        
        st.info("The table above shows what every trained algorithm predicts for your input, along with their historical test accuracy.")