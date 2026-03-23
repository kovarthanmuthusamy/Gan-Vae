#!/usr/bin/env python3
"""
Single script to generate and view VAE architecture diagrams in a browser.
Usage: python3 view_architecture.py
"""

import os
import time
import subprocess
import webbrowser
import signal
import sys
import base64
from pathlib import Path

def encode_image(img_path):
    """Encode image to base64 data URI"""
    try:
        with open(img_path, 'rb') as f:
            img_data = f.read()
            encoded = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return ""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VAE Encoder, Decoder, and Losses</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
        }
        .container {
            max-width: 1500px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.4em;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .tabs {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
            background: #ecf0f1;
            border-radius: 10px;
            padding: 5px;
            flex-wrap: wrap;
        }
        .tab {
            padding: 14px 18px;
            margin: 5px;
            border: none;
            background: transparent;
            cursor: pointer;
            border-radius: 8px;
            font-size: 0.95em;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .tab.active {
            background: #3498db;
            color: white;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }
        .tab:hover:not(.active) {
            background: #bdc3c7;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .info-box {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 25px;
        }
        .info-box ul {
            margin: 0;
            padding-left: 20px;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            font-size: 0.9em;
        }
        .mermaid {
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.05);
            margin-bottom: 25px;
        }
        .result-images {
            display: flex;
            flex-direction: column;
            gap: 30px;
        }
        .result-item {
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .result-item h3 {
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 15px;
        }
        .result-item img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        }
        .occupancy-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        .occupancy-item {
            text-align: center;
        }
        .occupancy-item h4 {
            color: #34495e;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Product of Experts Multi-Modal VAE</h1>
        <p class="subtitle">Multi-modal VAE with Product of Experts fusion (precision-weighted uncertainty). 4 modality branches, private+shared latent spaces (128D total), enhanced occupancy encoder (1.1M params), MAE impedance loss, no skip connections.</p>

        <div class="tabs">
            <button class="tab active" onclick="showTab(event, 'encoder')">🧩 Encoder</button>
            <button class="tab" onclick="showTab(event, 'decoder')">🎯 Decoder</button>
            <button class="tab" onclick="showTab(event, 'pipeline')">📊 Data Pipeline</button>
            <button class="tab" onclick="showTab(event, 'losses')">⚖️ Loss Functions</button>
            <button class="tab" onclick="showTab(event, 'results')">📊 Results</button>
        </div>

        <div id="encoder" class="tab-content active">
            <div class="info-box">
                <h2>🧩 Product of Experts (PoE) Multi-Modal Encoder</h2>
                <p>Four branches (heatmap, occupancy, impedance, max_value) encode independently. Each modality predicts the shared latent space with uncertainty, combined via precision-weighted Product of Experts fusion.</p>
            </div>

            <div class="mermaid">
graph TB
    %% Input Layer
    INPUT[🎯 MULTI-MODAL INPUTS]
    
    INPUT --> H_IN[Heatmap<br/>64x64x2]
    INPUT --> O_IN[Occupancy<br/>7x8x1]
    INPUT --> I_IN[Impedance<br/>231x1]
    INPUT --> M_IN[MaxValue<br/>scalar]
    
    %% Encoder Branches
    H_IN --> H_ENC["🔥 Heatmap Encoder<br/>CNN: 2→16→32→64<br/>FC: 4096→64"]
    O_IN --> O_ENC["🗺️ Occupancy Encoder<br/>CNN: 1→16→32→64→128<br/>FC: 7168→512→256→128→64"]
    I_IN --> I_ENC["📊 Impedance Encoder<br/>MLP: 231→128→128→64<br/>with BatchNorm"]
    M_IN --> M_ENC["📡 MaxValue Encoder<br/>MLP: 1→64→128→64"]
    
    H_ENC --> H_FEAT[Heatmap Feat 64D]
    O_ENC --> O_FEAT[Occupancy Feat 64D]
    I_ENC --> I_FEAT[Impedance Feat 64D]
    M_ENC --> M_FEAT[MaxValue Feat 64D]
    
    %% Private Latent Spaces
    H_FEAT --> H_PRIV["Private Space<br/>μ_h, σ²_h 16D"]
    O_FEAT --> O_PRIV["Private Space<br/>μ_o, σ²_o 16D"]
    I_FEAT --> I_PRIV["Private Space<br/>μ_i, σ²_i 16D"]
    M_FEAT --> M_PRIV["Private Space<br/>μ_m, σ²_m 16D"]
    
    %% Shared Latent Spaces for PoE
    H_FEAT --> H_SHARED["Shared Prediction<br/>μ_h, σ²_h 64D"]
    O_FEAT --> O_SHARED["Shared Prediction<br/>μ_o, σ²_o 64D"]
    I_FEAT --> I_SHARED["Shared Prediction<br/>μ_i, σ²_i 64D"]
    M_FEAT --> M_SHARED["Shared Prediction<br/>μ_m, σ²_m 64D"]
    
    %% Product of Experts Fusion
    H_SHARED --> POE["⚡ Product of Experts<br/>Precision-Weighted Fusion<br/>1/σ²_comb = Σ1/σ²_i<br/>μ_comb = σ²_comb·Σμ_i/σ²_i"]
    O_SHARED --> POE
    I_SHARED --> POE
    M_SHARED --> POE
    
    POE --> SHARED["Shared Space<br/>μ_s, σ²_s 64D"]
    
    %% Concatenation
    H_PRIV --> CONCAT["Concatenate<br/>Private + Shared"]
    O_PRIV --> CONCAT
    I_PRIV --> CONCAT
    M_PRIV --> CONCAT
    SHARED --> CONCAT
    
    CONCAT --> FINAL["Final Distribution<br/>μ, logvar 128D<br/>64 private + 64 shared"]
    
    %% Reparameterization
    FINAL --> SAMPLE["🎲 Reparameterize<br/>z = μ + σ·ε, ε~N(0,1)"]
    SAMPLE --> LATENT["🧬 Latent z<br/>128D"]

    %% Styling
    style INPUT fill:#e0e0e0,stroke:#333,stroke-width:3px
    style H_ENC fill:#ffb3ba,stroke:#333,stroke-width:2px
    style O_ENC fill:#baffc9,stroke:#333,stroke-width:2px
    style I_ENC fill:#ffd8a8,stroke:#333,stroke-width:2px
    style M_ENC fill:#ffaaee,stroke:#333,stroke-width:2px
    style H_PRIV fill:#ffe0e0,stroke:#333,stroke-width:2px
    style O_PRIV fill:#e0ffe0,stroke:#333,stroke-width:2px
    style I_PRIV fill:#fff0e0,stroke:#333,stroke-width:2px
    style M_PRIV fill:#ffe0ff,stroke:#333,stroke-width:2px
    style H_SHARED fill:#ffcccb,stroke:#333,stroke-width:2px
    style O_SHARED fill:#ccffcc,stroke:#333,stroke-width:2px
    style I_SHARED fill:#ffe6cc,stroke:#333,stroke-width:2px
    style M_SHARED fill:#ffccff,stroke:#333,stroke-width:2px
    style POE fill:#ff6b6b,color:#fff,stroke:#333,stroke-width:3px
    style SHARED fill:#ffa94d,stroke:#333,stroke-width:2px
    style CONCAT fill:#aed9e0,stroke:#333,stroke-width:2px
    style FINAL fill:#95e1d3,stroke:#333,stroke-width:2px
    style SAMPLE fill:#ffeaa7,stroke:#333,stroke-width:2px
    style LATENT fill:#b4a7f5,stroke:#333,stroke-width:3px
            </div>

            <div class="info-box">
                <h3>📊 Product of Experts Encoder Architecture:</h3>
                <ul>
                    <li><strong>🔥 Heatmap Branch:</strong> CNN (2→16→32→64 channels) + FC → 64D features</li>
                    <li><strong>🗺️ Occupancy Branch (Enhanced):</strong> 4 Conv layers (1→16→32→64→128) + 4 FC layers (7168→512→256→128→64) [~1.1M params, 17× original]</li>
                    <li><strong>📊 Impedance Branch:</strong> 3 FC layers with BatchNorm (231→128→128→64), NO skip connections</li>
                    <li><strong>📡 MaxValue Branch:</strong> MLP (1→64→128→64)</li>
                    <li><strong>🎯 Private Latent Spaces:</strong> Each modality → independent (μ_priv, σ²_priv) of 16D → Total 64D private</li>
                    <li><strong>⚡ Product of Experts (PoE):</strong> Each modality predicts shared space (μ_i, σ²_i) of 64D → Combined via precision-weighted fusion</li>
                    <li><strong>📐 PoE Formula:</strong> 1/σ²_combined = Σ(1/σ²_i), μ_combined = σ²_combined × Σ(μ_i/σ²_i)</li>
                    <li><strong>🧬 Total Latent:</strong> 128D = 64D private (modality-specific) + 64D shared (PoE-fused cross-modal)</li>
                    <li><strong>✨ Benefits:</strong> Uncertainty-aware fusion, robust to noisy modalities, automatic confidence weighting</li>
                </ul>
            </div>
        </div>

        <div id="decoder" class="tab-content">
            <div class="info-box">
                <h2>🎯 Multi-Modal Decoder with Shared-to-Private Reconstruction</h2>
                <p>Latent z (128D: 64 private + 64 shared) → Reverse fusion → 4 independent reconstruction heads</p>
            </div>

            <div class="mermaid">
graph TB
    LATENT[🧬 Latent z<br/>128D<br/>64 priv + 64 shared]
    
    subgraph "Reverse Fusion Layer"
        LATENT --> REVERSE_FC[FC 128→256]
        REVERSE_FC --> REVERSE_FEAT[Shared Features<br/>256D]
    end
    
    subgraph "Heatmap Decoder"
        REVERSE_FEAT --> H_FC[FC 256→4096]
        H_FC --> H_RESHAPE[Reshape 64×8×8]
        H_RESHAPE --> H_DECONV1[ConvT 64→32]
        H_DECONV1 --> H_DECONV2[ConvT 32→16]
        H_DECONV2 --> H_DECONV3[ConvT 16→2]
        H_DECONV3 --> H_SIG[Sigmoid]
        H_SIG --> H_OUT[🔥 Heatmap<br/>64x64x2]  
    end
    
    subgraph "Occupancy Decoder (Enhanced)"
        REVERSE_FEAT --> O_FC1[FC 256→512]
        O_FC1 --> O_FC2[FC 512→1024]
        O_FC2 --> O_FC3[FC 1024→7168]
        O_FC3 --> O_RESHAPE[Reshape 128×7×8]
        O_RESHAPE --> O_DECONV1[Conv 128→64]
        O_DECONV1 --> O_DECONV2[Conv 64→32]
        O_DECONV2 --> O_DECONV3[Conv 32→16]
        O_DECONV3 --> O_DECONV4[Conv 16→8]
        O_DECONV4 --> O_DECONV5[Conv 8→1]
        O_DECONV5 --> O_OUT[🗺️ Occupancy Logits<br/>7x8x1]
    end
    
    subgraph "Impedance Decoder"
        REVERSE_FEAT --> I_FC1[FC 256→128]
        I_FC1 --> I_FC2[FC 128→128]
        I_FC2 --> I_FC3[FC 128→231]
        I_FC3 --> I_SIG[Sigmoid]
        I_SIG --> I_OUT[📊 Impedance<br/>231x1]
    end
    
    subgraph "MaxValue Decoder"
        REVERSE_FEAT --> M_FC1[FC 256→64]
        M_FC1 --> M_FC2[FC 64→32]
        M_FC2 --> M_FC3[FC 32→1]
        M_FC3 --> M_SIG[Sigmoid]
        M_SIG --> M_OUT[📡 Max Value<br/>scalar]
    end
    
    style LATENT fill:#b4a7f5
    style REVERSE_FEAT fill:#e6f3ff
    style H_OUT fill:#ffb3ba
    style O_OUT fill:#baffc9
    style I_OUT fill:#ffd8a8
    style M_OUT fill:#ffaaee
            </div>

            <div class="info-box">
                <h3>📊 Decoder Architecture:</h3>
                <ul>
                    <li><strong>� Reverse Fusion:</strong> Latent 128D → FC to 256D shared features with BatchNorm</li>
                    <li><strong>🔥 Heatmap Decoder:</strong> 256→4096 → Reshape 64×8×8 → 3 ConvTranspose layers → Sigmoid → 64x64x2</li>
                    <li><strong>🗺️ Occupancy Decoder (Enhanced):</strong> 3 FC layers (256→512→1024→7168) + 5 Conv layers (128→64→32→16→8→1) [~450K params]</li>
                    <li><strong>📊 Impedance Decoder:</strong> 3 FC layers (256→128→128→231), NO skip connections, Sigmoid output</li>
                    <li><strong>📡 MaxValue Decoder:</strong> MLP (256→64→32→1) with Sigmoid → scalar output</li>
                    <li><strong>🎯 Design:</strong> All decoders independent, occupancy uses logits (BCEWithLogitsLoss), others use Sigmoid activation</li>
                </ul>
            </div>
        </div>

        <div id="pipeline" class="tab-content">
            <div class="info-box">
                <h2>📊 Data Pipeline</h2>
                <p>End-to-end workflow: normalization, training, and inference.</p>
            </div>

            <div class="mermaid">
graph TB
    subgraph "� Dataset Statistics"
        DS[15K Samples<br/>Quality: 8/8 EXCELLENT]
        DS --> H_STATS[Heatmap: CV=2.81<br/>exceptional diversity]
        DS --> I_STATS[Impedance: CV=0.38<br/>high diversity<br/>skew 2.72→-0.10]
        DS --> O_STATS[Occupancy: CV=0.31<br/>100% unique patterns<br/>46.4% density]
    end
    
    subgraph "🗂️ Data Loading"
        HEATMAP[Load Heatmap<br/>64x64x2]
        OCCUPANCY[Load Occupancy<br/>7x8x1]
        IMPEDANCE[Load Impedance<br/>231x1 normalized]
        MAXVALUE[Load MaxValue<br/>scalar]
        
        HEATMAP --> DATALOADER[DataLoader<br/>batch_size=64]
        OCCUPANCY --> DATALOADER
        IMPEDANCE --> DATALOADER
        MAXVALUE --> DATALOADER
    end
    
    subgraph "🏋️ Training Pipeline with PoE"
        DATALOADER --> ENCODER[VAE Encoder<br/>4 branches]
        ENCODER --> POE[Product of Experts<br/>precision-weighted fusion]
        POE --> LATENT[Latent z<br/>128D<br/>64 priv + 64 shared]
        
        LATENT --> DECODER[VAE Decoder<br/>4 heads]
        
        DECODER --> H_PRED[Heatmap Pred 64x64x2]
        DECODER --> O_PRED[Occupancy Pred 7x8x1]
        DECODER --> I_PRED[Impedance Pred 231x1]
        DECODER --> M_PRED[MaxValue Pred scalar]
        
        H_PRED --> L_H[MSE Loss]
        HEATMAP --> L_H
        
        O_PRED --> L_O[BCE Loss]
        OCCUPANCY --> L_O
        
        I_PRED --> L_I[MAE Loss]
        IMPEDANCE --> L_I
        
        M_PRED --> L_M[MSE Loss]
        MAXVALUE --> L_M
        
        LATENT --> KL_DIV[KL Divergence]
        
        L_H --> TOTAL[Total Loss]
        L_O --> TOTAL
        L_I --> TOTAL
        L_M --> TOTAL
        KL_DIV --> TOTAL
        
        TOTAL --> BACKPROP[Backprop<br/>Adam lr=1e-5]
    end
    
    subgraph "🔮 Inference"
        Z_SAMPLE[Sample z ~ N(0,1)]
        Z_SAMPLE --> DEC_INF[VAE Decoder]
        DEC_INF --> H_OUT[Heatmap 64x64x2]
        DEC_INF --> O_OUT[Occupancy 7x8x1]
        DEC_INF --> I_OUT[Impedance 231x1]
        DEC_INF --> M_OUT[MaxValue scalar]
    end
    
    style DS fill:#e6f3ff
    style POE fill:#ff6b6b,color:#fff
    style LATENT fill:#b4a7f5
    style H_PRED fill:#ffb3ba
    style O_PRED fill:#baffc9
    style I_PRED fill:#ffd8a8
    style M_PRED fill:#ffaaee
    style TOTAL fill:#d4edda
            </div>

            <div class="info-box">
                <h3>📊 Pipeline Details:</h3>
                <ul>
                    <li><strong>🗂️ Dataset:</strong> 15K samples, 8/8 quality score. Heatmap (2ch,64x64), Occupancy (1ch,7x8), Impedance (231D), MaxValue (scalar)</li>
                    <li><strong>📊 Diversity:</strong> Exceptional (CV=2.81 heatmap, 0.38 impedance, 0.31 occupancy). 100% unique patterns</li>
                    <li><strong>🔧 Normalization:</strong> Log-scale + percentile (1-99%) for impedance. Reduced skewness from 2.72 to -0.10</li>
                    <li><strong>🏋️ Training:</strong> Batch size 64, LR 1e-5, 300 epochs. PoE fusion with private+shared latent spaces</li>
                    <li><strong>🎯 Design:</strong> Product of Experts for uncertainty-aware multi-modal fusion, MAE for impedance, enhanced occupancy (1.1M params)</li>
                </ul>
            </div>
        </div>

        <div id="losses" class="tab-content">
            <div class="info-box">
                <h2>⚖️ Loss Functions</h2>
                <p>Multi-task loss with uncertainty-based automatic weighting, specialized losses for class imbalance, and KL annealing.</p>
            </div>

            <div class="mermaid">
graph TB
    subgraph "Heatmap Loss"
        H_PRED[Heatmap Pred<br/>64x64x2]
        H_TARGET[Heatmap Target<br/>64x64x2]
        
        H_PRED --> H_MSE[Mean Squared<br/>Error MSE]
        H_TARGET --> H_MSE
        H_MSE --> H_LOSS[L_heatmap]
    end
    
    subgraph "Occupancy Loss"
        O_PRED[Occupancy Logits<br/>7x8x1]
        O_TARGET[Occupancy Target<br/>7x8x1 binary]
        
        O_PRED --> O_BCE[Binary Cross Entropy<br/>With Logits]
        O_TARGET --> O_BCE
        O_BCE --> O_LOSS[L_occupancy]
    end
    
    subgraph "Impedance Loss"
        I_PRED[Impedance Pred<br/>231x1]
        I_TARGET[Impedance Target<br/>231x1]
        
        I_PRED --> I_MAE[Mean Absolute<br/>Error L1]
        I_TARGET --> I_MAE
        I_MAE --> I_LOSS[L_impedance<br/>changed from MSE to MAE]
    end
    
    subgraph "MaxValue Loss"
        M_PRED[MaxValue Pred<br/>scalar]
        M_TARGET[MaxValue Target<br/>scalar]
        
        M_PRED --> M_MSE[Mean Squared<br/>Error MSE]
        M_TARGET --> M_MSE
        M_MSE --> M_LOSS[L_maxvalue]
    end
    
    subgraph "KL Regularization with Beta Annealing"
        MU[mu<br/>128D]
        LOGVAR[logvar<br/>128D]
        
        MU --> KL_CALC[KL ∼ -0.5 × sum<br/>1 + logvar - μ² - exp logvar]
        LOGVAR --> KL_CALC
        KL_CALC --> KL_RAW[L_KL_raw]
        
        BETA[beta annealing<br/>0.0 → 0.01<br/>epochs 0-100]
        KL_RAW --> KL_MULT[Multiply]
        BETA --> KL_MULT
        KL_MULT --> KL_LOSS[L_KL = beta × L_KL_raw]
    end
    
    H_LOSS --> RECON[Reconstruction Loss<br/>L_recon]
    O_LOSS --> RECON
    I_LOSS --> RECON
    M_LOSS --> RECON
    
    RECON --> TOTAL[Total Loss]
    KL_LOSS --> TOTAL
    
    TOTAL --> FINAL[L_total = L_heatmap + L_occupancy<br/>+ L_impedance + L_maxvalue + beta×L_KL]
    
    style H_LOSS fill:#ffb3ba
    style O_LOSS fill:#baffc9
    style I_LOSS fill:#ffd8a8
    style M_LOSS fill:#ffaaee
    style KL_LOSS fill:#b4a7f5
    style BETA fill:#ffeaa7
    style RECON fill:#e6f3ff
    style TOTAL fill:#d4edda
    style FINAL fill:#d4edda
            </div>

            <div class="info-box">
                <h3>📊 Loss Function Details:</h3>
                <ul>
                    <li><strong>🔥 Heatmap Loss:</strong> MSE (reduction='mean') on normalized heatmap 64x64x2</li>
                    <li><strong>🗺️ Occupancy Loss:</strong> Binary Cross Entropy with Logits (BCEWithLogitsLoss) for binary classification 7x8</li>
                    <li><strong>📊 Impedance Loss:</strong> MAE/L1 Loss (changed from MSE) - better for skewed data with outliers. Normalized data: skewness 2.72→-0.10</li>
                    <li><strong>📡 MaxValue Loss:</strong> MSE (reduction='mean') on scalar prediction</li>
                    <li><strong>🧬 KL Divergence:</strong> Regularizes latent to N(0,1). Formula: -0.5 × sum(1 + logvar - μ² - exp(logvar))</li>
                    <li><strong>⏳ Beta Annealing:</strong> β goes from 0.0 → 0.01 over epochs 0-100 to prevent posterior collapse</li>
                    <li><strong>⚖️ Weighting:</strong> All losses use reduction='mean' for comparable scales. No uncertainty weighting (removed complexity)</li>
                    <li><strong>🎯 Total:</strong> L_total = MSE + BCE + MAE + MSE + β×KL, batch size 64, lr=1e-5, 300 epochs</li>
                </ul>
            </div>
        </div>

        <div id="results" class="tab-content">
            <div class="info-box">
                <h2>📊 Experiment Results</h2>
                <p>Visualization comparisons between generated and real data for heatmaps, impedance profiles, and occupancy maps.</p>
            </div>

            <div class="result-images">
                <div class="result-item">
                    <h3>🔥 Heatmap Comparison</h3>
                    <img src="__HEATMAP_SRC__" alt="Generated vs Real Heatmap">
                </div>

                <div class="result-item">
                    <h3>📈 Impedance Profile Comparison</h3>
                    <img src="__IMPEDANCE_SRC__" alt="Generated vs Real Impedance Profile">
                </div>

                <div class="result-item">
                    <h3>🗺️ Occupancy Map Visualizations</h3>
                    <div class="occupancy-grid">
                        <div class="occupancy-item">
                            <h4>Data Sample 0</h4>
                            <img src="__OCC0_SRC__" alt="Occupancy Map Sample 0">
                        </div>
                        <div class="occupancy-item">
                            <h4>Data Sample 1</h4>
                            <img src="__OCC1_SRC__" alt="Occupancy Map Sample 1">
                        </div>
                        <div class="occupancy-item">
                            <h4>Data Sample 2</h4>
                            <img src="__OCC2_SRC__" alt="Occupancy Map Sample 2">
                        </div>
                        <div class="occupancy-item">
                            <h4>Data Sample 3</h4>
                            <img src="__OCC3_SRC__" alt="Occupancy Map Sample 3">
                        </div>
                        <div class="occupancy-item">
                            <h4>Data Sample 4</h4>
                            <img src="__OCC4_SRC__" alt="Occupancy Map Sample 4">
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Use the tabs to explore encoder flow, decoder hierarchy, composed loss, or experimental results.</p>
            <p>Right-click any diagram or image to export.</p>
        </div>
    </div>

    <script>
        console.log('Starting Mermaid initialization...');
        
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true
            }
        });
        
        console.log('Mermaid initialized');

        // Force render all diagrams by temporarily showing all tabs
        window.addEventListener('DOMContentLoaded', () => {
            console.log('DOM loaded, forcing initial render of all diagrams...');
            
            const allTabs = document.querySelectorAll('.tab-content');
            const originalDisplays = [];
            
            // Temporarily show all tabs
            allTabs.forEach((tab, index) => {
                originalDisplays[index] = tab.style.display;
                tab.style.display = 'block';
            });
            
            // Give Mermaid time to render
            setTimeout(() => {
                // Restore original display states
                allTabs.forEach((tab, index) => {
                    if (!tab.classList.contains('active')) {
                        tab.style.display = 'none';
                    }
                });
                console.log('All diagrams rendered, tabs restored');
            }, 500);
        });

        function showTab(evt, tabName) {
            console.log('Switching to tab:', tabName);
            
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => {
                content.classList.remove('active');
                content.style.display = 'none';
            });

            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));

            const targetTab = document.getElementById(tabName);
            targetTab.classList.add('active');
            targetTab.style.display = 'block';
            evt.target.classList.add('active');
            
            console.log('Tab switched to:', tabName);
        }
    </script>
</body>
</html>
"""

def main():
    print("🔄 Encoding images into HTML...")
    os.makedirs('temp_visuals', exist_ok=True)
    html_path = 'temp_visuals/vae_architecture_diagram.html'
    
    # Encode all images
    heatmap_src = encode_image("experiments/exp012/visuals/generated_vs_real_heatmap.png")
    impedance_src = encode_image("experiments/exp012/visuals/generated_vs_real_impedance_profile.png")
    occ0_src = encode_image("experiments/exp012/visuals/data_sample_0/occupancy_map_visual.png")
    occ1_src = encode_image("experiments/exp012/visuals/data_sample_1/occupancy_map_visual.png")
    occ2_src = encode_image("experiments/exp012/visuals/data_sample_2/occupancy_map_visual.png")
    occ3_src = encode_image("experiments/exp012/visuals/data_sample_3/occupancy_map_visual.png")
    occ4_src = encode_image("experiments/exp012/visuals/data_sample_4/occupancy_map_visual.png")
    
    # Replace placeholders with actual base64 data
    html_content = HTML_TEMPLATE.replace('__HEATMAP_SRC__', heatmap_src)
    html_content = html_content.replace('__IMPEDANCE_SRC__', impedance_src)
    html_content = html_content.replace('__OCC0_SRC__', occ0_src)
    html_content = html_content.replace('__OCC1_SRC__', occ1_src)
    html_content = html_content.replace('__OCC2_SRC__', occ2_src)
    html_content = html_content.replace('__OCC3_SRC__', occ3_src)
    html_content = html_content.replace('__OCC4_SRC__', occ4_src)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("✅ Architecture diagram HTML created with embedded images")
    print(f"   File size: {len(html_content) / (1024*1024):.2f} MB")

    port = 8899
    print(f"🌐 Starting HTTP server on port {port}...")

    # Run server from temp_visuals since images are embedded
    server_process = subprocess.Popen(
        ['python3', '-m', 'http.server', str(port)],
        cwd='temp_visuals',
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(1)

    url = f'http://localhost:{port}/vae_architecture_diagram.html'
    print(f"🚀 Opening diagram in browser: {url}")
    print("📋 Press Ctrl+C to stop the server")

    try:
        webbrowser.open(url)
    except Exception:
        print(f"   Manually open: {url}")

    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped")

if __name__ == '__main__':
    main()
