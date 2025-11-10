# Amazon ML Challenge 2025: Smart Product Pricing

This repository contains my solution for the Amazon ML Challenge 2025 (Smart Product Pricing). The objective was to predict the price of e-commerce products using a multimodal dataset consisting of product images and textual descriptions.

This solution implements an end-to-end deep learning pipeline using PyTorch, Hugging Face Transformers, and Weights & Biases for experiment tracking.

## 🏆 Final Result

* **Final Rank:** 341 / 82,000+ Teams (Top 0.5%)
* **Metric:** Symmetric Mean Absolute Percentage Error (SMAPE)
* **Final Score:** 46.38 on the private leaderboard

## 🚀 Solution Overview

My approach consists of a multimodal fusion model that leverages state-of-the-art pretrained encoders for both vision and text data.

### 1. Model Architecture

The model fuses features from two separate encoders and passes them through a regression head to predict the final price.

* **Vision Encoder:** EfficientNet-B4 (pretrained on ImageNet)
   * The final classification layer is replaced with an `nn.Identity()` layer to extract 1792-dimensional feature embeddings from the images.

* **Text Encoder:** BERT (bert-base-uncased) (pretrained, from Hugging Face)
   * The `[CLS]` token's output embedding (768 dimensions) is used to represent the product's `catalog_content`.

* **Fusion & Regression Head:**
   1. The image and text embeddings are concatenated to form a single 2560-dimensional vector (`1792 + 768`).
   2. This combined vector is passed through a regression head consisting of `LayerNorm`, `Linear` layers, `GELU` activation, and `Dropout` to predict a single continuous value.
   3. **Target Transformation:** The model is trained to predict the `log1p` of the price to stabilize training and handle the skewed price distribution. The final output is reversed using `expm1` to get the actual price.
```
[Product Image] ----> [EfficientNet-B4] ----> [Image Embedding (1792-dim)] ----+
                                                                               |
                                                                               |--> [Concatenate] --> [Regression Head] --> [Predicted Log-Price]
                                                                               |
[Product Text]  ----> [BERT]            ----> [Text Embedding (768-dim)]  ----+
```

### 2. Training & Validation Pipeline

The entire training and inference process is managed by `train.py` and configured by `config.yaml`.

* **Configuration Driven:** All hyperparameters (learning rates, batch size, model names, paths) are managed in `config.yaml` for easy tuning and reproducibility.

* **Custom Dataset:** A custom PyTorch `Dataset` class (`ECommerceDataset`) loads images from disk and tokenizes text on-the-fly.

* **Image Augmentation:** Training images are augmented using `torchvision.transforms`, including `RandomHorizontalFlip`, `ColorJitter`, and `RandomAffine`.

* **Differential Learning Rates:** A lower learning rate is used for the pretrained encoder backbones (`1e-5` for vision, `2e-5` for text) and a higher learning rate (`1e-4`) is used for the newly added regression head to promote faster convergence.

* **Advanced Training:**
   * **Warmup Epochs:** The backbones are frozen for the first 2 epochs to allow the regression head to stabilize.
   * **Automatic Mixed Precision (AMP):** `torch.cuda.amp` is used to speed up training and reduce memory usage.
   * **Early Stopping:** Training stops automatically if the validation SMAPE score does not improve for 5 consecutive epochs.
   * **Optimizer:** AdamW with weight decay.
   * **Scheduler:** `ReduceLROnPlateau` monitors the validation SMAPE and reduces the learning rate on plateaus.

### 3. MLOps: Experiment Tracking

* **Weights & Biases (wandb):** The `train.py` script is fully integrated with `wandb`.
* **Logs:** All key metrics (training loss, validation loss, SMAPE), gradients, and hyperparameters are logged for each run.
* **Artifacts:** The best-performing model (`price_predictor.pth`) is saved as a `wandb` artifact for versioning and easy retrieval.

## 🛠️ Key Technologies

* **Core:** Python, PyTorch
* **NLP:** Hugging Face `transformers` (for BERT)
* **Vision:** `torchvision` (for EfficientNet-B4 & transforms)
* **MLOps:** Weights & Biases (wandb)
* **Data Handling:** `pandas`, `numpy`, `scikit-learn`, `PIL`
* **Configuration:** `yaml`

## 📂 Project Structure
```
.
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   ├── train.csv
│   └── test.csv
├── outputs/
│   ├── price_predictor.pth
│   └── submission.csv
├── config.yaml           # All hyperparameters and paths
├── get_model.py          # Dynamically loads vision/text encoders
├── train.py              # Main script: loads data, trains model, runs inference
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🏃 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/CoderXAyush/Amazon-ML-Challenge-2025.git
cd Amazon-ML-Challenge-2025
```

### 2. Set up the environment

It is recommended to use a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `torch`
- `torchvision`
- `transformers`
- `pandas`
- `numpy`
- `scikit-learn`
- `wandb`
- `pyyaml`
- `tqdm`
- `Pillow`

### 4. Download Data

* Place `train.csv` and `test.csv` in the `data/` folder.
* Download the images into `data/images/train/` and `data/images/test/` (e.g., using the `utils.py` from the challenge).

### 5. Configure

* Review `config.yaml` and adjust paths or hyperparameters as needed.
* Log in to Weights & Biases (optional, but recommended):
```bash
wandb login
```

### 6. Run Training & Inference
```bash
python train.py
```

The script will:
1. Train the model based on `config.yaml`.
2. Save the best model to `outputs/price_predictor.pth`.
3. Generate `outputs/submission.csv` using the trained model.

## 📊 Results & Performance

The multimodal approach with differential learning rates and careful data augmentation achieved:
- **Top 0.5%** ranking among 82,000+ teams
- **SMAPE Score:** 46.38 on private leaderboard
- Effective handling of price distribution skewness through log transformation
- Robust generalization through early stopping and learning rate scheduling

## 🔍 Key Insights

1. **Multimodal Fusion:** Combining visual and textual features significantly improved performance over single-modality approaches.
2. **Transfer Learning:** Pretrained models (EfficientNet-B4 and BERT) provided strong baseline features.
3. **Target Engineering:** Log transformation of prices was crucial for handling the skewed distribution.
4. **Training Strategy:** Warmup epochs with frozen backbones prevented overfitting and improved convergence.
5. **Experiment Tracking:** Weights & Biases integration enabled systematic hyperparameter tuning.

## 🙏 Acknowledgments

* Amazon ML Challenge 2025 organizers
* Hugging Face for the Transformers library
* PyTorch and torchvision teams
* Weights & Biases for experiment tracking tools

