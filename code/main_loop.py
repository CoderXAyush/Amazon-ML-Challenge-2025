# -*- coding: utf-8 -*-
"""
E-Commerce Price Prediction - A Modular Pipeline Approach.

This script implements a multimodal price prediction model using a highly
modular and functional structure. It features a static Config class for clean,
IDE-friendly access to all hyperparameters.
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# =============================================================================
# 1. GLOBAL CONFIGURATION & SETUP
# =============================================================================

class Config:
    """
    A static class for managing all hyperparameters and settings.
    This approach allows for clean, dot-notation access (e.g., Config.BATCH_SIZE)
    and provides excellent IDE auto-completion and error-checking.
    """
    # --- File Paths ---
    # Assumes a project structure like:
    # /data/
    #   - train.csv
    #   - test.csv
    #   /images/
    #     - train/
    #     - test/
    # /outputs/
    #   - (models and submissions will be saved here)
    TRAIN_CSV = "data/train.csv"
    TEST_CSV = "data/test.csv"
    IMAGE_DIR_TRAIN = "data/images/train/"
    IMAGE_DIR_TEST = "data/images/test/"
    MODEL_PATH = "outputs/price_predictor.pth"
    SUBMISSION_PATH = "outputs/submission.csv"

    # --- Model Architecture ---
    VISION_ENCODER_NAME = "efficientnet_b4"
    TEXT_ENCODER_NAME = "bert-base-uncased"
    VISION_FEATURE_DIM = 1792
    TEXT_FEATURE_DIM = 768
    DROPOUT = 0.35

    # --- Data Processing ---
    IMAGE_SIZE = 384
    MAX_TEXT_LEN = 256
    VAL_SPLIT_RATIO = 0.15

    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    EPOCHS = 20
    WARMUP_EPOCHS = 2
    PATIENCE = 5
    LR_VISION = 1e-5
    LR_TEXT = 2e-5
    LR_HEAD = 1e-4
    WEIGHT_DECAY = 1e-3

    # --- System Settings ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 2024
    NUM_WORKERS = max(0, os.cpu_count() // 2)
    USE_AMP = torch.cuda.is_available() # Automatic Mixed Precision


def apply_seed(seed_value=Config.SEED):
    """Sets a global seed for random number generators to ensure reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =============================================================================
# 2. DATASET & TRANSFORMATION LOGIC
# =============================================================================

def get_image_transforms(is_train: bool):
    """Provides image transformations: augmentation for training, basic for validation/test."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

class ECommerceDataset(Dataset):
    """Dataset class with helpers to process image and text data."""
    def __init__(self, df, tokenizer, image_dir, is_train):
        self.df = df
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.is_train = is_train
        self.transform = get_image_transforms(is_train)

    def __len__(self):
        return len(self.df)

    def _load_and_transform_image(self, sample_id):
        """Loads an image, applies transforms, and handles missing files."""
        prefix = "" # Image files are assumed to be named directly by ID
        path = os.path.join(self.image_dir, f"{prefix}{sample_id}.jpg")
        try:
            image = Image.open(path).convert("RGB")
        except (FileNotFoundError, IOError):
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (220, 220, 220))
        return self.transform(image)

    def _tokenize_text(self, text):
        """Tokenizes text content, handling missing or NaN values."""
        if pd.isna(text) or str(text).strip().lower() == 'nan':
            text = "no description provided"
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=Config.MAX_TEXT_LEN,
            return_tensors='pt'
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokenized_output = self._tokenize_text(row['catalog_content'])
        
        data_packet = {
            'image': self._load_and_transform_image(row['sample_id']),
            'input_ids': tokenized_output['input_ids'].squeeze(),
            'attention_mask': tokenized_output['attention_mask'].squeeze(),
        }

        if 'price' in row:
            data_packet['target'] = torch.tensor(np.log1p(row['price']), dtype=torch.float32)

        return data_packet

# =============================================================================
# 3. MODEL ARCHITECTURE
# =============================================================================

class MultimodalPricePredictor(nn.Module):
    """A model that fuses features from vision and text encoders."""
    def __init__(self):
        super().__init__()
        # Vision Branch
        self.vision_encoder = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.vision_encoder.classifier = nn.Identity()

        # Text Branch
        self.text_encoder = AutoModel.from_pretrained(Config.TEXT_ENCODER_NAME)

        # Fusion Head
        combined_dim = Config.VISION_FEATURE_DIM + Config.TEXT_FEATURE_DIM
        self.regression_head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 1)
        )

    def forward(self, image, input_ids, attention_mask):
        img_features = self.vision_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        combined = torch.cat([img_features, text_features], dim=1)
        return self.regression_head(combined).squeeze(-1)

# =============================================================================
# 4. TRAINING & EVALUATION FUNCTIONS
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    """Executes a single training epoch."""
    model.train()
    total_loss = 0
    # Freeze backbones during warmup
    is_warmup = epoch < Config.WARMUP_EPOCHS
    for param in model.vision_encoder.parameters(): param.requires_grad = not is_warmup
    for param in model.text_encoder.parameters(): param.requires_grad = not is_warmup
    
    progress = tqdm(loader, desc=f"Epoch {epoch+1} Train", leave=False)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        images = batch['image'].to(Config.DEVICE)
        ids = batch['input_ids'].to(Config.DEVICE)
        mask = batch['attention_mask'].to(Config.DEVICE)
        targets = batch['target'].to(Config.DEVICE)
        
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
            preds = model(images, ids, mask)
            loss = criterion(preds, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_model(model, loader, criterion):
    """Calculates loss and SMAPE for a given dataset."""
    model.eval()
    total_loss = 0
    all_preds_log, all_targets_log = [], []
    
    progress = tqdm(loader, desc="Evaluate", leave=False)
    for batch in progress:
        images = batch['image'].to(Config.DEVICE)
        ids = batch['input_ids'].to(Config.DEVICE)
        mask = batch['attention_mask'].to(Config.DEVICE)
        targets = batch['target'].to(Config.DEVICE)
        
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
            preds = model(images, ids, mask)
            loss = criterion(preds, targets)
            
        total_loss += loss.item()
        all_preds_log.append(preds.cpu())
        all_targets_log.append(targets.cpu())

    # Compute SMAPE on original scale
    preds_orig = np.expm1(torch.cat(all_preds_log).numpy())
    targets_orig = np.expm1(torch.cat(all_targets_log).numpy())
    smape = 100 * np.mean(2 * np.abs(preds_orig - targets_orig) / (np.abs(preds_orig) + np.abs(targets_orig) + 1e-8))
    
    return total_loss / len(loader), smape

# =============================================================================
# 5. PIPELINE ORCHESTRATION
# =============================================================================

def build_dataloaders():
    """Loads data, creates datasets, and returns dataloaders."""
    print("Building dataloaders...")
    df = pd.read_csv(Config.TRAIN_CSV)
    train_df, val_df = train_test_split(df, test_size=Config.VAL_SPLIT_RATIO, random_state=Config.SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_ENCODER_NAME)
    
    train_ds = ECommerceDataset(train_df, tokenizer, Config.IMAGE_DIR_TRAIN, is_train=True)
    val_ds = ECommerceDataset(val_df, tokenizer, Config.IMAGE_DIR_TRAIN, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    return train_loader, val_loader

def initialize_components():
    """Initializes model, optimizer, criterion, and scheduler."""
    print("Initializing model components...")
    model = MultimodalPricePredictor().to(Config.DEVICE)
    
    optimizer = torch.optim.AdamW([
        {'params': model.vision_encoder.parameters(), 'lr': Config.LR_VISION},
        {'params': model.text_encoder.parameters(), 'lr': Config.LR_TEXT},
        {'params': model.regression_head.parameters(), 'lr': Config.LR_HEAD},
    ], weight_decay=Config.WEIGHT_DECAY)
    
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)
    
    return model, optimizer, criterion, scheduler, scaler

def run_training_cycle(model, train_loader, val_loader, optimizer, criterion, scheduler, scaler):
    """Executes the main training loop with early stopping."""
    print("Starting training cycle...")
    best_smape = float('inf')
    patience = 0
    
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
        val_loss, val_smape = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SMAPE: {val_smape:.2f}%")
        scheduler.step(val_smape)
        
        if val_smape < best_smape:
            best_smape = val_smape
            patience = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> Model saved. New best SMAPE: {best_smape:.2f}%")
        else:
            patience += 1
            if patience >= Config.PATIENCE:
                print("  -> Early stopping triggered.")
                break
    return best_smape

def generate_submission():
    """Generates predictions on the test set and saves the submission file."""
    print("Generating submission file...")
    test_df = pd.read_csv(Config.TEST_CSV)
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_ENCODER_NAME)
    test_ds = ECommerceDataset(test_df, tokenizer, Config.IMAGE_DIR_TEST, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    model = MultimodalPricePredictor().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch['image'].to(Config.DEVICE)
            ids = batch['input_ids'].to(Config.DEVICE)
            mask = batch['attention_mask'].to(Config.DEVICE)
            log_preds = model(images, ids, mask)
            all_preds.append(torch.expm1(log_preds).cpu())
            
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': torch.cat(all_preds).numpy()
    })
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to '{Config.SUBMISSION_PATH}'")

def main():
    """Main function to orchestrate the entire pipeline."""
    warnings.filterwarnings('ignore')
    apply_seed()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    
    # Step 1: Prepare data
    train_loader, val_loader = build_dataloaders()
    
    # Step 2: Initialize model and training components
    model, optimizer, criterion, scheduler, scaler = initialize_components()
    
    # Step 3: Run the training and validation cycle
    best_score = run_training_cycle(model, train_loader, val_loader, optimizer, criterion, scheduler, scaler)
    print(f"\nTraining finished. Best validation SMAPE: {best_score:.2f}%")
    
    # Step 4: Generate predictions for submission
    generate_submission()

if __name__ == "__main__":
    main()