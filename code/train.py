# -*- coding: utf-8 -*-
"""
E-Commerce Price Prediction - An Enhanced Modular Pipeline.

This script implements a flexible, configuration-driven, and trackable
multimodal price prediction model.
"""
import os
import random
import warnings
import yaml
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from get_model import get_vision_encoder, get_text_encoder

# =============================================================================
# 1. SETUP & HELPER FUNCTIONS
# =============================================================================

def load_config(config_path="config.yaml"):
    """Loads YAML config and converts to a SimpleNamespace for dot notation access."""
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return SimpleNamespace(**config_dict)

def apply_seed(seed_value):
    """Sets a global seed for random number generators to ensure reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(device_config: str):
    """Gets the appropriate torch device based on config."""
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config

# =============================================================================
# 2. DATASET & TRANSFORMATION LOGIC
# =============================================================================
# (Dataset and Transforms remain largely the same, but now accept the config object)

def get_image_transforms(config, is_train: bool):
    """Provides image transformations: augmentation for training, basic for validation/test."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
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
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

class ECommerceDataset(Dataset):
    def __init__(self, df, tokenizer, image_dir, config, is_train):
        self.df = df
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.config = config
        self.transform = get_image_transforms(config, is_train)

    def __len__(self):
        return len(self.df)

    def _load_and_transform_image(self, sample_id):
        path = os.path.join(self.image_dir, f"{sample_id}.jpg")
        try:
            image = Image.open(path).convert("RGB")
        except (FileNotFoundError, IOError):
            image = Image.new("RGB", (self.config.data.image_size, self.config.data.image_size), (220, 220, 220))
        return self.transform(image)

    def _tokenize_text(self, text):
        if pd.isna(text) or str(text).strip().lower() == 'nan':
            text = "no description provided"
        return self.tokenizer(
            text, padding='max_length', truncation=True, max_length=self.config.data.max_text_len, return_tensors='pt'
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
# 3. MODEL ARCHITECTURE (NOW DYNAMIC)
# =============================================================================

class MultimodalPricePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_encoder, vision_dim = get_vision_encoder(config.model.vision_encoder_name)
        self.text_encoder, _, text_dim = get_text_encoder(config.model.text_encoder_name)

        combined_dim = vision_dim + text_dim
        self.regression_head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
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
# (These functions now accept the config object and device)

def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch, config, device):
    model.train()
    total_loss = 0
    is_warmup = epoch < config.training.warmup_epochs
    for param in model.vision_encoder.parameters(): param.requires_grad = not is_warmup
    for param in model.text_encoder.parameters(): param.requires_grad = not is_warmup
    
    progress = tqdm(loader, desc=f"Epoch {epoch+1} Train", leave=False)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        images = batch['image'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        with torch.cuda.amp.autocast(enabled=config.system.use_amp):
            preds = model(images, ids, mask)
            loss = criterion(preds, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_model(model, loader, criterion, config, device):
    model.eval()
    total_loss = 0
    all_preds_log, all_targets_log = [], []
    
    progress = tqdm(loader, desc="Evaluate", leave=False)
    for batch in progress:
        images = batch['image'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        with torch.cuda.amp.autocast(enabled=config.system.use_amp):
            preds = model(images, ids, mask)
            loss = criterion(preds, targets)
            
        total_loss += loss.item()
        all_preds_log.append(preds.cpu())
        all_targets_log.append(targets.cpu())

    preds_orig = np.expm1(torch.cat(all_preds_log).numpy())
    targets_orig = np.expm1(torch.cat(all_targets_log).numpy())
    smape = 100 * np.mean(2 * np.abs(preds_orig - targets_orig) / (np.abs(preds_orig) + np.abs(targets_orig) + 1e-8))
    
    return total_loss / len(loader), smape

# =============================================================================
# 5. PIPELINE ORCHESTRATION
# =============================================================================

def build_dataloaders(config):
    print("Building dataloaders...")
    df = pd.read_csv(config.paths.train_csv)
    train_df, val_df = train_test_split(df, test_size=config.data.val_split_ratio, random_state=config.system.seed)
    
    _, tokenizer, _ = get_text_encoder(config.model.text_encoder_name)
    
    train_ds = ECommerceDataset(train_df, tokenizer, config.paths.image_dir_train, config, is_train=True)
    val_ds = ECommerceDataset(val_df, tokenizer, config.paths.image_dir_train, config, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=config.system.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.system.num_workers)
    
    return train_loader, val_loader

def initialize_components(config, device):
    print("Initializing model components...")
    model = MultimodalPricePredictor(config).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.vision_encoder.parameters(), 'lr': config.training.lr_vision},
        {'params': model.text_encoder.parameters(), 'lr': config.training.lr_text},
        {'params': model.regression_head.parameters(), 'lr': config.training.lr_head},
    ], weight_decay=config.training.weight_decay)
    
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=config.system.use_amp)
    
    return model, optimizer, criterion, scheduler, scaler

def run_training_cycle(config, model, train_loader, val_loader, optimizer, criterion, scheduler, scaler, device):
    print("Starting training cycle...")
    best_smape = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch, config, device)
        val_loss, val_smape = evaluate_model(model, val_loader, criterion, config, device)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SMAPE: {val_smape:.2f}%")
        
        # W&B Logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_smape": val_smape,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        scheduler.step(val_smape)
        
        if val_smape < best_smape:
            best_smape = val_smape
            patience_counter = 0
            torch.save(model.state_dict(), config.paths.model_path)
            print(f"  -> Model saved. New best SMAPE: {best_smape:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                print("  -> Early stopping triggered.")
                break
    return best_smape

def generate_submission(config, device):
    print("Generating submission file...")
    test_df = pd.read_csv(config.paths.test_csv)
    _, tokenizer, _ = get_text_encoder(config.model.text_encoder_name)
    test_ds = ECommerceDataset(test_df, tokenizer, config.paths.image_dir_test, config, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=config.system.num_workers)
    
    model = MultimodalPricePredictor(config).to(device)
    model.load_state_dict(torch.load(config.paths.model_path))
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch['image'].to(device)
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            log_preds = model(images, ids, mask)
            all_preds.append(torch.expm1(log_preds).cpu())
            
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': torch.cat(all_preds).numpy()
    })
    
    os.makedirs(os.path.dirname(config.paths.submission_path), exist_ok=True)
    submission_df.to_csv(config.paths.submission_path, index=False)
    print(f"Submission saved to '{config.paths.submission_path}'")

def main():
    warnings.filterwarnings('ignore')
    
    # Step 1: Load config and initialize environment
    config = load_config()
    apply_seed(config.system.seed)
    device = get_device(config.system.device)
    os.makedirs(os.path.dirname(config.paths.model_path), exist_ok=True)
    
    # Step 2: Initialize Weights & Biases
    wandb.init(project=config.project_name, name=config.run_name, config=vars(config))

    # Step 3: Prepare data
    train_loader, val_loader = build_dataloaders(config)
    
    # Step 4: Initialize model and training components
    model, optimizer, criterion, scheduler, scaler = initialize_components(config, device)
    wandb.watch(model, criterion, log="all", log_freq=100) # Log gradients

    # Step 5: Run the training and validation cycle
    best_score = run_training_cycle(config, model, train_loader, val_loader, optimizer, criterion, scheduler, scaler, device)
    print(f"\nTraining finished. Best validation SMAPE: {best_score:.2f}%")
    
    # Step 6: Generate predictions and save model artifact to W&B
    generate_submission(config, device)
    artifact = wandb.Artifact(name=config.run_name, type="model")
    artifact.add_file(config.paths.model_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    main()
