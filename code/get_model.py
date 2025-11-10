import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoTokenizer

# A registry to hold model-specific information
VISION_REGISTRY = {
    "efficientnet_b4": {"model": models.efficientnet_b4, "weights": models.EfficientNet_B4_Weights.DEFAULT, "dim": 1792, "layer": "classifier"},
    "resnet50": {"model": models.resnet50, "weights": models.ResNet50_Weights.DEFAULT, "dim": 2048, "layer": "fc"},
}

def get_vision_encoder(model_name: str):
    """
    Dynamically loads a pretrained vision encoder from torchvision.

    Args:
        model_name (str): The name of the model (e.g., 'efficientnet_b4').

    Returns:
        A tuple containing the model and its output feature dimension.
    """
    if model_name not in VISION_REGISTRY:
        raise ValueError(f"Vision model '{model_name}' not supported. Available: {list(VISION_REGISTRY.keys())}")
    
    config = VISION_REGISTRY[model_name]
    model = config["model"](weights=config["weights"])
    
    # Replace the final classification layer with an identity layer
    setattr(model, config["layer"], nn.Identity())
    
    return model, config["dim"]

def get_text_encoder(model_name: str):
    """
    Dynamically loads a pretrained text encoder and its tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the model from Hugging Face Hub.

    Returns:
        A tuple containing the model, tokenizer, and its output feature dimension.
    """
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    feature_dim = model.config.hidden_size
    
    return model, tokenizer, feature_dim
