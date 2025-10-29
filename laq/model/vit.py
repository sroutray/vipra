from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from torchvision.models.vision_transformer import VisionTransformer as TorchVisionTransformer
from torchvision.models.vision_transformer import interpolate_embeddings


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        vit_size: str = "base",
        if_normalize_img: bool = True,
        normalize_img_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_img_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.vit_size = vit_size
        self.model = self._initialize_vit(vit_size, image_size, patch_size)
        self._remove_classifier_head()

        self.if_normalize_img = if_normalize_img
        self.normalize_img_mean = torch.tensor(normalize_img_mean).reshape(1, 3, 1, 1)
        self.normalize_img_std = torch.tensor(normalize_img_std).reshape(1, 3, 1, 1)

    def _initialize_vit(self, vit_size: str, image_size: int, patch_size: int) -> TorchVisionTransformer:
        model_type = f"{vit_size}_{patch_size}"
        if model_type == "base_16":
            model = vit_b_16(weights="IMAGENET1K_V1")
        elif model_type == "base_32":
            model = vit_b_32(weights="IMAGENET1K_V1")
        elif model_type == "large_16":
            model = vit_l_16(weights="IMAGENET1K_V1")
        elif model_type == "large_32":
            model = vit_l_32(weights="IMAGENET1K_V1")
        else:
            raise ValueError("Invalid model type. Choose 'base_16/32', 'large_16/32'.")
        if model.image_size != image_size:
            new_state_dict = interpolate_embeddings(image_size, patch_size, model.state_dict(), reset_heads=True)
            model.encoder.pos_embedding = nn.Parameter(new_state_dict["encoder.pos_embedding"])
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"Interpolated model state dict: {msg}")
        return model

    def _remove_classifier_head(self) -> None:
        if isinstance(self.model, TorchVisionTransformer):
            del self.model.heads
        else:
            raise TypeError("Unexpected model type. Cannot remove classifier head.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward pass through the encoder
        n = x.shape[0]
        device = x.device
        if self.if_normalize_img:
            x = (x - self.normalize_img_mean.to(device)) / self.normalize_img_std.to(device)
        x_tokens = self.model.conv_proj(x)
        x_tokens = x_tokens.reshape(n, self.model.hidden_dim, -1)
        x_tokens = x_tokens.permute(0, 2, 1)

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x_tokens_in = torch.cat([batch_class_token, x_tokens], dim=1)

        x_features = self.model.encoder(x_tokens_in)
        x_features = x_features[:, 1:]  # Remove class token

        return x_tokens, x_features


class DINOv2Encoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        vit_size: str = "reg_base",
        if_normalize_img: bool = True,
        normalize_img_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_img_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.vit_size = vit_size
        self.use_registers = "reg" in vit_size
        self.model = self._initialize_vit(vit_size, patch_size)  # model is image_size agnostic

        self.if_normalize_img = if_normalize_img
        self.normalize_img_mean = torch.tensor(normalize_img_mean).reshape(1, 3, 1, 1)
        self.normalize_img_std = torch.tensor(normalize_img_std).reshape(1, 3, 1, 1)

    def _initialize_vit(self, vit_size: str, patch_size: int) -> nn.Module:
        model_type = f"{vit_size}_{patch_size}"
        if model_type == "reg_base_14":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        elif model_type == "base_14":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        else:
            raise ValueError("Invalid model type. Choose 'base_14', 'reg_base_14'.")
        return model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        if self.if_normalize_img:
            x = (x - self.normalize_img_mean.to(device)) / self.normalize_img_std.to(device)
        x_features = self.model.forward_features(x)["x_norm_patchtokens"]
        num_patches = x_features.shape[1]
        x_tokens = self.model.prepare_tokens_with_masks(x)[:, -num_patches:]
        return x_tokens, x_features


# Example usage
if __name__ == "__main__":
    # Instantiate ViT encoders of different sizes
    model = DINOv2Encoder(224, 14).cuda()

    # Create a sample input tensor
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    # Forward pass through each encoder
    tokens, features = model(input_tensor)

    # Print output shapes
    print(f"Base ViT encoder output shape: {features.shape}")
