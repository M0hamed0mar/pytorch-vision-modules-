"""
Vision Transformer (ViT) implementation.
Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""
import torch
import torch.nn as nn
from typing import Optional


class PatchEmbedding(nn.Module):
    """
    Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        patch_size: Size of each patch (default: 16)
        embedding_dim: Dimension of embedding vector (default: 768)
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768):
        super().__init__()
        
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                out_channels=embedding_dim,
                                kernel_size=patch_size,
                                stride=patch_size,
                                padding=0)
        
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        if hasattr(self, 'patch_size'):
            patch_size = self.patch_size
        else:
            patch_size = self.patcher.kernel_size[0]
            
        assert image_resolution % patch_size == 0, f"Input image size must be divisible by patch size"
        
        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        
        # Adjust so the embedding is on the final dimension [batch_size, num_patches, embedding_dim]
        return x_flattened.permute(0, 2, 1)


class MultiheadSelfAttentionBlock(nn.Module):
    """
    Creates a multi-head self-attention block ("MSA block" for short).
    """
    def __init__(self, embedding_dim: int = 768, num_heads: int = 12, attn_dropout: float = 0):
        super().__init__()
        
        # Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        # Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                   num_heads=num_heads,
                                                   dropout=attn_dropout,
                                                   batch_first=True)
        
    def forward(self, x):
        x = self.layer_norm(x)
        
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        return attn_output


class MLPBlock(nn.Module):
    """
    Creates a layer normalized multilayer perceptron block ("MLP block" for short).
    """
    def __init__(self, embedding_dim: int = 768, mlp_size: int = 3072, dropout: float = 0.1):
        super().__init__()
        
        # Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        # Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Creates a Transformer Encoder block.
    """
    def __init__(self, embedding_dim: int = 768, num_heads: int = 12, mlp_size: int = 3072,
                 mlp_dropout: float = 0.1, attn_dropout: float = 0):
        super().__init__()
        
        # Create MSA block
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    attn_dropout=attn_dropout)
        
        # Create MLP block
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                 mlp_size=mlp_size,
                                 dropout=mlp_dropout)
        
    def forward(self, x):
        # Create residual connection for MSA block
        x = self.msa_block(x) + x
        
        # Create residual connection for MLP block
        x = self.mlp_block(x) + x
        
        return x


class ViT(nn.Module):
    """
    Creates a Vision Transformer architecture.
    """
    def __init__(self, img_size: int = 224, in_channels: int = 3, patch_size: int = 16,
                 num_transformer_layers: int = 12, embedding_dim: int = 768, mlp_size: int = 3072,
                 num_heads: int = 12, attn_dropout: float = 0, mlp_dropout: float = 0.1,
                 embedding_dropout: float = 0.1, num_classes: int = 1000):
        super().__init__()
        
        # Make sure the image size is divisible by the patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size"
        
        # Calculate number of patches
        self.num_patches = (img_size * img_size) // patch_size**2
        
        # Create learnable class embedding
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)
        
        # Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad=True)
        
        # Create embedding dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                             patch_size=patch_size,
                                             embedding_dim=embedding_dim)
        
        # Create Transformer Encoder blocks
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoderBlock(embedding_dim=embedding_dim,
                                   num_heads=num_heads,
                                   mlp_size=mlp_size,
                                   mlp_dropout=mlp_dropout,
                                   attn_dropout=attn_dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )
        
    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]
        
        # Create class token embedding
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        
        # Create patch embedding
        x = self.patch_embedding(x)
        
        # Concat class embedding and patch embedding
        x = torch.cat((class_token, x), dim=1)
        
        # Add position embedding
        x = self.position_embedding + x
        
        # Apply embedding dropout
        x = self.embedding_dropout(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Put 0 index logit through classifier
        x = self.classifier(x[:, 0])
        
        return x