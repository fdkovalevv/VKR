from __future__ import annotations

import torch
from torch import nn


class ProjectionBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class EarlyFusionClassifier(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        text_dim: int,
        image_dim: int,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tabular_projector = ProjectionBlock(tabular_dim, projection_dim, dropout)
        self.text_projector = ProjectionBlock(text_dim, projection_dim, dropout)
        self.image_projector = ProjectionBlock(image_dim, projection_dim, dropout)

        fused_dim = projection_dim * 3 + 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tabular_features: torch.Tensor,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        modality_flags: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat(
            [
                self.tabular_projector(tabular_features),
                self.text_projector(text_features),
                self.image_projector(image_features),
                modality_flags,
            ],
            dim=1,
        )
        return self.classifier(fused).squeeze(1)
