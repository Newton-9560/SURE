import torch
import torch.nn as nn


class MAMP_MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dropout_p: float = 0.3,
        task: str = "binary"
    ):
        super().__init__()

        self.task = task

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Layer 2
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Layer 3 (gentle reduction)
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Output layer
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)

        if self.task == "binary":
            out = torch.sigmoid(out)

        return out