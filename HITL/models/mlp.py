import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGatingUnit(nn.Module):
    """
    Applies a spatial gating mechanism to an input tensor.

    This unit splits the input tensor along the channel dimension and uses one half
    to gate the other half after a linear projection. This allows for interactions

    between different 'spatial' locations (here, we treat feature dimensions as spatial).
    """
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn // 2)
        # Project along the sequence/feature dimension
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        # x has shape [batch_size, seq_len, d_ffn]
        # Split into two halves along the last dimension (d_ffn)
        u, v = x.chunk(2, dim=-1)

        # Apply gating: v is normalized and projected to learn spatial interactions
        v = self.norm(v)
        # Transpose to [batch_size, d_ffn/2, seq_len] for projection
        v = v.transpose(1, 2)
        v = self.proj(v)
        # Transpose back to [batch_size, seq_len, d_ffn/2]
        v = v.transpose(1, 2)

        # Element-wise multiplication to gate u
        return u * v


class gMLPBlock(nn.Module):
    """
    A single gMLP block which includes normalization, channel projections,
    and a spatial gating unit.
    """
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # Project up to the feed-forward dimension
        self.channel_proj1 = nn.Linear(d_model, d_ffn)
        # The core spatial gating unit
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        # Project back down to the model dimension
        self.channel_proj2 = nn.Linear(d_ffn // 2, d_model)

    def forward(self, x):
        # x has shape [batch_size, seq_len, d_model]
        residual = x
        x = self.norm(x)
        
        # Apply GELU activation after the first projection
        x = F.gelu(self.channel_proj1(x))
        
        # Apply the spatial gating
        x = self.sgu(x)
        
        # Apply the final projection
        x = self.channel_proj2(x)
        
        # Add the residual connection
        return x + residual


class LatentClassifier(nn.Module):
    """
    Classifier using a gMLP block for a single latent vector input.
    """
    def __init__(self, input_dim=384, d_ffn_factor=2, depth=2, bayesian=False, dropout_rate=0.2):
        super().__init__()
        # We treat the input dimension as the sequence length
        self.seq_len = input_dim
        # We treat the model dimension as 1 for simplicity, as we have only one vector
        self.d_model = 1
        d_ffn = self.d_model * d_ffn_factor
        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        # Stack multiple gMLP blocks if needed
        self.gmlp_layers = nn.Sequential(
            *[gMLPBlock(self.d_model, d_ffn, self.seq_len) for _ in range(depth)]
        )

        # Classification head
        self.pooler = nn.Linear(input_dim, input_dim)
        
        if self.bayesian:
            self.dropout1 = nn.Dropout(dropout_rate)  # After gMLP
            self.dropout2 = nn.Dropout(dropout_rate)  # After pooler
            
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Input x has shape [batch_size, input_dim]
        
        # 1. Reshape for gMLP: [batch_size, input_dim] -> [batch_size, input_dim, 1]
        # This treats the feature dimension (input_dim) as the "sequence".
        x = x.unsqueeze(-1)

        # 2. Pass through the gMLP block(s)
        gmlp_output = self.gmlp_layers(x)

        # 3. Reshape back to original format: [batch_size, input_dim, 1] -> [batch_size, input_dim]
        x = gmlp_output.squeeze(-1)
        
        # 4. Apply dropout after gMLP
        if self.bayesian:
            x = self.dropout1(x)

        # 5. Final classification head
        x = F.gelu(self.pooler(x))
        
        # 6. Apply dropout after pooler
        if self.bayesian:
            x = self.dropout2(x)
            
        logits = self.classifier(x)

        # 7. Return logits (sigmoid will be applied in loss function or when probabilities are needed)
        return logits.squeeze(-1)

