import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """
        inputs: predicted probabilities (after sigmoid), shape [batch_size]
        targets: true labels (0 or 1), shape [batch_size]
        """
        # Clamp to prevent log(0)
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        
        # Calculate focal loss
        bce_loss = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)
        
        # Modulating factor
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification tasks.
        
        Args:
            alpha (float, optional): Weighting factor for the rare class. 
                                     Set to None if using balanced mini-batches.
            gamma (float): Focusing parameter to down-weight easy examples.
            reduction (str): 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: raw outputs from your ensemble networks (before sigmoid)
        # targets: binary ground truth (0 for non-SL, 1 for SL), same shape as logits
        
        # 1. Compute standard BCE loss with logits for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. Derive pt (the probability of the true class)
        # Since BCE is -log(pt), we can get pt via exp(-BCE)
        pt = torch.exp(-bce_loss)
        
        # 3. Compute the focal factor: (1 - pt)^gamma
        focal_factor = (1 - pt) ** self.gamma
        
        # 4. Apply alpha weighting if an alpha value was provided
        if self.alpha is not None:
            alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_factor = alpha_weight * focal_factor
            
        # 5. Calculate the final focal loss
        focal_loss = focal_factor * bce_loss
        
        # 6. Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss