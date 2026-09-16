import torch
import torch.nn as nn


class MaskedCrossViewContrastiveLoss(nn.Module):
    """L_con: masked, adjacency-aware cross-view contrastive loss."""
    def __init__(self, temperature):
        super(MaskedCrossViewContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j, mask_i, mask_j, adj):
        if torch.sum(mask_i) == 0 or torch.sum(mask_j) == 0:
            return 0
        adj = 1 - adj
        adj = adj.fill_diagonal_(1)
        adj = torch.cat((adj, adj), dim=1)
        adj = torch.cat((adj, adj), dim=0)
        mask_ij = torch.cat((mask_i, mask_j))
        mask = mask_ij.view(-1, 1) * mask_ij
        n = z_i.shape[0]
        N = 2 * n
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        sim = sim * adj
        sim[mask == 0] = -1e5  # except missing instances
        sim = sim.fill_diagonal_(-1e5)  # self

        label = torch.cat((torch.arange(n, N), torch.arange(0, n))).to(z_i.device)
        loss = self.criterion(sim, label)

        # Subtract the excess constant
        n1 = 2 * torch.sum(1 - mask_i * mask_j)  # the num when mask[i,v]*mask[i,w]=0
        n2 = torch.sum(1 - mask_ij)  # v-th instance missing i.e., mask[i,v]=0.
        loss += n2 * torch.log(torch.tensor(1 / N))  #
        loss -= (n1 - n2) * 1e5  #
        loss /= N

        return loss


class MaskedBCELoss(nn.Module):
    """Observed-label BCE used to build the TVI-Net classification loss L_bce."""
    def __init__(self, ):
        super(MaskedBCELoss, self).__init__()

    def forward(self, y_true, y_pred, mask_label):
        if torch.sum(mask_label) == 0:
            return torch.tensor(0.)

        eps = 1e-5
        loss = -(mask_label * (y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps)))
        loss = torch.sum(loss) / torch.sum(mask_label)

        return loss


class ViewReconstructionLoss(nn.Module):
    """L_rec: view reconstruction with additional artificial-mask supervision."""
    def __init__(self, reduction='mean'):
        super(ViewReconstructionLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, x_reconstructed, x_original, mask, artificial_mask=None, lambda_masked=0.5):
        """
        Calculate reconstruction loss for incomplete multi-view data with optional masked-view reconstruction.
        
        Args:
            x_reconstructed: List of reconstructed views
            x_original: List of original views
            mask: Original mask matrix indicating missing views (batch_size, view_num)
            artificial_mask: Optional artificial mask for masked-view reconstruction training
            lambda_masked: Weight for masked reconstruction loss
            
        Returns:
            total_loss: Combined loss (loss_obs + lambda_masked * loss_masked)
            loss_dict: Dictionary containing individual loss components
        """
        total_loss = 0.0
        view_num = len(x_reconstructed)
        
        loss_obs = 0.0
        loss_masked = 0.0
        
        for v in range(view_num):
            if torch.sum(mask[:, v]) > 0:
                valid_indices = mask[:, v].bool()
                loss_view = self.criterion(x_reconstructed[v][valid_indices], x_original[v][valid_indices])
                loss_obs += loss_view
            
            if artificial_mask is not None:
                newly_masked = (mask[:, v] == 1) & (artificial_mask[:, v] == 0)
                if newly_masked.sum() > 0:
                    loss_masked_view = self.criterion(
                        x_reconstructed[v][newly_masked],
                        x_original[v][newly_masked]
                    )
                    loss_masked += loss_masked_view
        
        loss_obs = loss_obs / view_num
        loss_masked = loss_masked / view_num
        
        total_loss = loss_obs + lambda_masked * loss_masked
        
        loss_dict = {
            'total': total_loss,
            'loss_obs': loss_obs,
            'loss_masked': loss_masked
        }
        
        return total_loss, loss_dict
