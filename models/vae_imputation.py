import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEViewImputation(nn.Module):
    """
    Framework (a): VAE-based View Imputation.
    Mask-aware gated fusion, variational sampling and view-specific decoding.
    """
    def __init__(self, view_dims, hidden_dim=512, latent_dim=256, dropout=0.2):
        super(VAEViewImputation, self).__init__()
        self.view_num = len(view_dims)
        
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim)
            )
            for dim in view_dims
        ])
        
        self.gate_network = nn.Sequential(
            nn.Linear(latent_dim * self.view_num, self.view_num),
            nn.Sigmoid()
        )
        
        self.to_mu = nn.Linear(latent_dim, latent_dim)
        self.to_logvar = nn.Linear(latent_dim, latent_dim)
        
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim)
            )
            for dim in view_dims
        ])

        # Kept to preserve the original model initialization RNG sequence.
        self.imputation_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, view_dims[i])
            )
            for i in range(self.view_num)
        ])
        
    def encode(self, x_list, mask_data):
        """Encode available views and fuse them"""
        # Encode each view
        encoded = []
        for i in range(self.view_num):
            z_i = self.encoders[i](x_list[i])
            encoded.append(z_i)
        
        # Stack encodings
        encoded_stack = torch.stack(encoded, dim=1)  # [batch, view_num, latent_dim]
        
        # Compute gates based on all encodings
        concat_encoded = torch.cat(encoded, dim=1)
        gates = self.gate_network(concat_encoded)  # [batch, view_num]
        
        # Apply mask and gates
        gates = gates * mask_data
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted fusion
        fused = (encoded_stack * gates.unsqueeze(-1)).sum(dim=1)
        
        return fused, gates
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def apply_artificial_mask(self, x_list, mask_data, mask_ratio=0.3):
        """
        Apply artificial mask to observed views for masked-view reconstruction training.
        For each observed view, randomly mask a portion to simulate missing data.
        """
        artificial_mask = mask_data.clone().float()
        
        for i in range(self.view_num):
            obs_indices = (mask_data[:, i] == 1).nonzero(as_tuple=True)[0]
            if len(obs_indices) > 0:
                num_to_mask = int(len(obs_indices) * mask_ratio)
                if num_to_mask > 0:
                    perm = torch.randperm(len(obs_indices))[:num_to_mask]
                    mask_indices = obs_indices[perm]
                    artificial_mask[mask_indices, i] = 0
        
        artificial_mask = artificial_mask.bool()
        
        return x_list, artificial_mask
    
    def forward(self, x_list, mask_data, apply_artificial_mask=False, artificial_mask_ratio=0.3):
        """
        Forward pass with imputation
        """
        if apply_artificial_mask:
            x_list, artificial_mask = self.apply_artificial_mask(x_list, mask_data, artificial_mask_ratio)
            mask_for_vae = artificial_mask.float()
            original_mask = mask_data.clone()
        else:
            mask_for_vae = mask_data
            original_mask = mask_data
            artificial_mask = None
        
        fused, gates = self.encode(x_list, mask_for_vae)
        
        mu = self.to_mu(fused)
        logvar = self.to_logvar(fused)
        z = self.reparameterize(mu, logvar)
        
        reconstructed = []
        for i in range(self.view_num):
            recon = self.decoders[i](z)
            reconstructed.append(recon)
        
        imputed = []
        for i in range(self.view_num):
            view_mask = mask_for_vae[:, i].unsqueeze(1).float()
            imputed_view = x_list[i] * view_mask + reconstructed[i] * (1 - view_mask)
            imputed.append(imputed_view)
        
        outputs = {
            'imputed': imputed,
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'gates': gates,
            'artificial_mask': artificial_mask,
            'original_mask': original_mask
        }
        
        return imputed, outputs
    
    def compute_loss(self, x_list, mask_data, outputs, recon_weight=1.0, kl_weight=0.1, lambda_masked=0.5):
        """
        Compute VAE loss with both observed reconstruction and masked-view reconstruction.
        
        loss = loss_obs + lambda_masked * loss_masked + KL
        
        loss_obs: reconstruction on originally observed positions
        loss_masked: reconstruction on artificially masked positions (for imputation training)
        """
        reconstructed = outputs['reconstructed']
        mu = outputs['mu']
        logvar = outputs['logvar']
        artificial_mask = outputs.get('artificial_mask', None)
        original_mask = outputs.get('original_mask', mask_data)
        
        loss_obs = 0
        loss_masked = 0
        
        for i in range(self.view_num):
            view_original_mask = original_mask[:, i].unsqueeze(1).float()
            
            recon_orig = F.mse_loss(
                reconstructed[i] * view_original_mask,
                x_list[i] * view_original_mask,
                reduction='sum'
            )
            loss_obs += recon_orig / (view_original_mask.sum() + 1e-8)
            
            if artificial_mask is not None:
                newly_masked = (original_mask[:, i] == 1) & (artificial_mask[:, i] == 0)
                if newly_masked.sum() > 0:
                    new_mask = newly_masked.unsqueeze(1).float()
                    recon_masked = F.mse_loss(
                        reconstructed[i] * new_mask,
                        x_list[i] * new_mask,
                        reduction='sum'
                    )
                    loss_masked += recon_masked / (new_mask.sum() + 1e-8)
        
        loss_obs = loss_obs / self.view_num
        loss_masked = loss_masked / self.view_num
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        
        total_loss = recon_weight * (loss_obs + lambda_masked * loss_masked) + kl_weight * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss_obs': loss_obs,
            'recon_loss_masked': loss_masked,
            'kl_loss': kl_loss
        }
