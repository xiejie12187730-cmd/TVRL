import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class DualWeightedReliabilityEstimator(nn.Module):
    """Framework (b): multiplicative view-level and sample-level reliability."""
    def __init__(self, hidden_dim, view_num, trust_hidden_dim=128):
        super(DualWeightedReliabilityEstimator, self).__init__()
        self.hidden_dim = hidden_dim
        self.view_num = view_num
        self.trust_hidden_dim = trust_hidden_dim
        
        self.view_trust_mlp = nn.Sequential(
            nn.Linear(hidden_dim * view_num, trust_hidden_dim),
            nn.BatchNorm1d(trust_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(trust_hidden_dim, trust_hidden_dim),
            nn.BatchNorm1d(trust_hidden_dim),
            nn.ReLU(),
            nn.Linear(trust_hidden_dim, view_num),
        )
        
        self.sample_trust_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * (view_num + 1), trust_hidden_dim // 2),
                nn.BatchNorm1d(trust_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(trust_hidden_dim // 2, 1),
            ) for _ in range(view_num)
        ])
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, view_embeddings, mask_data):
        view_embeddings_cat = torch.cat(view_embeddings, dim=1)
        view_trust_scores = self.view_trust_mlp(view_embeddings_cat)
        
        view_trust_scores = torch.sigmoid(view_trust_scores)
        
        sample_trust_scores = []
        for v in range(self.view_num):
            combined = torch.cat([
                view_embeddings[v],
                view_embeddings_cat
            ], dim=1)
            
            trust_score = self.sample_trust_mlp[v](combined)
            trust_score = torch.sigmoid(trust_score)
            sample_trust_scores.append(trust_score)
        
        sample_trust_scores = torch.cat(sample_trust_scores, dim=1)
        
        trust_scores = view_trust_scores * sample_trust_scores
        
        if mask_data is not None:
            missing_mask = (mask_data == 0).float()
            trust_scores = trust_scores * (1 - missing_mask) + missing_mask * 0.1
        
        return trust_scores, view_trust_scores


class AdaptiveViewFusion(nn.Module):
    def __init__(self, hidden_dim, view_num, fusion_type='attention'):
        super(AdaptiveViewFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.view_num = view_num
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            self.attention_weights = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                ) for _ in range(view_num)
            ])
        
        elif fusion_type == 'gated':
            self.gate_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                ) for _ in range(view_num)
            ])
            
            self.fusion_network = nn.Sequential(
                nn.Linear(hidden_dim * view_num, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        
        elif fusion_type == 'trust':
            self.trust_projection = nn.Sequential(
                nn.Linear(hidden_dim * view_num, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            
            self.trust_weights = nn.Linear(hidden_dim, view_num)
    
    def forward(self, view_embeddings, trust_scores=None):
        if self.fusion_type == 'attention':
            attention_scores = []
            for v in range(self.view_num):
                score = self.attention_weights[v](view_embeddings[v])
                attention_scores.append(score)
            
            attention_scores = torch.cat(attention_scores, dim=1)
            attention_weights = F.softmax(attention_scores, dim=1)
            
            if trust_scores is not None:
                attention_weights = attention_weights * trust_scores
                attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            fused_embedding = torch.zeros_like(view_embeddings[0])
            for v in range(self.view_num):
                fused_embedding += attention_weights[:, v:v+1] * view_embeddings[v]
            
            return fused_embedding, attention_weights
        
        elif self.fusion_type == 'gated':
            gate_values = []
            for v in range(self.view_num):
                combined = torch.cat([view_embeddings[v], torch.mean(view_embeddings[v], dim=1, keepdim=True).expand(-1, self.hidden_dim)], dim=1)
                gate = self.gate_networks[v](combined)
                gate_values.append(gate)
            
            gate_values = torch.cat(gate_values, dim=1)
            
            if trust_scores is not None:
                gate_values = gate_values * trust_scores
            
            gate_weights = F.softmax(gate_values, dim=1)
            
            gated_embeddings = []
            for v in range(self.view_num):
                gated_embeddings.append(gate_weights[:, v:v+1] * view_embeddings[v])
            
            concatenated = torch.cat(gated_embeddings, dim=1)
            fused_embedding = self.fusion_network(concatenated)
            
            return fused_embedding, gate_weights
        
        elif self.fusion_type == 'trust':
            concatenated = torch.cat(view_embeddings, dim=1)
            projected = self.trust_projection(concatenated)
            
            trust_weights = F.softmax(self.trust_weights(projected), dim=1)
            
            if trust_scores is not None:
                trust_weights = trust_weights * trust_scores
                trust_weights = trust_weights / (trust_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            fused_embedding = torch.zeros_like(view_embeddings[0])
            for v in range(self.view_num):
                fused_embedding += trust_weights[:, v:v+1] * view_embeddings[v]
            
            return fused_embedding, trust_weights
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")


class LabelWiseUncertaintyQuantifier(nn.Module):
    """Framework (c): label-wise mean logits and positive predictive variance."""
    def __init__(self, hidden_dim, class_num, uncertainty_dim=64):
        super(LabelWiseUncertaintyQuantifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.uncertainty_dim = uncertainty_dim
        
        self.uncertainty_network = nn.Sequential(
            nn.Linear(hidden_dim, uncertainty_dim),
            nn.BatchNorm1d(uncertainty_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(uncertainty_dim, uncertainty_dim),
            nn.BatchNorm1d(uncertainty_dim),
            nn.ReLU(),
            nn.Linear(uncertainty_dim, class_num * 2)
        )
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, fused_embedding, view_embeddings=None, trust_scores=None):
        uncertainty_output = self.uncertainty_network(fused_embedding)
        
        mean_logits = uncertainty_output[:, :self.class_num]
        log_variance = uncertainty_output[:, self.class_num:]
        
        variance = F.softplus(log_variance) + 1e-6
        
        epistemic_uncertainty = variance
        
        if view_embeddings is not None and trust_scores is not None:
            aleatoric_components = []
            for v in range(len(view_embeddings)):
                trust_weight = trust_scores[:, v:v+1]
                aleatoric_component = variance * (1 - trust_weight)
                aleatoric_components.append(aleatoric_component)
            
            aleatoric_uncertainty = torch.mean(torch.stack(aleatoric_components), dim=0)
        else:
            aleatoric_uncertainty = variance
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return mean_logits, variance, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty


class TrustworthyMultiLabelClassification(nn.Module):
    """Framework (c): auxiliary trust-guided fusion and uncertainty quantification.

    Includes the dual-weighted reliability estimator from framework (b).
    The primary classifier and L_bce are evaluated by TVRL."""
    def __init__(self, args, logger=None):
        super(TrustworthyMultiLabelClassification, self).__init__()
        self.hidden_dim = args.hidden
        self.view_num = args.view_num
        self.class_num = args.class_num
        
        self.trust_evaluator = DualWeightedReliabilityEstimator(
            hidden_dim=self.hidden_dim,
            view_num=self.view_num,
            trust_hidden_dim=self.hidden_dim // 2
        )
        
        self.uncertainty_quantifier = LabelWiseUncertaintyQuantifier(
            hidden_dim=self.hidden_dim,
            class_num=self.class_num,
            uncertainty_dim=self.hidden_dim // 2
        )
        
        self.trust_temperature = getattr(args, 'trust_temperature', 0.1)
        self.uncertainty_weight = getattr(args, 'uncertainty_weight', 0.1)
        self.trust_consistency_weight = getattr(args, 'trust_consistency_weight', 0.05)

    def trust_weighted_sum(self, view_embeddings, trust_weights):
        z_t = torch.zeros_like(view_embeddings[0])
        for v in range(self.view_num):
            z_t += trust_weights[:, v:v + 1] * view_embeddings[v]
        return z_t
    
    def forward(self, view_embeddings, mask_data=None, return_details=False):
        trust_scores, view_trust_scores = self.trust_evaluator(view_embeddings, mask_data)
        
        trust_weights = F.softmax(trust_scores / self.trust_temperature, dim=1)
        
        trusted_embedding = self.trust_weighted_sum(view_embeddings, trust_weights)
        
        mean_logits, variance, epistemic, aleatoric, total_uncertainty = self.uncertainty_quantifier(
            trusted_embedding, view_embeddings, trust_scores
        )
        trust_prediction = torch.sigmoid(mean_logits)
        
        if return_details:
            return {
                'mean_logits': mean_logits,
                'trust_prediction': trust_prediction,
                'variance': variance,
                'epistemic_uncertainty': epistemic,
                'aleatoric_uncertainty': aleatoric,
                'total_uncertainty': total_uncertainty,
                'trust_scores': trust_scores,
                'view_trust_scores': view_trust_scores,
                'trust_weights': trust_weights,
                'fusion_weights': trust_weights,
                'trusted_embedding': trusted_embedding,
                'fused_embedding': trusted_embedding
            }
        
        return mean_logits, variance, epistemic, aleatoric, total_uncertainty, trust_scores, trusted_embedding

    def calculate_trust_consistency_loss(self, trust_prediction, fusion_prediction, mask_label):
        eps = 1e-8
        if torch.sum(mask_label) == 0:
            return torch.tensor(0.0, device=trust_prediction.device)

        squared_error = torch.pow(trust_prediction - fusion_prediction.detach(), 2) * mask_label
        valid_samples = (torch.sum(mask_label, dim=1) > 0).float()
        per_sample_error = torch.sum(squared_error, dim=1)
        consistency_loss = torch.sum(per_sample_error * valid_samples) / (torch.sum(valid_samples) + eps)

        return consistency_loss
    
    def _calculate_legacy_trust_loss(self, view_embeddings, labels, mask_label):
        trust_scores, _ = self.trust_evaluator(view_embeddings, None)
        
        trust_weights = F.softmax(trust_scores / self.trust_temperature, dim=1)
        
        fused_embedding = self.trust_weighted_sum(view_embeddings, trust_weights)
        
        mean_logits = self.uncertainty_quantifier.uncertainty_network(fused_embedding)[:, :self.class_num]
        
        # 使用sigmoid激活确保值在[0,1]范围内
        mean_probs = torch.sigmoid(mean_logits)
        
        eps = 1e-8
        loss_ce = -(mask_label * (labels * torch.log(mean_probs + eps) + (1 - labels) * torch.log(1 - mean_probs + eps)))
        loss_ce = torch.sum(loss_ce) / (torch.sum(mask_label) + 1e-8)
        
        trust_consistency = torch.std(trust_weights, dim=1).mean()
        
        total_trust_loss = loss_ce + self.trust_consistency_weight * trust_consistency
        
        return total_trust_loss, loss_ce, trust_consistency
    
    def calculate_uncertainty_loss(self, target_prediction, mask_label, mean_logits, variance):
        eps = 1e-8
        
        # 确保variance为正数
        variance = torch.clamp(variance, min=eps)
        
        log_variance = torch.log(variance)
        
        pred_probs = torch.sigmoid(mean_logits)
        prediction_error = torch.pow(target_prediction.detach() - pred_probs, 2)
        negative_log_likelihood = 0.5 * (log_variance + prediction_error / variance)
        negative_log_likelihood = negative_log_likelihood * mask_label
        nll_loss = torch.sum(negative_log_likelihood) / (torch.sum(mask_label) + 1e-8)
        
        # 确保uncertainty_penalty为正数
        uncertainty_penalty = torch.mean(variance)
        
        total_uncertainty_loss = nll_loss + self.uncertainty_weight * uncertainty_penalty
        
        # 确保损失值为正数
        total_uncertainty_loss = torch.clamp(total_uncertainty_loss, min=0.0)
        
        return total_uncertainty_loss, nll_loss, uncertainty_penalty
    
