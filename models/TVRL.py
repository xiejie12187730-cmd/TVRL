"""TVRL framework.

(a) VAE-based View Imputation.
(b) Dual-weighted Multi-view Representation Learning.
(c) Trustworthy Multi-view Multi-label Classification with Uncertainty Quantification.
"""

from argparse import Namespace
import torch
import torch.nn as nn
import torch.optim
from utils.evaluation import do_metric
import copy
from utils.util import AverageMeter, fusion_mask_similarity, get_similarity
from datasets.dataloader import get_dataloader
from models.basemodel import MultiLabelClassifier, Decoder, Encoder, CrossViewProjectionHead
from models.trustworthy import TrustworthyMultiLabelClassification
from models.dynamic_weight_scheduler import DynamicLossWeightScheduler
from models.vae_imputation import VAEViewImputation
from utils.loss import MaskedCrossViewContrastiveLoss, ViewReconstructionLoss, MaskedBCELoss
from torch.nn.functional import normalize
import time


class TVRL(nn.Module):
    """TVRL."""
    def __init__(self, args: Namespace, logger):
        super(TVRL, self).__init__()
        self.sample_adj = None
        self.args = args
        self.logger = logger
        self.view_num = args.view_num
        self.class_num = args.class_num
        self.hidden = self.args.hidden
        self.encoder_dims = [[self.args.input_dims[i]] + [self.args.mid_dim]*self.args.layer_num + [self.args.hidden] for i in range(self.view_num)]
        
        # Framework (b): view-specific representation encoders.
        self.encoders = [Encoder(self.encoder_dims[i], dropout=self.args.dropout) for i in range(self.view_num)]
        self.encoders = nn.ModuleList(self.encoders)
        
        self.instance_project = [CrossViewProjectionHead(self.hidden) for _ in range(self.view_num)]
        self.instance_project = nn.ModuleList(self.instance_project)
        
        self.classify = MultiLabelClassifier(self.hidden, self.class_num)
        
        self.decoder_dims = [[self.args.hidden] + [self.args.mid_dim]*self.args.layer_num + [self.args.input_dims[i]] for i in range(self.view_num)]
        self.decoders = [Decoder(self.decoder_dims[i], dropout=self.args.dropout) for i in range(self.view_num)]
        self.decoders = nn.ModuleList(self.decoders)
        
        self.criterion_con = MaskedCrossViewContrastiveLoss(self.args.ins_temperature)
        self.criterion_bce = MaskedBCELoss()
        self.criterion_rec = ViewReconstructionLoss()
        
        self.loss_con = torch.tensor(0.0).to(self.args.device)
        self.loss_bce = torch.tensor(0.0).to(self.args.device)
        self.loss_rec = torch.tensor(0.0).to(self.args.device)
        
        # Framework (b) reliability estimation and (c) auxiliary prediction.
        self.trustworthy_module = None
        if getattr(args, 'use_trustworthy', True):
            self.trustworthy_module = TrustworthyMultiLabelClassification(args, logger)
        
        # Framework (a): VAE-based View Imputation.
        self.vae_imputation = None
        if getattr(args, 'use_vae_imputation', True):
            self.vae_imputation = VAEViewImputation(
                view_dims=args.input_dims,
                hidden_dim=getattr(args, 'vae_hidden_dim', 512),
                latent_dim=getattr(args, 'vae_latent_dim', 256),
                dropout=self.args.dropout
            )
            # 将VAE模块添加为子模块，确保.to(device)时也会移动到GPU
            self.add_module('vae_imputation_module', self.vae_imputation)
            self.vae_recon_weight = getattr(args, 'vae_recon_weight', 1.0)
            self.vae_kl_weight = getattr(args, 'vae_kl_weight', 0.1)
        
        # 人工遮挡重建训练参数
        self.use_artificial_mask = getattr(args, 'use_artificial_mask', True)
        self.artificial_mask_ratio = getattr(args, 'artificial_mask_ratio', 0.3)
        self.lambda_masked_recon = getattr(args, 'lambda_masked_recon', 0.5)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        # Dynamic loss weight scheduling for alpha, beta, gamma and lambda_t.
        self.weight_scheduler = DynamicLossWeightScheduler(args, self.args.epochs)

    def make_adj(self, train_dataset):
        train_mask_data = train_dataset.mask_data
        train_datas = train_dataset.data

        start = time.time()
        dists = []
        for i in range(self.view_num):
            data_com = train_datas[i][train_mask_data[:, i] == 1]
            dist = get_similarity(data_com)

            dists.append(dist)
        print("Make view similarity time:{:.2f}s".format(time.time() - start))

        start = time.time()
        sim_f = fusion_mask_similarity(dists, train_mask_data)

        print("Make fusion similarity time:{:.2f}s".format(time.time() - start))

        sim_f = sim_f.to(self.args.device)

        return sim_f

    def run_train(self, train_dataset, val_dataset, test_dataset, mode="train"):
        print("run.....")
        best_result = 0
        best_model = self.state_dict()
        best_epoch = -1

        self.sample_adj = self.make_adj(train_dataset)

        train_loader = get_dataloader(self.args, train_dataset, mode)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        for epoch in range(self.args.epochs):
            self.train()
            losses = AverageMeter()
            loss_con = AverageMeter()
            loss_bce = AverageMeter()
            loss_rec = AverageMeter()
            loss_trust = AverageMeter()
            loss_unc = AverageMeter()
            loss_vae = AverageMeter()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 获取当前epoch的动态权重
            weights = self.weight_scheduler.get_weights(epoch)
            current_alpha = weights['alpha']
            current_beta = weights['beta']
            current_gamma = weights['gamma']
            current_lambda_t = weights['lambda_t']
            current_uncertainty_loss_weight = weights['uncertainty_loss_weight']
            
            for _, (train_data, true_label, mask_data, mask_label, indices) in enumerate(train_loader):
                self.loss_con = torch.tensor(0.0).to(self.args.device)
                self.loss_bce = torch.tensor(0.0).to(self.args.device)
                self.loss_rec = torch.tensor(0.0).to(self.args.device)
                self.loss_vae = torch.tensor(0.0).to(self.args.device)
                
                true_label = true_label.to(self.args.device)
                mask_data = mask_data.to(self.args.device)
                mask_label = mask_label.to(self.args.device)

                sample_adj = self.sample_adj[indices[:, None], indices]

                # 将train_data移动到设备上
                train_data = [data.to(self.args.device) for data in train_data]

                # 使用VAE进行缺失数据插补（可选地应用人工遮挡重建训练）
                if self.vae_imputation is not None:
                    train_data_imputed, vae_outputs = self.vae_imputation(
                        train_data, mask_data,
                        apply_artificial_mask=self.use_artificial_mask,
                        artificial_mask_ratio=self.artificial_mask_ratio
                    )
                    
                    artificial_mask = vae_outputs.get('artificial_mask', None)
                    original_mask_for_recon = vae_outputs.get('original_mask', mask_data)
                    
                    # 主Decoder使用原始mask进行重建计算
                    forward_result = self(train_data_imputed, mask_data)
                    
                    # 计算VAE损失（使用原始train_data和原始mask计算loss_obs）
                    vae_losses = self.vae_imputation.compute_loss(
                        train_data, original_mask_for_recon, vae_outputs,
                        recon_weight=self.vae_recon_weight,
                        kl_weight=self.vae_kl_weight,
                        lambda_masked=self.lambda_masked_recon
                    )
                    self.loss_vae = current_alpha * vae_losses['total_loss']
                else:
                    train_data_imputed = train_data
                    artificial_mask = None
                    forward_result = self(train_data_imputed, mask_data)
                z, z_pro, label_pred, label_pred_fusion, x_reconstructed = forward_result
                
                for v in range(self.view_num):
                    for w in range(v + 1, self.view_num):
                        mask_v = torch.ones_like(mask_data[:, v])
                        ones_positions = torch.nonzero(mask_data[:, v] == 1).squeeze()
                        drop_count = int(self.args.mask_rate * len(ones_positions))
                        drop_indices = torch.randperm(len(ones_positions))[:drop_count]
                        mask_v[ones_positions[drop_indices]] = 0
                        mask_v = mask_v * mask_data[:, v]

                        mask_w = torch.ones_like(mask_data[:, w])
                        ones_positions = torch.nonzero(mask_data[:, w] == 1).squeeze()
                        drop_count = int(self.args.mask_rate * len(ones_positions))
                        drop_indices = torch.randperm(len(ones_positions))[:drop_count]
                        mask_w[ones_positions[drop_indices]] = 0
                        mask_w = mask_w * mask_data[:, w]

                        self.loss_con += self.criterion_con(z_pro[v], z_pro[w], mask_v, mask_w, sample_adj)  

                self.loss_con *= current_gamma

                for i in range(self.view_num):
                    self.loss_bce += self.criterion_bce(true_label[mask_data[:, i].bool()],
                                                        label_pred[i][mask_data[:, i].bool()],
                                                        mask_label[mask_data[:, i].bool()])
                self.loss_bce += self.criterion_bce(true_label, label_pred_fusion, mask_label)
                self.loss_bce /= (self.view_num + 1)

                # 主Decoder重建损失（支持人工遮挡重建训练）
                if self.use_artificial_mask and self.vae_imputation is not None and artificial_mask is not None:
                    recon_result, _ = self.criterion_rec(
                        x_reconstructed, train_data_imputed, mask_data, 
                        artificial_mask, self.lambda_masked_recon
                    )
                    self.loss_rec = recon_result * current_beta
                else:
                    recon_result, _ = self.criterion_rec(
                        x_reconstructed, train_data_imputed, mask_data
                    )
                    self.loss_rec = recon_result * current_beta
                
                # L = alpha*L_vae + beta*L_rec + gamma*L_con + lambda_t*L_trust.
                # Weighted components are accumulated below.
                total_loss = self.loss_con + self.loss_rec
                
                if self.trustworthy_module is not None:
                    z_list = [zi.detach() for zi in z]
                    
                    uncertainty_result = self.trustworthy_module(z_list, mask_data, return_details=True)
                    mean_logits = uncertainty_result['mean_logits']
                    trust_prediction = uncertainty_result['trust_prediction']
                    variance = uncertainty_result['variance']

                    loss_tc = self.trustworthy_module.calculate_trust_consistency_loss(
                        trust_prediction, label_pred_fusion, mask_label
                    )
                    
                    loss_unc_value, _, _ = self.trustworthy_module.calculate_uncertainty_loss(
                        label_pred_fusion, mask_label, mean_logits, variance
                    )
                    
                    # Ensure loss values are valid before forming L_trust.
                    loss_tc = torch.clamp(loss_tc, min=0.0, max=100.0)
                    loss_unc_value = torch.clamp(loss_unc_value, min=0.0, max=100.0)
                    
                    # Framework (c): L_trust = L_bce + L_tc + L_unc.
                    loss_trust_value = self.loss_bce + loss_tc + loss_unc_value
                    self.loss_trust = current_lambda_t * loss_trust_value
                    self.loss_unc = loss_unc_value
                    
                    total_loss = total_loss + self.loss_trust
                else:
                    total_loss = total_loss + self.loss_bce
                
                # 添加VAE插补损失
                if self.vae_imputation is not None:
                    total_loss = total_loss + self.loss_vae
                
                # 确保总损失有效
                total_loss = torch.clamp(total_loss, min=0.0, max=1000.0)
                
                loss_con.update(self.loss_con.item())
                loss_bce.update(self.loss_bce.item())
                loss_rec.update(self.loss_rec.item())
                losses.update(total_loss.item())
                if self.trustworthy_module is not None:
                    loss_trust.update(self.loss_trust.item())
                    loss_unc.update(self.loss_unc.item())
                if self.vae_imputation is not None:
                    loss_vae.update(self.loss_vae.item())

                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            inform = (
                "[TVRL/train] ==> epoch[{}/{}], alpha:{:.4f}, beta:{:.4f}, gamma:{:.4f}, lambda_t:{:.4f}, uncertainty_weight:{:.4f}, loss_con:{:.2f}, loss_bce:{:.2f}, loss_rec:{:.2f}, loss:{:.2f}".format(
                    epoch + 1, self.args.epochs, current_alpha, current_beta, current_gamma, current_lambda_t, current_uncertainty_loss_weight, loss_con.avg, loss_bce.avg, loss_rec.avg, losses.avg,))
            if self.trustworthy_module is not None:
                inform += ", loss_trust:{:.2f}, loss_unc:{:.2f}".format(
                    loss_trust.avg if hasattr(loss_trust, 'avg') else 0,
                    loss_unc.avg if hasattr(loss_unc, 'avg') else 0
                )
            if self.vae_imputation is not None:
                inform += ", loss_vae:{:.2f}".format(
                    loss_vae.avg if hasattr(loss_vae, 'avg') else 0
                )
            self.logger.info(inform)

            if (epoch + 1) % 1 == 0:

                evaluation_results, inform_val = self.test(val_dataset, epoch, mode="val")
                self.logger.info(inform_val)

                if 2 * evaluation_results[0] + evaluation_results[1] + evaluation_results[2] + evaluation_results[3] > best_result:
                    best_result = 2 * evaluation_results[0] + evaluation_results[1] + evaluation_results[2] + evaluation_results[3]
                    best_model = copy.deepcopy(self.state_dict())
                    best_epoch = epoch

            if self.args.early_stop and epoch - best_epoch >= 50:
                break
        
        self.load_state_dict(best_model)

        final_test_res, final_test_inform = self.test(test_dataset, best_epoch, mode="test")

        final_test_inform = "TVRL FINAL TEST RESULTS: " + final_test_inform
        self.logger.info(final_test_inform)

        return final_test_res

    def test(self, test_dataset, epoch, mode="test"):
        self.eval()
        with torch.no_grad():
            test_datas = test_dataset.data
            test_mask_data = test_dataset.mask_data.to(self.args.device)
            test_true_labels = test_dataset.labels.to(self.args.device)

            # 将test_datas移动到设备上
            test_datas = [data.to(self.args.device) for data in test_datas]

            # 使用VAE进行缺失数据插补
            if self.vae_imputation is not None:
                test_datas_imputed, _ = self.vae_imputation(test_datas, test_mask_data)
            else:
                test_datas_imputed = test_datas

            forward_result = self(test_datas_imputed, test_mask_data)
            _, _, _, label_pred_fusion, _ = forward_result

            evaluation_results = do_metric(label_pred_fusion.cpu().numpy(), test_true_labels.cpu().numpy())
            inform = (
                '[TVRL/{}] ==> epoch[{}/{}]  AP:{:.4f}  1-HL:{:.4f}  1-RL:{:.4f}  AUC:{:.4f}  1-OE:{:.4f}  1-Cov:{:.4f}'.format(
                    mode, epoch + 1, self.args.epochs, evaluation_results[0], evaluation_results[1], evaluation_results[2],
                    evaluation_results[3], evaluation_results[4], evaluation_results[5]
                ))

        return evaluation_results, inform

    def forward(self, datas, mask_data):
        z = [None for _ in range(self.view_num)]
        z_pro = [None for _ in range(self.view_num)]
        label_pred = [None for _ in range(self.view_num)]
        x_reconstructed = [None for _ in range(self.view_num)]
        
        original_datas = []
        for i in range(self.view_num):
            original_datas.append(datas[i].to(self.args.device))

        for i in range(self.view_num):
            z[i] = self.encoders[i](original_datas[i])
            z_pro[i] = normalize(self.instance_project[i](z[i]), dim=1)
            label_pred[i] = self.classify(z[i])
            
            x_reconstructed[i] = self.decoders[i](z[i])

        z_complete = [None for _ in range(self.view_num)]
        for i in range(self.view_num):
            z_complete[i] = z[i].clone()
            
            if torch.sum(mask_data[:, i]) < mask_data.shape[0]:
                other_views = [z[j] for j in range(self.view_num) if j != i]
                if other_views:
                    other_views_avg = torch.mean(torch.stack(other_views), dim=0)
                    missing_indices = ~mask_data[:, i].bool()
                    if missing_indices.sum() > 0:
                        z_complete[i][missing_indices] = other_views_avg[missing_indices]
        
        z_fusion = torch.mean(torch.stack(z_complete), dim=0)

        label_pred_fusion = self.classify(z_fusion)
        
        if self.trustworthy_module is not None:
            z_for_trust = [zi.detach() for zi in z]
            self.trustworthy_module(z_for_trust, mask_data, return_details=True)

        return z, z_pro, label_pred, label_pred_fusion, x_reconstructed
