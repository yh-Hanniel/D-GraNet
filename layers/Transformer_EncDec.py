import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import hier_half_token_weight
from layers.Embed import PatchEmbedding
import math
import random
from typing import List
from layers.RevIN import RevIN
from utils.CKA import CudaCKA


# =========================================================================
# 辅助函数 (保持不变)
# =========================================================================
def norm_layer(type_name, channels):
    if type_name == 'batchnorm':
        return nn.BatchNorm2d(channels)
    elif type_name == 'layernorm':
        return nn.GroupNorm(1, channels)
    else:
        return nn.Identity()


def act_layer(type_name):
    if isinstance(type_name, str):
        if type_name.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif type_name.lower() == 'gelu':
            return nn.GELU()
    return nn.Identity()


class Seq(nn.Sequential):
    pass


def initialize_memberships(B, N, K, device):
    M = torch.rand(B, N, K, device=device)
    M = M / M.sum(dim=2, keepdim=True)
    return M


def fuzzy_c_means(x, n_clusters, m=2, epsilon=1e-6, max_iter=10):
    B, C, N, _ = x.shape
    X = x.squeeze(-1).transpose(1, 2)  # [B, N, C]
    M = initialize_memberships(B, N, n_clusters, x.device)
    for _ in range(max_iter):
        W = M.pow(m).unsqueeze(2)  # [B, N, 1, K]
        num = (W * X.unsqueeze(3)).sum(dim=1)  # [B, C, K]
        den = W.sum(dim=1).clamp_min(epsilon)  # [B, 1, K]
        centers = num / den  # [B, C, K]
        diff = X.unsqueeze(2) - centers.transpose(1, 2).unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=3) + epsilon)
        inv = dist.pow(-2 / (m - 1))
        M_new = inv / inv.sum(dim=2, keepdim=True).clamp_min(epsilon)
        if torch.max(torch.abs(M_new - M)) < epsilon:
            M = M_new
            break
        M = M_new
    return M, centers


# =========================================================================
# 2. 核心 Block (修改版：支持超图内部消融)
# =========================================================================
class HypergraphBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes,
                 num_dynamic_clusters=15, num_static_clusters=15,
                 m=2, threshold=0.5, act='relu', norm='batchnorm',
                 dropout=0.1, bias=True,
                 use_dynamic=True, use_static=True):  # 新增开关
        super().__init__()
        self.use_dynamic = use_dynamic
        self.use_static = use_static
        self.num_dynamic_clusters = num_dynamic_clusters
        self.num_static_clusters = num_static_clusters
        self.m = m
        self.threshold = threshold
        self.dropout = nn.Dropout(dropout)

        if use_static and num_static_clusters > 0:
            self.static_adj_param = nn.Parameter(torch.Tensor(1, num_nodes, num_static_clusters))
            nn.init.xavier_uniform_(self.static_adj_param)
            self.static_centers_param = nn.Parameter(torch.Tensor(1, in_channels, num_static_clusters))
            nn.init.xavier_normal_(self.static_centers_param)
        else:
            self.register_parameter('static_adj_param', None)
            self.register_parameter('static_centers_param', None)

        self.node2he = Seq(
            nn.Conv2d(in_channels, in_channels, 1, bias=bias),
            norm_layer(norm, in_channels),
            act_layer(act),
            nn.Dropout(dropout)
        )
        self.he2node = Seq(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            norm_layer(norm, out_channels),
            act_layer(act),
            nn.Dropout(dropout)
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, u):
        shortcut = u
        B, C, New_N, _ = u.shape

        # 1. 动态分支
        if self.use_dynamic:
            M_dyn, centers_dyn = fuzzy_c_means(u, self.num_dynamic_clusters, self.m)
            A_dyn = (M_dyn > self.threshold).float()

        # 2. 静态分支
        if self.use_static and self.static_adj_param is not None:
            A_sta = F.softmax(self.static_adj_param, dim=2).expand(B, -1, -1)
            centers_sta = self.static_centers_param.expand(B, -1, -1)

        # 3. 融合判断 (消融核心)
        if self.use_dynamic and self.use_static:
            A = torch.cat([A_dyn, A_sta], dim=2)
            centers = torch.cat([centers_dyn, centers_sta], dim=2)
        elif self.use_dynamic:
            A, centers = A_dyn, centers_dyn
        elif self.use_static:
            A, centers = A_sta, centers_sta
        else:
            return shortcut  # 去除所有超图子项

        A = self.dropout(A)
        X = u.squeeze(-1).transpose(1, 2)
        H = torch.bmm(A.permute(0, 2, 1), X)
        H = H.permute(0, 2, 1).unsqueeze(-1)
        agg_h = self.node2he(H) + (1 + self.eps) * centers.unsqueeze(-1)
        H2 = agg_h.squeeze(-1).permute(0, 2, 1)
        Nf = torch.bmm(A, H2)
        Nf = Nf.permute(0, 2, 1).unsqueeze(-1)
        out = self.he2node(Nf)
        return out + shortcut


# =========================================================================
# 3. LinearEncoder (修改版：支持全局图、超图及FFN消融)
# =========================================================================
class LinearEncoder(nn.Module):
    def __init__(self, d_model, d_ff=None, CovMat=None, dropout=0.0, activation="relu",
                 token_num=None, patch_size=256,
                 w_ratio=0.05, num_dynamic_clusters=15, num_static_clusters=15,
                 m=2.0, threshold=0.5,
                 use_global_graph=True, use_hyper_graph=True, use_ffn=True,  # 新增消融开关
                 use_dynamic=True, use_static_h=True,  # 传递给超图内部的开关
                 **kwargs):
        super(LinearEncoder, self).__init__()

        self.d_model = d_model
        self.use_global_graph = use_global_graph
        self.use_hyper_graph = use_hyper_graph
        self.use_ffn = use_ffn
        self.patch_size = patch_size
        self.token_num = token_num

        # 自动确定输出维度 (若双图并存则各分一半，否则独占全维度)
        if use_global_graph and use_hyper_graph:
            self.d_out = d_model // 2
        else:
            self.d_out = d_model

        # --- 静态全局图分支 ---
        if use_global_graph:
            self.w_ratio = w_ratio
            init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0
            self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])
            self.v_proj_static = nn.Linear(d_model, d_model)
            self.out_proj_static = nn.Linear(d_model, self.d_out)

        # --- PatchHGNN 分支 ---
        if use_hyper_graph:
            self.num_patches = self.d_model // self.patch_size
            total_nodes_hgnn = token_num * self.num_patches
            self.hgnn_block = HypergraphBlock(
                in_channels=patch_size, out_channels=patch_size,
                num_nodes=total_nodes_hgnn, num_dynamic_clusters=num_dynamic_clusters,
                num_static_clusters=num_static_clusters, m=m, threshold=threshold,
                act=activation, dropout=dropout,
                use_dynamic=use_dynamic, use_static=use_static_h
            )
            self.out_proj_hgnn = nn.Linear(d_model, self.d_out)

        # 融合层
        if use_global_graph and use_hyper_graph:
            self.fusion_mix = nn.Linear(d_model, d_model)
        else:
            self.fusion_mix = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN 分支
        if use_ffn:
            d_ff = d_ff or 4 * d_model
            self.conv1 = nn.Conv1d(d_model, d_ff, 1)
            self.conv2 = nn.Conv1d(d_ff, d_model, 1)
            self.activation = F.relu if activation == "relu" else F.gelu
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        B, N, D = x.shape
        residual = x
        branch_feats = []

        # A. Static Graph 分支
        if self.use_global_graph:
            val_static = self.v_proj_static(x)
            a = self.weight_mat.squeeze(0)
            adj = F.relu(torch.tanh(a))
            adj = adj - torch.diag(torch.diag(adj))
            k = int(adj.numel() * self.w_ratio)
            values_topk, indices = torch.topk(adj.reshape(-1), k, largest=True)
            mask = torch.zeros_like(adj.reshape(-1), device=adj.device)
            mask[indices] = 1
            adj = mask.view_as(adj) * adj
            adj = adj + torch.eye(adj.size(0)).to(adj.device)
            A = adj / (adj.sum(1) + 1e-6).view(-1, 1)
            feat_static = self.out_proj_static(A @ val_static)
            branch_feats.append(feat_static)

        # B. PatchHGNN 分支
        if self.use_hyper_graph:
            if not x.is_contiguous(): x = x.contiguous()
            x_reshaped = x.view(B, N, self.num_patches, self.patch_size)
            x_nodes = x_reshaped.flatten(1, 2)
            u = x_nodes.permute(0, 2, 1).unsqueeze(-1)
            out_u = self.hgnn_block(u)
            out_nodes = out_u.squeeze(-1).permute(0, 2, 1).contiguous()
            full_feat_hgnn = out_nodes.view(B, N, D)
            feat_hgnn = self.out_proj_hgnn(full_feat_hgnn)
            branch_feats.append(feat_hgnn)

        # C. 拼接与残差
        if not branch_feats:  # 去除所有图的情况
            combined = torch.zeros_like(x)
        else:
            combined = torch.cat(branch_feats, dim=-1)

        combined = self.fusion_mix(combined)
        x = self.norm1(residual + self.dropout(combined))

        # D. FFN 分支
        if self.use_ffn:
            y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            x = self.norm2(x + y)

        return x, None


# Encoder_ori 保持不变
class Encoder_ori(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, one_output=False, CKA_flag=False):
        super(Encoder_ori, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.one_output = one_output
        self.CKA_flag = CKA_flag

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        X0 = None
        layer_len = len(self.attn_layers)
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
            if not self.training and self.CKA_flag and layer_len > 1:
                if i == 0: X0 = x
                if i == layer_len - 1 and random.uniform(0, 1) < 1e-1:
                    cka_val = CudaCKA(device=x.device).linear_CKA(X0.flatten(0, 1)[:1000], x.flatten(0, 1)[:1000])
                    print(f'CKA: \t{cka_val:.3f}')
        if isinstance(x, (tuple, list)): x = x[0]
        if self.norm is not None: x = self.norm(x)
        return x if self.one_output else (x, attns)