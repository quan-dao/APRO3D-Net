import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): (B, *)

        Returns:
            torch.Tensor: (B, *)
        """
        return drop_path(x, self.drop_prob, self.training)


class VectorAttentionLayerV3(nn.Module):
    """LOCAL Vector Attention"""
    def __init__(self, d, hidden_scale, **kwargs):
        super().__init__()
        self.d = d
        self.make_Q = nn.Linear(d, d, bias=False)
        self.make_KV = nn.Linear(d, d * 2, bias=False)
        d_hidden = int(d * hidden_scale)
        self.mlp = nn.Sequential(
            nn.Conv2d(d, d_hidden, 1, bias=False),
            nn.BatchNorm2d(d_hidden),
            nn.ReLU(True),
            nn.Conv2d(d_hidden, d, 1, bias=False)
        )

    def forward(self, center_feat, neighbor_feat, pos_enc, return_QKV=False):
        """

        Args:
            center_feat (torch.Tenor): (B, N, C)
            neighbor_feat (torch.Tensor):  (B, N, M, C)
            pos_enc (torch.Tensor): (B, N, M, C)
            return_QKV (bool):

        Returns:
            new_center_feat (torch.Tensor): (B, N, C)
        """
        Q = self.make_Q(center_feat)  # (B, N, C)
        K, V = torch.chunk(self.make_KV(neighbor_feat), 2, dim=-1)  # (B, N, M, C) - each
        # print(f"Q.shape: {Q.shape}| K.shape: {K.shape}| pos_enc.shape: {pos_enc.shape}")
        attn_mat = Q.unsqueeze(2) - K + pos_enc  # (B, N, M, C)
        attn_mat = self.mlp(attn_mat.permute(0, 3, 1, 2).contiguous())  # (B, C, N, M)
        attn_mat = attn_mat.permute(0, 2, 3, 1).contiguous()  # (B, N, M, C)

        new_center_feat = torch.softmax(attn_mat, dim=2) * (V + pos_enc)  # (B, N, M, C)
        new_center_feat = torch.sum(new_center_feat, dim=2)  # (B, N, C)

        if return_QKV:
            return new_center_feat, Q, K, V
        return new_center_feat


class MetaFormer(nn.Module):

    def __init__(self, d, d_pos_emb, drop_path_prob, hidden_scale=2., drop_mlp_prob=0.):
        super().__init__()
        self.d = d
        d_hidden = int(d * hidden_scale)
        mixing_layer = VectorAttentionLayerV3

        self.norm_in_mixer = nn.BatchNorm1d(d)
        self.mixer = mixing_layer(d, hidden_scale)

        self.norm_in_mlp = nn.BatchNorm1d(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.GELU(),
            nn.Dropout(drop_mlp_prob),
            nn.Linear(d_hidden, d),
            nn.Dropout(drop_mlp_prob)
        )
        if d_pos_emb > 0:
            self.pos_encoder = nn.Sequential(
                nn.Conv2d(d_pos_emb, d_hidden, 1, bias=False),
                nn.BatchNorm2d(d_hidden),
                nn.GELU(),
                nn.Conv2d(d_hidden, d, 1, bias=False)
            )
        else:
            self.pos_encoder = None

        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def global_mixing(self, feat, pos_emb, n_key_token, **kwargs):
        """
        Mixing "key" tokens with every token in the sequence

        Args:
            feat (torch.Tensor): (B, N, C) - feat of full sequences
            pos_emb (torch.Tensor): (B, N, d_pos_emb) - position embedding of full sequences
            n_key_token (int): number of key tokens

        Returns:
            new_key_feat (torch.Tensor): (B, n_key_token, C)
        """
        B, n_token, C = feat.shape
        assert n_key_token <= n_token, "n_key_token must not be bigger than total number of tokens"

        '''
        NOTE: key tokens must be places at the beginning of feat & pos_emb
        '''

        '''
        norm -> token_mixer -> residual
        '''
        original_key_feat = feat[:, :n_key_token]  # (B, n_k, C)

        key_feat = self.norm_in_mixer(original_key_feat.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  # (B, n_k, C)

        pos_encoding = pos_emb[:, :n_key_token].unsqueeze(2) - pos_emb[:, n_key_token:].unsqueeze(1)  # (B, n_k, N_o, d_pos_emb)
        pos_encoding = self.pos_encoder(pos_encoding.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        # all_feat = torch.cat((key_feat, feat[:, n_key_token:]), dim=1)  # (B, N, C)
        other_feat = feat[:, n_key_token:]
        other_feat = other_feat.unsqueeze(1).repeat(1, n_key_token, 1, 1)  # (B, n_k, N_o, C)
        attn_key_feat = self.mixer(key_feat, other_feat, pos_encoding)

        key_feat = original_key_feat + self.drop_path(attn_key_feat)
        return key_feat

    def forward(self, feat, pos_emb, n_key_token):
        """

        Args:
            feat (torch.Tensor): (B, N, C) - feat of full sequences
            pos_emb (torch.Tensor): (B, N, d_pos_emb) - position embedding of full sequences
            n_key_token: number of key tokens

        Returns:
            new_key_feat (torch.Tensor): (B, n_key_token, C)
        """
        key_feat = self.global_mixing(feat, pos_emb, n_key_token)

        '''
        norm -> mlp -> residual
        '''
        final_key_feat = key_feat
        key_feat = self.norm_in_mlp(key_feat.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        key_feat = self.mlp(key_feat)
        final_key_feat = final_key_feat + self.drop_path(key_feat)  # (B, n_key_token, C)

        B, _, C = feat.shape
        assert final_key_feat.shape == (B, n_key_token, C), \
            f"final_key_feat.shape: {final_key_feat.shape}, expect {(B, n_key_token, C)}"
        return final_key_feat


class StackMetaFormers(nn.Module):
    def __init__(self, n_transformers, d, d_pos_emb,
                 hidden_scale=2.0, drop_mlp_prob=0., drop_path_probs=None, last_drop_path_p=0.3):
        """

        Args:
            n_transformers (int): number of Transformers in this stack
            d (int): attention dimension
            d_pos_emb (int): dimension of position embedding
            hidden_scale (float):
            drop_mlp_prob (float):
            drop_path_probs (list[float]):
            last_drop_path_p (float): drop path prob of the last Transformer
        """
        super().__init__()
        self.n_transformers = n_transformers
        self.metaformer_stack = nn.ModuleList()
        for i in range(n_transformers):
            if drop_path_probs is None:
                drop_path_p = i * last_drop_path_p / float(n_transformers - 1) if n_transformers > 1 else last_drop_path_p
            else:
                drop_path_p = drop_path_probs[i]
            self.metaformer_stack.append(MetaFormer(d, d_pos_emb, drop_path_p, hidden_scale, drop_mlp_prob))

    def global_mixing(self, feat, pos_emb, n_key_token):
        """
        Invoke `forward()` of Transformer

        Args:
            feat (torch.Tensor): (B, N, C) - feat of full sequence
            pos_emb (torch.Tensor): (B, N, d_pos_emb) - position embedding of full sequence
            n_key_token (int): number of key tokens

        Returns:
            new_key_feat (torch.Tensor): (B, n_key_token, C)
        """
        B, N, C = feat.shape

        feat_ = feat
        for i in range(self.n_metaformers):
            new_key_feat = self.metaformer_stack[i](feat_, pos_emb, n_key_token)
            # update feat_ to be prepared for the next module
            feat_ = torch.cat((new_key_feat, feat[:, n_key_token:]), dim=1)
            assert feat_.shape == (B, N, C)

        new_key_feat = feat_[:, :n_key_token]
        return new_key_feat

    def forward(self, feat, pos_emb, n_key_token, src_names=None, key_feat=None, key_pos_emb=None):
        """
        Choose between "all at once" & "sequential"

        Args:
            feat (torch.Tensor): (B, N, C) - feat of full sequence
            pos_emb (torch.Tensor): (B, N, d_pos_emb) - position embedding of full sequence
            n_key_token (int): number of key tokens
            src_names (list[str]): to enforce order of in round-robin
            key_feat (torch.Tensor): (B, n_key_token, C)
            key_pos_emb (torch.Tensor): (B, n_key_token, d_init_pos_emb)

        Returns:
            new_key_feat (torch.Tensor): (B, n_key_token, C)
        """
        if src_names is None:
            # all-at-once: different sources are concatenated and then fed into stack of Transformers
            new_key_feat = self.global_mixing(feat, pos_emb, n_key_token)
        else:
            # sequential: sources are sequentially fea into each Transformer in the stack
            assert isinstance(key_feat, torch.Tensor) and isinstance(key_pos_emb, torch.Tensor)
            assert isinstance(feat, dict) and isinstance(pos_emb, dict)
            new_key_feat = self.sequential_attn(feat, pos_emb, src_names, key_feat, key_pos_emb)

        if isinstance(feat, torch.Tensor):
            B, _, C = feat.shape
        else:
            B, _, C = key_feat.shape
        assert new_key_feat.shape == (B, n_key_token, C)
        return new_key_feat

    def sequential_attn(self, pooled_feat, pooled_pos_emb, src_names, key_feat, key_pos_emb):
        """
        Round-robin mixing to handle multi-scale feat map
        Args:
            pooled_feat (dict):
                src_name (torch.Tensor): (B, N_pooled, C) - feat of pooled points (come from the feat map named src)
                    NOT INCLUDE feat of key points
            pooled_pos_emb (dict):
                src_name (torch.Tensor): (B, N_pooled, d_pos_emb) - pos embedding of pooled points.
                    NOT INCLUDE feat of key points
            src_names (list[str]): to enforce order of in round-robin
            key_feat (torch.Tensor): (B, n_key_token, C)
            key_pos_emb (torch.Tensor): (B, n_key_token, d_init_pos_emb)

        Returns:
            new_key_feat (torch.Tensor): (B, n_key_token, C)
        """
        B, n_key_token, C = key_feat.shape

        for i, src in enumerate(src_names):
            key_feat = self.metaformer_stack[i](
                torch.cat((key_feat, pooled_feat[src]), dim=1),
                torch.cat((key_pos_emb, pooled_pos_emb[src]), dim=1),
                n_key_token=key_feat.shape[1]
            )

        assert key_feat.shape == (B, n_key_token, C), f"key_feat.shape: {key_feat.shape}, expect: {(B, n_key_token, C)}"
        return key_feat
