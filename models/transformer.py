import math
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention
from functools import reduce

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


class Transformer(nn.Module):

    def __init__(self, batch_size=16, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(batch_size, encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec, d_model=d_model)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.batch_size = batch_size
        self.proj_1 = nn.Linear(d_model, 1)
        self.self_attn = nn.MultiheadAttention(256, 1, dropout=0.1)
        self.dropout = nn.Dropout(0.1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def scale_top_k(self, visual_scale, k):
        idx_temp = nn.Softmax(dim=1)(self.proj_1(visual_scale)).squeeze(-1)
        a, idx = torch.sort(idx_temp, descending=True)  # descending=False-->low2high
        topk_idx = idx[:, :k]
        vv_k = torch.zeros((self.batch_size, k, 256)).cuda()
        for i in range(self.batch_size):
            xx = torch.index_select(visual_scale[i, :, :], 0, topk_idx[i, :])
            vv_k[i, :, :] = xx
        return vv_k

    def forward(self, src, patch, canny, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        i = 0
        j = 0
        k = 0
        memory, color_scale, pos_scale, canny_scale = self.encoder(src, bs, h, w, patch, canny, src_key_padding_mask=mask, pos=pos_embed)

        """color_select"""
        c0 = color_scale[0]
        c1 = color_scale[1]
        c2 = color_scale[2]
        c3 = color_scale[3]
        c4 = color_scale[4]
        c5 = color_scale[5]
        if i == 0:
            c_50 = c0
        if i == 1:
            cc = torch.cat((c0, c1), dim=1)
            c_50 = self.scale_top_k(cc, 50)
        if i == 2:
            cc = torch.cat((c0, c1, c2), dim=1)
            c_50 = self.scale_top_k(cc, 50)
        if i == 3:
            cc = torch.cat((c0, c1, c2, c3), dim=1)
            c_50 = self.scale_top_k(cc, 50)
        if i == 4:
            cc = torch.cat((c0, c1, c2, c3, c4), dim=1)
            c_50 = self.scale_top_k(cc, 50)
        if i == 5:
            cc = torch.cat((c0, c1, c2, c3, c4, c5), dim=1)
            c_50 = self.scale_top_k(cc, 50)
        """pos_select"""
        p0 = pos_scale[0]
        p1 = pos_scale[1]
        p2 = pos_scale[2]
        p3 = pos_scale[3]
        p4 = pos_scale[4]
        p5 = pos_scale[5]
        if j == 0:
            p_50 = p0
        if j == 1:
            pp = torch.cat((p0, p1), dim=1)
            p_50 = self.scale_top_k(pp, 50)
        if j == 2:
            pp = torch.cat((p0, p1, p2), dim=1)
            p_50 = self.scale_top_k(pp, 50)
        if j == 3:
            pp = torch.cat((p0, p1, p2, p3), dim=1)
            p_50 = self.scale_top_k(pp, 50)
        if j == 4:
            pp = torch.cat((p0, p1, p2, p3, p4), dim=1)
            p_50 = self.scale_top_k(pp, 50)
        if j == 5:
            pp = torch.cat((p0, p1, p2, p3, p4, p5), dim=1)
            p_50 = self.scale_top_k(pp, 50)
        """canny_select"""
        e0 = canny_scale[0]
        e1 = canny_scale[1]
        e2 = canny_scale[2]
        e3 = canny_scale[3]
        e4 = canny_scale[4]
        e5 = canny_scale[5]
        if k == 0:
            e_50 = e0
        if k == 1:
            ee = torch.cat((e0, e1), dim=1)
            e_50 = self.scale_top_k(ee, 50)
        if k == 2:
            ee = torch.cat((e0, e1, e2), dim=1)
            e_50 = self.scale_top_k(ee, 50)
        if k == 3:
            ee = torch.cat((e0, e1, e2, e3), dim=1)
            e_50 = self.scale_top_k(ee, 50)
        if k == 4:
            ee = torch.cat((e0, e1, e2, e3, e4), dim=1)
            e_50 = self.scale_top_k(ee, 50)
        if k == 5:
            ee = torch.cat((e0, e1, e2, e3, e4, e5), dim=1)
            e_50 = self.scale_top_k(ee, 50)

        aa = c_50 + p_50 + e_50
        cpem = torch.cat((aa, memory.transpose(0, 1)), dim=1).transpose(0, 1)
        cpem = self.dropout(self.self_attn(cpem, cpem, cpem)[0]).transpose(0, 1)
        tgt = self.scale_top_k(cpem[:, 50:, :], 50).transpose(0, 1)

        hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)

        i += 1
        j += 1
        k += 1

        return hs, references


class MyPool(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

        self.patch_pool_0 = nn.Parameter(nn.init.uniform_(torch.Tensor(batch_size, 196, 120)))
        self.patch_pool_1 = nn.Parameter(nn.init.uniform_(torch.Tensor(batch_size, 120, 100)))
        self.patch_pool_2 = nn.Parameter(nn.init.uniform_(torch.Tensor(batch_size, 100, 80)))
        self.patch_pool_3 = nn.Parameter(nn.init.uniform_(torch.Tensor(batch_size, 80, 70)))
        self.patch_pool_4 = nn.Parameter(nn.init.uniform_(torch.Tensor(batch_size, 70, 60)))
        self.patch_pool_5 = nn.Parameter(nn.init.uniform_(torch.Tensor(batch_size, 60, 50)))

    def forward(self, patch, layer_idx):
        if layer_idx == 0:
            return torch.bmm(nn.Softmax(dim=1)(self.patch_pool_0.transpose(1, 2)), patch)
        if layer_idx == 1:
            return torch.bmm(nn.Softmax(dim=1)(self.patch_pool_1.transpose(1, 2)), patch)
        if layer_idx == 2:
            return torch.bmm(nn.Softmax(dim=1)(self.patch_pool_2.transpose(1, 2)), patch)
        if layer_idx == 3:
            return torch.bmm(nn.Softmax(dim=1)(self.patch_pool_3.transpose(1, 2)), patch)
        if layer_idx == 4:
            return torch.bmm(nn.Softmax(dim=1)(self.patch_pool_4.transpose(1, 2)), patch)
        if layer_idx == 5:
            return torch.bmm(nn.Softmax(dim=1)(self.patch_pool_5.transpose(1, 2)), patch)


class TransformerEncoder(nn.Module):

    def __init__(self, batch_size, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.B = batch_size

        self.color_proj = nn.Linear(9, 256)
        self.pos_proj = nn.Linear(4, 256)
        self.canny_tdlr_proj = nn.Linear(16, 128)
        self.canny_proj = nn.Linear(512, 256)
        self.myPool = MyPool(batch_size)
        self.self_attn_0 = nn.MultiheadAttention(256, 4, dropout=0.1)
        self.dropout_0 = nn.Dropout(0.1)
        self.weight_Wco = nn.Parameter(nn.init.uniform_(torch.empty(1)))
        self.weight_Wpo = nn.Parameter(nn.init.uniform_(torch.empty(1)))
        self.weight_Wc = nn.Parameter(nn.init.uniform_(torch.empty(1)))
        self.proj_1 = nn.Linear(256, 1)

    def forward(self, src, bs, h, w, patch, canny,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        """step 1"""
        patch_color = F.normalize(self.color_proj(patch[:, :, :9]))
        patch_pos = F.normalize(self.pos_proj(patch[:, :, 9:]))

        canny_top = F.normalize(self.canny_tdlr_proj(canny[:, :, :16]))
        canny_down = F.normalize(self.canny_tdlr_proj(canny[:, :, 16:32]))
        canny_left = F.normalize(self.canny_tdlr_proj(canny[:, :, 32:48]))
        canny_right = F.normalize(self.canny_tdlr_proj(canny[:, :, 48:64]))

        color_scale = []
        pos_scale = []
        canny_scale = []
        output_scale = []
        for i, layer in enumerate(self.layers):
            """Wco"""
            Wco = nn.Softmax(dim=-1)(torch.bmm(patch_color, patch_color.transpose(1, 2)))
            patch_color = torch.bmm(Wco, patch_color)
            patch_color = F.normalize(self.myPool(patch_color, i))
            color_scale.append(patch_color)
            """Wpo"""
            Wpo = nn.Softmax(dim=-1)(torch.bmm(patch_pos, patch_pos.transpose(1, 2)))
            patch_pos = torch.bmm(Wpo, patch_pos)
            patch_pos = F.normalize(self.myPool(patch_pos, i))
            pos_scale.append(patch_pos)
            """欧式距离"""
            # Wpo = []
            # for k in range(self.B):
            #     po_dis = self.euclidean_dist(patch_pos[k, :, :], patch_pos[k, :, :])
            #     po_sim = torch.exp(0 - po_dis / 0.1)
            #     Wpo.append(po_sim)
            # Wpo = torch.stack(Wpo)
            # patch_pos = torch.bmm(Wpo, patch_pos)
            # patch_pos = F.normalize(self.myPool(patch_pos, i))
            # pos_scale.append(patch_pos)

            """Wc"""
            top_down = torch.cat((canny_top.unsqueeze(2), canny_down.unsqueeze(2)), dim=2)
            left_right = torch.cat((canny_left.unsqueeze(2), canny_right.unsqueeze(2)), dim=2)
            td_sim = nn.Softmax(dim=-1)(torch.matmul(top_down, top_down.transpose(3, 2)))
            sigma_12 = torch.cat((td_sim[:, :, 0, 1].unsqueeze(-1), td_sim[:, :, 1, 0].unsqueeze(-1)), dim=-1)
            lr_sim = nn.Softmax(dim=-1)(torch.matmul(left_right, left_right.transpose(3, 2)))
            sigma_34 = torch.cat((lr_sim[:, :, 0, 1].unsqueeze(-1), lr_sim[:, :, 1, 0].unsqueeze(-1)), dim=-1)
            max_1234 = torch.max(torch.cat((sigma_12, sigma_34), dim=-1), 2).values  # [B,196]

            if i == 0:
                Wc = torch.zeros((self.B, 196, 196)).cuda()
                for j in range(196):
                    Wc[:, j, :] = max_1234
            if i == 1:
                Wc = torch.zeros((self.B, 120, 120)).cuda()
                for j in range(120):
                    Wc[:, j, :] = max_1234
            if i == 2:
                Wc = torch.zeros((self.B, 100, 100)).cuda()
                for j in range(100):
                    Wc[:, j, :] = max_1234
            if i == 3:
                Wc = torch.zeros((self.B, 80, 80)).cuda()
                for j in range(80):
                    Wc[:, j, :] = max_1234
            if i == 4:
                Wc = torch.zeros((self.B, 70, 70)).cuda()
                for j in range(70):
                    Wc[:, j, :] = max_1234
            if i == 5:
                Wc = torch.zeros((self.B, 60, 60)).cuda()
                for j in range(60):
                    Wc[:, j, :] = max_1234

            canny_feat = torch.cat((canny_top, canny_down, canny_left, canny_right), dim=-1)
            canny_feat = torch.bmm(Wc, canny_feat)
            canny_feat = F.normalize(self.myPool(canny_feat, i))
            ########################
            canny_top = canny_feat[:, :, :128]
            canny_down = canny_feat[:, :, 128:256]
            canny_left = canny_feat[:, :, 256:384]
            canny_right = canny_feat[:, :, 384:]
            canny_feat = self.canny_proj(canny_feat)
            canny_scale.append(canny_feat)

            patch = torch.cat((self.weight_Wco * patch_color, self.weight_Wpo * patch_pos, self.weight_Wc * canny_feat), dim=1)

            patch = self.dropout_0(self.self_attn_0(patch.transpose(0, 1), patch.transpose(0, 1), patch.transpose(0, 1))[0].transpose(0, 1))

            src_patch_W = nn.Softmax(dim=-1)(torch.bmm(output.transpose(0, 1), patch.transpose(1, 2)))
            output = torch.bmm(src_patch_W, patch).transpose(0, 1) + output

            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            output_scale.append(output)

        if self.norm is not None:
            output = reduce(lambda x, y: x + y, output_scale)
            output = self.norm(output)

        return output, color_scale, pos_scale, canny_scale


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(d_model, d_model, 2, 2)
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        # patch_pos = output[50:, :, :]
        # pp_qp_sim = nn.Softmax(dim=-1)(torch.bmm(query_pos.permute(1,0,2), patch_pos.permute(1,2,0)))
        # query_pos_1 = F.normalize(torch.bmm(pp_qp_sim, patch_pos.transpose(0, 1)).transpose(0, 1)) + query_pos

        intermediate = []
        # reference_points_before_sigmoid = self.ref_point_head(output)
        reference_points_before_sigmoid = self.ref_point_head(query_pos)
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)
            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)

            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return [torch.stack(intermediate).transpose(1, 2), reference_points]

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed=None,
                     is_first=False):

        # ========== Begin of Self-Attention =============
        q_content = self.sa_qcontent_proj(tgt)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)

        v = self.sa_v_proj(tgt)

        # num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256

        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False):
        if self.normalize_before:
            raise NotImplementedError
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed, is_first)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        batch_size=args.batch_size,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

