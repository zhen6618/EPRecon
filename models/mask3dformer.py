import logging
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.cuda.amp import autocast

from models.voxel_position_encoding import PositionEmbeddingCoordsSine
from utils import sparse_to_dense_channel, sparse_to_dense_torch


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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


class MultiScaleMaskedTransformerDecoder(nn.Module):
    _version = 2

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    def __init__(
            self,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # voxel positional encoding
        self.pos_enc_type = "fourier"

        self.num_queries = num_queries

        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        if self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=mask_dim,
                gauss_scale=1.0,
                normalize=True,
            )
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="sine",
                d_pos=mask_dim,
                normalize=True,
            )
        else:
            assert False, "pos enc type not known"

        for _ in range(self.num_layers):  # Each stage's transformer block loops num_layers times
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim*4, mask_dim, 3)  # pred mask

    def get_pos_encs(self, coords, spitial_shape):  # [B, N, 3]
        pos_encodings_pcd = []
        device = coords[0].device

        for i in range(len(coords)):
            for coords_batch in coords[i]:
                scene_min = torch.Tensor([0, 0, 0]).to(coords_batch.dtype).to(device)  # Absolute position encoding + normalized time scale fixed scaling
                scene_max = torch.Tensor([spitial_shape[0], spitial_shape[1], spitial_shape[2]]).to(coords_batch.dtype).to(device)
                scene_min = scene_min.view(-1, 3)
                scene_max = scene_max.view(-1, 3)

                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(), input_range=[scene_min, scene_max])  # tmp: [1, c, N]

                pos_encodings_pcd.append(tmp)  # tmp: [1, c, N]

        return pos_encodings_pcd


    def forward(self, panoptic_features, panoptic_coords, mask_features, spitial_shape):
        """
        :param panoptic_features: [bs=1, c, N_voxels]
        :param panoptic_coords: [bs=1, N_voxels, 3(xyz)]
        :param mask_features: [bs=1, c, N_voxels]
        """
        src = []
        size_list = []
        # get voxel positional encodings
        pos = self.get_pos_encs(panoptic_coords, spitial_shape)

        for i in range(self.num_feature_levels):
            size_list.append(panoptic_coords[i].shape[1])
            src.append(panoptic_features[i] + self.level_embed.weight[i][None, :, None])

            # flatten BxCxN to NxBxC
            pos[i] = pos[i].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # compute mask indices
        mask_indices = []
        # 0
        distances = torch.cdist(panoptic_coords[2].squeeze(0).float(), panoptic_coords[0].float(), p=2)  # Calculate Euclidean distance
        nearest_indices = torch.argmin(distances, dim=1)  # Find the index of the nearest neighbor
        mask_indices.append(nearest_indices.view(-1))
        # 1
        distances = torch.cdist(panoptic_coords[2].squeeze(0).float(), panoptic_coords[1].float(), p=2)
        nearest_indices = torch.argmin(distances, dim=1)
        mask_indices.append(nearest_indices.view(-1))
        # 2
        mask_indices.append(torch.ones(panoptic_coords[2].shape[1],  dtype=torch.bool, device=panoptic_coords[2].device))

        # QxBxC  query
        query_embed = self.query_embed.weight.unsqueeze(1)
        output = self.query_feat.weight.unsqueeze(1)

        predictions_class = []
        predictions_mask = []

        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output,
                                                                               mask_features,
                                                                               attn_mask_target_size=size_list[0],
                                                                               mask_indices=mask_indices[0],)

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for j in range(self.num_layers):  # transformer loop
            level_index = j % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[j](
                output, src[level_index],
                memory_mask=attn_mask,  # True means not to participate
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[j](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[j](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output,
                                                                                   mask_features,
                                                                                   attn_mask_target_size=size_list[(j + 1) % self.num_feature_levels],
                                                                                   mask_indices=mask_indices[(j + 1) % self.num_feature_levels],)

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, mask_indices):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bcl->bql", mask_embed, mask_features)

        # [B, Q, L] -> [B, Q, L] -> [B, h, Q, L] -> [B*h, Q, L]
        attn_mask = outputs_mask[..., mask_indices]
        # attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="linear", align_corners=False)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])  # Discard the last list actually output by the model
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


"******************************************   panoptic post-processing   **********************************************"
def panoptic_post(outputs,
                  semantic_on=False,
                  panoptic_on=True,
                  instance_on=False,
                  occupied=None,
                  ):

    mask_cls_results = outputs["pred_logits"]  # [bs=1, num_query, num_class+1]
    if occupied == None:
        mask_pred_results = outputs["pred_masks"]
    else:
        mask_pred_results = outputs["pred_masks"][..., occupied]  # [bs=1, num_query, num_voxel]

    # upsample masks if necessary
    # mask_pred_results = F.interpolate(
    #     mask_pred_results,
    #     size=(images.tensor.shape[-2], images.tensor.shape[-1]),
    #     mode="bilinear",
    #     align_corners=False,
    # )

    del outputs

    processed_results = {}
    for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):

        # semantic segmentation inference
        if semantic_on:
            r = semantic_inference(mask_cls_result, mask_pred_result)  # [num_class, num_voxel]
            processed_results["sem_seg"] = r

        # panoptic segmentation inference
        if panoptic_on:
            panoptic_r = panoptic_inference(mask_cls_result, mask_pred_result)
            processed_results["panoptic_seg"] = panoptic_r

        # instance segmentation inference
        if instance_on:
            instance_r = instance_inference(mask_cls_result, mask_pred_result)
            processed_results["instances"] = instance_r

    return processed_results


def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]  # K+1 discard the first empty category
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("qc,ql->cl", mask_cls, mask_pred)
    return semseg

# 1(1)-wall  2(2)-floor  3(3)-cabinet  4(4)-bed  5(5)-chair
# 6(6)-sofa  7(7)-table  8(8)-door  9(9)-window  10(10)-bookshelf
# 11(11)-picture  12(12)-counter  13(14)-desk  14(16)-curtain 15(24)-refrigerator
# 16(28)-shower curtain  17(33)-toilet  18(34)-sink  19(36)-bathtub  20(39)-otherfurniture
def panoptic_inference(mask_cls,
                       mask_pred,
                       object_mask_threshold=0.3,
                       thing_id=list(range(3, 21)),
                       overlap_threshold=0.5,
                       ):

    scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
    mask_pred = mask_pred.sigmoid()

    keep = labels.ne(0) & (scores > object_mask_threshold)  # score > thresd
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_masks = mask_pred[keep]

    cur_prob_masks = cur_scores.view(-1, 1) * cur_masks  # Consider both cls and mask

    l = cur_masks.shape[-1]
    panoptic_seg = torch.zeros((l), dtype=torch.int32, device=cur_masks.device)
    segments_info = []

    current_segment_id = 0
    pred_mask_count = 0

    if cur_masks.shape[0] == 0:
        # We didn't detect any mask :(
        return [panoptic_seg, segments_info]
    else:
        '--- take argmax ---'
        cur_mask_ids = cur_prob_masks.argmax(0)  # For each voxel, take the maximum probability of cls*mask
        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in thing_id
            mask_area = (cur_mask_ids == k).sum().item()  # [N_voxels,] For each query_mask, for each voxel, the current query is the most prominent one among all queries
            original_area = (cur_masks[k] >= 0.5).sum().item()  # [N_voxels,] For each query_mask>0.5
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)  # [N_voxels,] For each voxel, the most prominent one among all queries with a value >= 0.5 is assigned
            # print('pred_class: ', pred_class, 'mask_num: ', mask.sum())
            pred_mask_count += mask.sum()

            # The contribution of the current mask to the final panoramic segmentation mask must be greater than the threshold
            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < overlap_threshold:
                    continue

                # merge stuff regions
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )
        # print('pred_mask_count: ', pred_mask_count, '/', mask_pred.shape[1])

        return [panoptic_seg, segments_info]

def instance_inference(mask_cls,
                       mask_pred,
                       num_classes=20,
                       panoptic_on=True,
                       thing_id=list(range(3, 21)),
                       ):
    # mask_pred is already processed to have the same shape as original input
    num_queries, l = mask_pred.shape[0], mask_pred.shape[1]
    test_topk_per_volume = int(num_queries / 2)  # Only consider a specified number of instances for processing

    # [Q, K]
    scores = F.softmax(mask_cls, dim=-1)[:, 1:]  # K+1 discard the first empty category
    labels = torch.arange(start=1, end=num_classes+1, step=1, device=mask_pred.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

    scores_per_volume, topk_indices = scores.flatten(0, 1).topk(test_topk_per_volume, sorted=False)  # A mask may correspond to multiple categories
    labels_per_volume = labels[topk_indices]

    topk_indices = topk_indices // num_classes
    # mask_pred = mask_pred.unsqueeze(1).repeat(1, num_classes, 1).flatten(0, 1)
    mask_pred = mask_pred[topk_indices]

    # if this is panoptic segmentation, we only keep the "thing" classes
    if panoptic_on:
        keep = torch.zeros_like(scores_per_volume).bool()
        for i, lab in enumerate(labels_per_volume):
            keep[i] = lab in thing_id

        scores_per_volume = scores_per_volume[keep]
        labels_per_volume = labels_per_volume[keep]
        mask_pred = mask_pred[keep]

    result = {}
    # mask (before sigmoid)
    result["pred_masks"] = (mask_pred > 0).float()
    result["pred_boxes"] = torch.zeros(mask_pred.size(0), 4)
    # Uncomment the following to get boxes from masks (this is slow)
    # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

    # calculate average mask prob
    mask_scores_per_volume = (mask_pred.sigmoid() * result["pred_masks"]).sum(1) / (result["pred_masks"].sum(1) + 1e-6)  # [test_topk_per_volume,]
    result["scores"] = scores_per_volume * mask_scores_per_volume
    result["pred_classes"] = labels_per_volume
    return result


def prepare_targets(targets, images):
    h_pad, w_pad = images.tensor.shape[-2:]
    new_targets = []
    for targets_per_image in targets:
        # pad gt
        gt_masks = targets_per_image.gt_masks
        padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
        padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
        new_targets.append(
            {
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            }
        )
    return new_targets



















