import logging

import torch
import torch.nn.functional as F
from torch import nn


def compute_pos_weight(targets):
    """
    targets: [N,] bool
    """

    n_all = targets.reshape(-1).shape[0]
    n_p = targets.reshape(-1).sum()

    pos_weight = (n_all - n_p).float() / n_p

    return pos_weight

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks  # loss per bs per mask

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = []
    for b in range(targets.shape[0]):
        pos_weight = compute_pos_weight(targets[b])
        pos_weight = torch.clamp(pos_weight, max=30)
        loss.append(F.binary_cross_entropy_with_logits(inputs[b].view(-1, 1), targets[b].view(-1, 1), pos_weight=pos_weight, reduction="none").mean())
        #  loss.append(F.binary_cross_entropy_with_logits(inputs[b].view(-1, 1), targets[b].view(-1, 1), reduction="none").mean())

    loss = sum(loss) / len(loss)  # loss per bs per voxel

    return loss


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.valid_classes = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.true_classes = torch.arange(1, 21)

        # # pointwise mask loss parameters
        # self.num_points = num_points
        # self.oversample_ratio = oversample_ratio
        # self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if len(target_classes_o) == 0:
            losses = {"loss_ce": torch.Tensor([0.0]).cuda()[0] * src_logits.sum()}

        else:
            target_classes = torch.full(
                src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
            )
            target_classes[idx] = target_classes_o

            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
            losses = {"loss_ce": loss_ce}

        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]

        target_masks = masks[0].unsqueeze(0)
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        if len(src_masks) == 0 or len(target_masks) == 0:
            losses = {
                "loss_masks": torch.Tensor([0.0]).cuda()[0] * src_masks.sum(),
                "loss_dice": torch.Tensor([0.0]).cuda()[0] * src_masks.sum(),
            }

        else:
            losses = {
                "loss_mask": sigmoid_ce_loss(src_masks, target_masks, num_masks),  # loss per voxel
                "loss_dice": dice_loss(src_masks, target_masks, num_masks),
            }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        device = outputs['pred_masks'].device
        self.valid_classes = self.valid_classes.to(device)

        # Only 20 classes remain
        # 1. Delete the masks that are not in the 20 classes
        valid_mask_in_20 = torch.zeros_like(targets[0]['masks'][0], dtype=torch.bool)
        is_in_valid_classes = torch.zeros_like(targets[0]['labels'], dtype=torch.bool)  # False
        for idx, cls in enumerate(targets[0]['labels']):
            if cls in self.valid_classes:
                is_in_valid_classes[idx] = True
                valid_mask_in_20 = valid_mask_in_20 | targets[0]['masks'][idx]

        targets[0]['labels'] = targets[0]['labels'][is_in_valid_classes]
        targets[0]['masks'] = targets[0]['masks'][is_in_valid_classes][:, valid_mask_in_20]
        count3_mask = targets[0]['masks'].sum(dim=1)
        # 2. Mapping semantic categories to 20 classes
        for idx, cls in enumerate(targets[0]['labels']):
            idx_20 = torch.where(self.valid_classes == cls)[0]
            targets[0]['labels'][idx] = idx_20 + 1

        if targets[0]['masks'].sum() == 0:
            return {"loss_dice": torch.Tensor([0.0]).cuda()[0] * outputs['pred_logits'].sum()}

        outputs['pred_masks'] = outputs['pred_masks'][..., valid_mask_in_20]
        if "aux_outputs" in outputs:
            for aux_output in outputs["aux_outputs"]:
                # aux_output['pred_masks'] = aux_output['pred_masks'][..., valid_voxel_index][..., valid_mask_in_20]
                aux_output['pred_masks'] = aux_output['pred_masks'][..., valid_mask_in_20]
        
        # Remove classes with <100 voxels to reduce noise
        valid_mask_in_20 = torch.zeros_like(targets[0]['masks'][0], dtype=torch.bool)
        is_in_valid_classes = torch.zeros_like(targets[0]['labels'], dtype=torch.bool)
        for idx in range(len(targets[0]['labels'])):
            if targets[0]['masks'][idx].sum() > 100:
                is_in_valid_classes[idx] = True
                valid_mask_in_20 = valid_mask_in_20 | targets[0]['masks'][idx]

        targets[0]['labels'] = targets[0]['labels'][is_in_valid_classes]
        targets[0]['masks'] = targets[0]['masks'][is_in_valid_classes][:, valid_mask_in_20]
        count4_mask = targets[0]['masks'].sum(dim=1)

        if targets[0]['masks'].sum() == 0:
            return {"loss_dice": torch.Tensor([0.0]).cuda()[0] * outputs['pred_logits'].sum()}

        outputs['pred_masks'] = outputs['pred_masks'][..., valid_mask_in_20]
        if "aux_outputs" in outputs:
            for aux_output in outputs["aux_outputs"]:
                aux_output['pred_masks'] = aux_output['pred_masks'][..., valid_mask_in_20]

        # start compute loss
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # print('pano_targets: ', targets[0]['labels'], len(targets[0]['labels']), 'pano_masks: ', targets[0]['masks'].shape[1])
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=device
        )

        num_masks = torch.clamp(num_masks, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))  # loss per bs per query/mask

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            # "num_points: {}".format(self.num_points),
            # "oversample_ratio: {}".format(self.oversample_ratio),
            # "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
