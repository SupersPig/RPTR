# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import torch
import torch.nn.functional as F
from torch import nn
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .matcher import build_matcher
from .criterion import SetCriterion, PostProcess
from .transformer import build_transformer

class SGG(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_rel_classes, num_entities, num_triplets, aux_loss=False, matcher=None):
        super().__init__()
        self.num_entities = num_entities
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.entity_embed = nn.Embedding(num_entities, hidden_dim*2)
        self.triplet_embed = nn.Embedding(num_triplets, hidden_dim*3)
        self.so_embed = nn.Embedding(2, hidden_dim) # subject and object encoding

        # entity prediction
        self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # subject/object label classfication and box regression
        self.sub_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # predicate classification
        self.rel_projection = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.normal_(self.rel_projection, std=hidden_dim ** -0.5)

        self.spatial_dk = True
        if self.spatial_dk:
            self.rel_class_embed = MLP(hidden_dim * 2, hidden_dim, num_rel_classes + 1, 2)
            self.spatial_embed = MLP(14, hidden_dim, hidden_dim, 2)
        else:
            self.rel_class_embed = MLP(hidden_dim + 20, hidden_dim, num_rel_classes + 1, 2)
            self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, samples: NestedTensor):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, hs_t, _ = self.transformer(self.input_proj(src), mask, self.entity_embed.weight,
                                                 self.triplet_embed.weight, pos[-1], self.so_embed.weight)

        hs_sub, hs_obj = torch.split(hs_t, self.hidden_dim, dim=-1)
        # 解码实体类别和框
        outputs_class = self.entity_class_embed(hs)
        outputs_coord = self.entity_bbox_embed(hs).sigmoid()
        # 解码主语类别和框
        outputs_class_sub = self.sub_class_embed(hs_sub)
        outputs_coord_sub = self.sub_bbox_embed(hs_sub).sigmoid()
        # 解码宾语类别和框
        outputs_class_obj = self.obj_class_embed(hs_obj)
        outputs_coord_obj = self.obj_bbox_embed(hs_obj).sigmoid()

        hs_pred = hs_sub - hs_obj
        hs_pred = hs_pred @ self.rel_projection

        if self.spatial_dk:
            hs_s = self.calc_spatial_features_dk(outputs_coord_sub, outputs_coord_obj)
        else:
            hs_s = self.calc_spatial_features(outputs_coord_sub, outputs_coord_obj)

        outputs_class_rel = self.rel_class_embed(torch.cat((hs_pred, hs_s), dim=-1))

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
               'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
               'rel_logits': outputs_class_rel[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                    outputs_class_obj, outputs_coord_obj, outputs_class_rel)
        return out

    def calc_spatial_features_dk(self, sub_boxes, obj_boxes):
        """
        计算主体和客体之间的空间关系特征
        输入:
            sub_boxes: [batch_size, num_queries, 4] (cx, cy, w, h) 格式的主体框
            obj_boxes: [batch_size, num_queries, 4] (cx, cy, w, h) 格式的客体框
        返回:
            spatial_feats: [batch_size, num_queries, hidden_dim] 空间关系特征
        """
        # 解构框坐标
        s_cx, s_cy, s_w, s_h = sub_boxes.unbind(-1)
        o_cx, o_cy, o_w, o_h = obj_boxes.unbind(-1)

        # 1. 基本相对位置特征
        rel_cx = (o_cx - s_cx) / (s_w + 1e-6)  # x方向相对位置(相对于主体宽度)
        rel_cy = (o_cy - s_cy) / (s_h + 1e-6)  # y方向相对位置(相对于主体高度)
        rel_wh = torch.log((o_w * o_h) / (s_w * s_h + 1e-6) + 1e-6)  # 面积比的对数

        # 2. 距离特征
        dx = o_cx - s_cx
        dy = o_cy - s_cy
        distance = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        norm_distance = distance / torch.sqrt(s_w ** 2 + s_h ** 2 + 1e-6)  # 归一化距离

        # 3. 方向特征(角度)
        angle = torch.atan2(dy, dx + 1e-6)  # [-pi, pi]
        angle_sin = torch.sin(angle)
        angle_cos = torch.cos(angle)

        # 4. IoU和覆盖特征
        # 计算交集
        s_x1, s_y1 = s_cx - s_w / 2, s_cy - s_h / 2
        s_x2, s_y2 = s_cx + s_w / 2, s_cy + s_h / 2
        o_x1, o_y1 = o_cx - o_w / 2, o_cy - o_h / 2
        o_x2, o_y2 = o_cx + o_w / 2, o_cy + o_h / 2

        inter_x1 = torch.max(s_x1, o_x1)
        inter_y1 = torch.max(s_y1, o_y1)
        inter_x2 = torch.min(s_x2, o_x2)
        inter_y2 = torch.min(s_y2, o_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        s_area = s_w * s_h
        o_area = o_w * o_h

        iou = inter_area / (s_area + o_area - inter_area + 1e-6)
        s_cover = inter_area / (s_area + 1e-6)  # 主体被覆盖的比例
        o_cover = inter_area / (o_area + 1e-6)  # 客体被覆盖的比例

        # 5. 相对大小特征
        rel_w = torch.log(o_w / (s_w + 1e-6) + 1e-6)
        rel_h = torch.log(o_h / (s_h + 1e-6) + 1e-6)

        # 组合所有空间特征
        raw_spatial_feats = torch.stack([
            rel_cx, rel_cy, rel_wh,
            dx, dy, distance, norm_distance,
            angle_sin, angle_cos,
            iou, s_cover, o_cover,
            rel_w, rel_h
        ], dim=-1)  # [batch_size, num_queries, 14]

        # 通过MLP编码为高维特征
        spatial_feats = self.spatial_embed(raw_spatial_feats)  # [batch_size, num_queries, hidden_dim]

        return spatial_feats

    def calc_spatial_features(self, sub_boxes, obj_boxes):
        # 变成xtl, ytl, xbr, ybr的格式
        sub_boxes_ = torch.stack(((sub_boxes[:, :, :,  0] - 0.5 * sub_boxes[:, :, :,  2]),
                                  (sub_boxes[:, :, :,  1] - 0.5 * sub_boxes[:, :, :,  3]),
                                  (sub_boxes[:, :, :,  0] + 0.5 * sub_boxes[:, :, :,  2]),
                                  (sub_boxes[:, :, :,  1] + 0.5 * sub_boxes[:, :, :,  3])), dim=-1)

        obj_boxes_ = torch.stack(((obj_boxes[:, :, :,  0] - 0.5 * obj_boxes[:, :, :,  2]),
                                  (obj_boxes[:, :, :,  1] - 0.5 * obj_boxes[:, :, :,  3]),
                                  (obj_boxes[:, :, :,  0] + 0.5 * obj_boxes[:, :, :,  2]),
                                  (obj_boxes[:, :, :,  1] + 0.5 * obj_boxes[:, :, :,  3])), dim=-1)

        # 通过广播机制，生成 NxN 个目标对拼接形成的特征矩阵
        so_boxes = torch.cat((sub_boxes_, obj_boxes_), dim=-1)

        dx = (so_boxes[:, :, :, 0] + so_boxes[:, :, :, 2] - so_boxes[:, :, :, 4] + so_boxes[:, :, :, 6]).unsqueeze(-1)
        dy = (so_boxes[:, :, :, 1] + so_boxes[:, :, :, 3] - so_boxes[:, :, :, 5] - so_boxes[:, :, :, 7]).unsqueeze(-1)

        dis = self.pdist(dx, dy).unsqueeze(dim=-1)
        dx[dx == 0] = 1e-6
        thea = torch.arctan(torch.div(dy, dx))  # 注意，这一步可能存在分母为零

        # 并集
        U = torch.stack((torch.min(so_boxes[:, :, :, 0], so_boxes[:, :, :, 4]),
                         torch.min(so_boxes[:, :, :, 1], so_boxes[:, :, :, 5]),
                         torch.max(so_boxes[:, :, :, 2], so_boxes[:, :, :, 6]),
                         torch.max(so_boxes[:, :, :, 3], so_boxes[:, :, :, 7])), dim=-1)

        # 并集
        I = torch.stack((torch.max(so_boxes[:, :, :, 0], so_boxes[:, :, :, 4]),
                         torch.max(so_boxes[:, :, :, 1], so_boxes[:, :, :, 5]),
                         torch.min(so_boxes[:, :, :, 2], so_boxes[:, :, :, 6]),
                         torch.min(so_boxes[:, :, :, 3], so_boxes[:, :, :, 7])), dim=-1)

        cond1 = I[..., 2] > I[..., 0]  # 第2个元素 > 第0个元素
        cond2 = I[..., 3] > I[..., 1]  # 第3个元素 > 第1个元素
        mask = cond1 | cond2  # 逻辑或合并条件
        mask_expanded = mask.unsqueeze(-1).expand_as(I)  # 形状变为 (2, 100, 100, 4)
        I_ = torch.where(mask_expanded, torch.zeros_like(I), I)

        Spatial_Feature = torch.cat((so_boxes, dx, dy, dis, thea, U, I_), dim=-1)

        return Spatial_Feature

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                      outputs_class_obj, outputs_coord_obj, outputs_class_rel):
        return [{'pred_logits': a, 'pred_boxes': b, 'sub_logits': c, 'sub_boxes': d, 'obj_logits': e, 'obj_boxes': f,
                 'rel_logits': g}
                for a, b, c, d, e, f, g in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_sub[:-1],
                                               outputs_coord_sub[:-1], outputs_class_obj[:-1], outputs_coord_obj[:-1],
                                               outputs_class_rel[:-1])]


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

def build(args):

    num_classes = 151 if args.dataset != 'oi' else 601 # some entity categories in OIV6 are deactivated.
    num_rel_classes = 51 if args.dataset != 'oi' else 30

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    matcher = build_matcher(args)
    model = SGG(
        backbone,
        transformer,
        num_classes=num_classes,
        num_rel_classes = num_rel_classes,
        num_entities=args.num_entities,
        num_triplets=args.num_triplets,
        aux_loss=args.aux_loss,
        matcher=matcher)

    weight_dict = {'loss_ce': 1,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,
                   'loss_rel': args.rel_loss_coef
                   }

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', "relations"]

    criterion = SetCriterion(num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

