"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import numpy as np
from functools import reduce
import math
from lib.pytorch_misc import intersect_2d, argsort_desc
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
np.set_printoptions(precision=3)

class BasicSceneGraphEvaluator:
    def __init__(self, mode, multiple_preds=False):
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {20: [], 50: [], 100: []}
        self.multiple_preds = multiple_preds

    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in ('sgdet', 'sgcls', 'predcls', 'so_pair')}
        return evaluators

    @classmethod
    def vrd_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, multiple_preds=True, **kwargs) for m in ('preddet', 'phrdet')}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_scores, viz_dict=None, iou_thresh=0.5):
        res = evaluate_from_dict(gt_entry, pred_scores, self.mode, self.result_dict,
                                  viz_dict=viz_dict, iou_thresh=iou_thresh, multiple_preds=self.multiple_preds)
        # self.print_stats()
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        output = {}
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))
            output['R@%i' % k] = np.mean(v)
        output['mode'] = self.mode
        return output

def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, multiple_preds=False,
                       viz_dict=None, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param mode: 'det' or 'cls'
    :param result_dict: 
    :param viz_dict: 
    :param kwargs: 
    :return: 
    """
    gt_rels = gt_entry['gt_relations']

    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    rel_scores = pred_entry['rel_scores']
    # 为什么要用‘1+’呢
    pred_rels = 1+rel_scores.argmax(1)
    predicate_scores = rel_scores.max(1)

    sub_boxes = pred_entry['sub_boxes']
    obj_boxes = pred_entry['obj_boxes']
    sub_score = pred_entry['sub_scores']
    obj_score = pred_entry['obj_scores']
    sub_class = pred_entry['sub_classes']
    obj_class = pred_entry['obj_classes']
    # pred_to_gt可能是预测值与真值对应的索引
    pred_to_gt, _, rel_scores = evaluate_recall(gt_rels, gt_boxes, gt_classes, mode,
                pred_rels, sub_boxes, obj_boxes, sub_score, obj_score, predicate_scores, sub_class, obj_class, phrdet= mode=='phrdet',
                **kwargs)

    for k in result_dict[mode + '_recall']:

        match = reduce(np.union1d, pred_to_gt[:k])
        # match 为匹配结果，rec_i就是R@i的召回率值
        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)
    return pred_to_gt, _, rel_scores


def evaluate_recall(gt_rels, gt_boxes, gt_classes, mode,
                    pred_rels, sub_boxes, obj_boxes, sub_score, obj_score, predicate_scores, sub_class, obj_class,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0, 5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0
    # 转换一下格式，变成<class, relation, class>这种格式，和对应的8维的boxes
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)


    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])

    pred_triplets = np.column_stack((sub_class, pred_rels, obj_class))
    pred_triplet_boxes = np.column_stack((sub_boxes, obj_boxes))
    relation_scores = np.column_stack((sub_score, obj_score, predicate_scores))  #TODO!!!! do not * 0.1 finally
    # relation_scores = np.column_stack((sub_score, obj_score))

    # prod(1) 将三元组的三个部分预测分数相乘，获取预测质量分数
    sorted_scores = relation_scores.prod(1)
    # argsort() 用于返回数组从小到大的索引，[::-1] 用于反转这个序列
    pred_triplets = pred_triplets[sorted_scores.argsort()[::-1], :]
    pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1], :]
    relation_scores = relation_scores[sorted_scores.argsort()[::-1], :]
    scores_overall = relation_scores.prod(1)


    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    if mode == 'sgdet':
        pred_to_gt = _compute_pred_matches(gt_triplets, pred_triplets,
                                           gt_triplet_boxes, pred_triplet_boxes,
                                           iou_thresh, phrdet=phrdet)
    elif mode == 'sgcls':
        pred_to_gt = _compute_pred_matches_sgcls(gt_triplets, pred_triplets,
                                           gt_triplet_boxes, pred_triplet_boxes,
                                           iou_thresh, phrdet=phrdet)
    elif mode == 'predcls':
        pred_to_gt = _compute_pred_matches_predcls(gt_triplets, pred_triplets,
                                                 gt_triplet_boxes, pred_triplet_boxes,
                                                 iou_thresh, phrdet=phrdet)
    elif mode == 'so_pair':
        pred_to_gt = _compute_pred_matches_so_pair(gt_triplets, pred_triplets,
                                                   gt_triplet_boxes, pred_triplet_boxes,
                                                   iou_thresh, phrdet=phrdet)

    return pred_to_gt, None, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    给定一组预测的三元组，返回每个给定预测的匹配GT列表
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: 
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds] # 提取标签匹配正确的预测框
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]
            # 框也匹配正确的索引
            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]: # 标签对，框也对，按顺序找索引
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt

def _compute_pred_matches_sgcls(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    so_iou = [[] for x in range(pred_boxes.shape[0])]
    pred_to_gt_ = [[] for x in range(pred_boxes.shape[0])]
    for i, (pred_s, pred_p, pred_o) in enumerate(pred_triplets):
        pred_s_bbox, pred_o_bbox = pred_boxes[i, :4], pred_boxes[i, 4:]
        iou = 0
        for j, (gt_s, gt_p, gt_o) in enumerate(gt_triplets):
            gt_s_bbox, gt_o_bbox = gt_boxes[j, :4], gt_boxes[j, 4:]
            gt_s_class, gt_o_class = gt_triplets[j, 0], gt_triplets[j, 2]

            s_iou = bbox_overlaps(pred_s_bbox[None, :4], gt_s_bbox[None, :4])[0]
            o_iou = bbox_overlaps(pred_o_bbox[None, :4], gt_o_bbox[None, :4])[0]

            iou = max(iou, s_iou[0] * o_iou[0])

            if s_iou < iou_thresh or o_iou < iou_thresh:
                continue

            if (pred_s == gt_s_class) and (pred_p == gt_p) and (pred_o == gt_o_class):
                pred_to_gt[i].append(j)
        so_iou[i] = iou
    temp = np.argsort(np.array(so_iou))[::-1]
    for j, p in enumerate(pred_to_gt):
        pred_to_gt_[temp[j]].append(p)
    return pred_to_gt_

def _compute_pred_matches_predcls(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    so_iou = [[] for x in range(pred_boxes.shape[0])]
    pred_to_gt_2 = [[] for x in range(pred_boxes.shape[0])]
    for i, (pred_s, pred_p, pred_o) in enumerate(pred_triplets):
        pred_s_bbox, pred_o_bbox = pred_boxes[i, :4], pred_boxes[i, 4:]
        iou = 0
        for j, (gt_s, gt_p, gt_o) in enumerate(gt_triplets):
            gt_s_bbox, gt_o_bbox = gt_boxes[j, :4], gt_boxes[j, 4:]
            # gt_s_class, gt_o_class = gt_triplets[j, 0], gt_triplets[j, 2]

            s_iou = bbox_overlaps(pred_s_bbox[None, :4], gt_s_bbox[None, :4])[0]
            o_iou = bbox_overlaps(pred_o_bbox[None, :4], gt_o_bbox[None, :4])[0]

            iou = max(iou, s_iou[0] * o_iou[0])

            if s_iou < iou_thresh or o_iou < iou_thresh:
                continue

            if pred_p == gt_p: #(gt_s == gt_s_class) and (gt_o == gt_o_class) and (pred_p == gt_p):
                pred_to_gt[i].append(j)
        so_iou[i] = iou
    temp = np.argsort(np.array(so_iou))[::-1]
    for j, p in enumerate(pred_to_gt):
        pred_to_gt_2[temp[j]].append(p)

    return pred_to_gt_2

def _compute_pred_matches_so_pair(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    gt_triplets[:, 1] = gt_triplets[:, 0]
    pred_triplets[:, 1] = pred_triplets[:, 0]
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]  # 提取标签匹配正确的预测框

        sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
        obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]
        # 框也匹配正确的索引
        inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:  # 标签对，框也对，按顺序找索引
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


# def _compute_pred_matches_so_pair_l(gt_triplets, pred_triplets,
#                                   gt_boxes, pred_boxes, iou_thresh, phrdet=False):
#     gt_triplets[:, 1] = gt_triplets[:, 0]
#     pred_triplets[:, 1] = pred_triplets[:, 0]
#     keeps = intersect_2d(gt_triplets, pred_triplets)
#     gt_has_match = keeps.any(1)
#     pred_to_gt = [[] for _ in range(pred_boxes.shape[0])]
#     for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
#                                          gt_boxes[gt_has_match],
#                                          keeps[gt_has_match],
#                                          ):
#         for i in np.where(keep_inds)[0]: # 标签对，按顺序找索引
#             pred_to_gt[i].append(int(gt_ind))
#     return pred_to_gt


def calculate_mR_from_evaluator_list(evaluator_list, mode, multiple_preds=False, save_file=None):
    all_rel_results = {}
    for (pred_id, pred_name, evaluator_rel) in evaluator_list:
        print('\n')
        print('relationship: ', pred_name)
        rel_results = evaluator_rel[mode].print_stats()
        all_rel_results[pred_name] = rel_results

    mean_recall = {}
    mR20 = 0.0
    mR50 = 0.0
    mR100 = 0.0
    for key, value in all_rel_results.items():
        if math.isnan(value['R@100']):
            continue
        mR20 += value['R@20']
        mR50 += value['R@50']
        mR100 += value['R@100']
    rel_num = len(evaluator_list)
    mR20 /= rel_num
    mR50 /= rel_num
    mR100 /= rel_num
    mean_recall['R@20'] = mR20
    mean_recall['R@50'] = mR50
    mean_recall['R@100'] = mR100
    all_rel_results['mean_recall'] = mean_recall

    if multiple_preds:
        recall_mode = 'mean recall without constraint'
    else:
        recall_mode = 'mean recall with constraint'
    print('\n')
    print('======================' + mode + '  ' + recall_mode + '============================')
    print('mR@20: ', mR20)
    print('mR@50: ', mR50)
    print('mR@100: ', mR100)
    mean_recall['mode'] = mode
    mean_recall['recall_mode'] = recall_mode
    return mean_recall
