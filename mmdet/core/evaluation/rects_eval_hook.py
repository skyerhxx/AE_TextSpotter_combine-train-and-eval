import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import collate, scatter
from mmcv.runner import Hook
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from mmdet import datasets
from .coco_utils import fast_eval_recall, results2json
from .mean_ap import eval_map
import editdistance
from collections import defaultdict
from collections import OrderedDict
import Polygon as plg


def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups


def get_union(pa, pb):
    pa_area = pa.area()
    pb_area = pb.area()
    return pa_area + pb_area - get_intersection(pa, pb)


def get_intersection(pa, pb):
    pInt = pa & pb
    if len(pInt) == 0:
        return 0
    else:
        return pInt.area()


def cat_best_hmean(gt, predictions, thresholds):
    num_gts = len([g for g in gt if g['ignore'] == False])
    image_gts = group_by_key(gt, 'name')
    image_gt_boxes = {k: np.array([b['bbox'] for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_trans = {k: np.array([b['trans'] for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_ignored = {k: np.array([b['ignore'] for b in boxes])
                        for k, boxes in image_gts.items()}
    image_gt_checked = {k: np.zeros((len(boxes), len(thresholds)))
                        for k, boxes in image_gts.items()}
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    ned = np.zeros((nd, len(thresholds)))
    for i, p in enumerate(predictions):
        pred_polygon = plg.Polygon(np.array(p['bbox']).reshape(-1, 2))
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p['name']]
            gt_ignored = image_gt_ignored[p['name']]
            gt_checked = image_gt_checked[p['name']]
            gt_trans = image_gt_trans[p['name']]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            ovmax = 0
            jmax = 0
            for j, gt_box in enumerate(gt_boxes):
                gt_polygon = plg.Polygon(np.array(gt_box).reshape(-1, 2))
                union = get_union(pred_polygon, gt_polygon)
                inter = get_intersection(pred_polygon, gt_polygon)
                overlap = inter / (union + 1e-6)
                if overlap > ovmax:
                    ovmax = overlap
                    jmax = j

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    if gt_ignored[jmax]:
                        tp[i, t] = 0.
                        ned[i, t] = 0
                    else:
                        tp[i, t] = 1.
                        ned[i, t] = 1 - editdistance.eval(p['trans'], gt_trans[jmax]) / \
                                    max(len(p['trans']), len(gt_trans[jmax]))
                    gt_checked[jmax, t] = 1
                else:
                    fp[i, t] = 1.
                    ned[i, t] = 0
            else:
                fp[i, t] = 1.

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    ned = np.cumsum(ned, axis=0) / (fp + num_gts + np.finfo(np.float64).eps)

    recalls = tp / float(num_gts)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    fmeasures = 2 * precisions * recalls / (precisions + recalls + 1e-6)
    # best_i = np.argmax(fmeasures)
    # print('[Best F-Measure] p: {:.2f}, r: {:.2f}, f: {:.2f}, 1-ned: {:.2f}, best_score_th: {:.3f}'.format(
    #     float(precisions[best_i]) * 100, float(recalls[best_i]) * 100, float(fmeasures[best_i]) * 100,
    #     float(ned[best_i]) * 100, predictions[best_i]['score']))

    best_i = np.argmax(ned)
    print('[Best 1-NED]     p: {:.2f}, r: {:.2f}, f: {:.2f}, 1-ned: {:.2f}, best_score_th: {:.3f}'.format(
        float(precisions[best_i]) * 100, float(recalls[best_i]) * 100, float(fmeasures[best_i]) * 100,
        float(ned[best_i]) * 100, predictions[best_i]['score']))
    return float(ned[best_i]) * 100

def trans_pred_format(pred):
    bdd = []
    img_name = pred['img_name']
    ponits = pred['points']
    scores = pred['scores']
    texts = pred['texts']
    for i in range(len(ponits)):
        bdd_i = {
            'category': 'text',
            'timestamp': 1000,
            'name': img_name,
            'bbox': np.array(ponits[i]).reshape(-1).tolist(),
            'score': scores[i],
            'trans': texts[i]
        }
        bdd.append(bdd_i)
    return bdd


def trans_gt_format(img_name, gt):
    bdd = []
    for i in range(len(gt)):
        bdd_i = {
            'category': 'text',
            'timestamp': 1000,
            'name': img_name,
            'bbox': gt[i]['points'],
            'score': 1,
            'ignore': gt[i]['ignore'],
            'trans': gt[i]['transcription']
        }
        bdd.append(bdd_i)
    return bdd

class DistEvalHook(Hook):
    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset, {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError


class ReCTSEvalHook(DistEvalHook):

    def evaluate(self, runner, results):
        new_results = []
        for idx, resu in enumerate(results):
            rects, scores, char_bbox_results, texts = resu
           
            with_char = True 
            if with_char:
                char_rects = []
                char_scores = []
                chars = []
                char_bboxes = mmcv.concat_list(char_bbox_results)
                char_labels = np.concatenate([
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(char_bbox_results)
                ])
                for char_bbox, char_label in zip(char_bboxes, char_labels):
                    char_bbox = [float(x) for x in char_bbox]
                    char_rect = [char_bbox[0], char_bbox[1],
                                char_bbox[0], char_bbox[3],
                                char_bbox[2], char_bbox[3],
                                char_bbox[2], char_bbox[1]]
                    char_rects.append(char_rect)
                    char_scores.append(char_bbox[-1])
                    chars.append(self.dataset.label2char[char_label])

            result_i = {
                'img_name': self.dataset.get_img_info(idx)['filename'],
                'points': rects,
                'scores': scores
            }

            if len(resu) == 4:
                result_i['texts'] = texts

            if with_char:
                result_i['chars'] = {
                    'points': char_rects,
                    'scores': char_scores,
                    'chars': chars
                }

            new_results.append(result_i)
        
        # import pdb; pdb.set_trace()
        preds_bdd = []
        gts_bdd = []
        for idx,pred in enumerate(new_results):
            img_name = self.dataset.get_img_info(idx)['filename']
            pred_bdd = trans_pred_format(pred)
            preds_bdd.extend(pred_bdd)

            gt_path = self.dataset.ann_root + img_name.replace('.jpg', '.json')
            gt = mmcv.load(gt_path)['lines']
            gt_bdd = trans_gt_format(img_name, gt)
            gts_bdd.extend(gt_bdd)

        cat_gt = group_by_key(gts_bdd, 'category')
        cat_pred = group_by_key(preds_bdd, 'category')
        thresholds = [0.5]
        ned = cat_best_hmean(cat_gt['text'], cat_pred['text'], thresholds)

        # gt_bboxes = []
        # gt_labels = []
        # gt_ignore = []

        # #self.dataset就是测试集
        # for i in range(len(self.dataset)):
        #     ann = self.dataset.get_ann_info(i)
        #     bboxes = ann['bboxes']
        #     labels = ann['labels']
        #     if 'bboxes_ignore' in ann:
        #         ignore = np.concatenate([
        #             np.zeros(bboxes.shape[0], dtype=np.bool),
        #             np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
        #         ])
        #         gt_ignore.append(ignore)
        #         bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
        #         labels = np.concatenate([labels, ann['labels_ignore']])
        #     gt_bboxes.append(bboxes)
        #     gt_labels.append(labels)
        # if not gt_ignore:
        #     gt_ignore = None
        # # If the dataset is VOC2007, then use 11 points mAP evaluation.
        # if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
        #     ds_name = 'voc07'
        # else:
        #     ds_name = self.dataset.CLASSES
        # mean_ap, eval_results = eval_map(
        #     results,
        #     gt_bboxes,
        #     gt_labels,
        #     gt_ignore=gt_ignore,
        #     scale_ranges=None,
        #     iou_thr=0.5,
        #     dataset=ds_name,
        #     print_summary=True)
        runner.log_buffer.output['1-ned'] = ned
        runner.log_buffer.ready = True

