import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
from datasets.hico_eval import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]
        if args.use_place365_pred_hier2:
            background = torch.zeros((len(targets),16)).to(device)
            for i in range(len(targets)):
                background[i] = targets[i]['use_place365_pred_hier2d']
        elif args.use_place365_pred_hier2reclass:
            background = torch.zeros((len(targets),33)).to(device)
            for i in range(len(targets)):
                background[i] = targets[i]['use_place365_pred_hier2_reclass']
        elif args.use_place365_pred_hier3:
            background = torch.zeros((len(targets),512)).to(device)
            for i in range(len(targets)):
                background[i] = targets[i]['use_place365_pred_hier3d']
        else:
            background = None
        if args.use_coco_panoptic_info or args.use_panoptic_info_attention:
            panoptic_info =  torch.zeros((len(targets),133)).to(device)
            for i in range(len(targets)):
                panoptic_info[i] = targets[i]['panoptic_class_info']
            if args.use_coco_panoptic_num_info:
                panoptic_num_info = torch.zeros((len(targets),133)).to(device)
                for i in range(len(targets)):
                    panoptic_num_info[i] = targets[i]['panoptic_class_num_info']
                background = [background,panoptic_info,panoptic_num_info]
            else:
                background = [background,panoptic_info]
        # if args.amp:     
        #   with autocast():
        #     outputs = model(samples, background)
        # else:
        outputs = model(samples, background)
        #print(targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        # if args.amp:
        #     #print("use amp")conda deactivate
        #     scalar.scale(losses).backward()
        #     scalar.step(optimizer)
        #     scalar.update()
        # else:
        losses.backward()
        optimizer.step()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        if args.use_place365_pred_hier2:
            background = torch.zeros((len(targets),16)).to(device)
            for i in range(len(targets)):
              background[i] = targets[i]['use_place365_pred_hier2d']
        elif args.use_place365_pred_hier3:# problem
            background = torch.zeros((len(targets),512)).to(device)
            for i in range(len(targets)):
              background[i] = targets[i]['use_place365_pred_hier3d']
        elif args.use_place365_pred_hier2reclass:
            background = torch.zeros((len(targets),33)).to(device)
            for i in range(len(targets)):
                background[i] = targets[i]['use_place365_pred_hier2_reclass']
        else:
            background = None
        if args.use_coco_panoptic_info or args.use_panoptic_info_attention:
            panoptic_info =  torch.zeros((len(targets),133)).to(device)
            for i in range(len(targets)):
              panoptic_info[i] = targets[i]['panoptic_class_info']
            if args.use_coco_panoptic_num_info:
              panoptic_num_info = torch.zeros((len(targets),133)).to(device)
              for i in range(len(targets)):
                panoptic_num_info[i] = targets[i]['panoptic_class_num_info']
              background = [background,panoptic_info,panoptic_num_info]
            else:
              background = [background,panoptic_info]
        outputs = model(samples, background)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if args.only_use_mask:
            results = postprocessors['hoi'](outputs, orig_target_sizes,background)
        elif len(args.mask_verb_scene_coour)!=0 or args.use_place365_pred_hier2:
            background = torch.zeros((len(targets),16)).to(device)
            for i in range(len(targets)):
              background[i] = targets[i]['use_place365_pred_hier2d']
            results = postprocessors['hoi'](outputs, orig_target_sizes, background)
        else:
            results = postprocessors['hoi'](outputs, orig_target_sizes)
        # results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))


    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)

    stats = evaluator.evaluate()

    return stats