from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import torch
import torch.utils.data
import torchvision

import datasets.transforms as T


class HICODetection(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries, args=None):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries

        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = list(range(1, 118))

        if img_set == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(img_anno['annotations']):
                        break
                else:
                    self.ids.append(idx)
        else:
            self.ids = list(range(len(self.annotations)))
        
        self.use_background = args.use_background
        self.use_place365_pred_hier2 = args.use_place365_pred_hier2
        self.use_place365_pred_hier3 = args.use_place365_pred_hier3
        self.mask_verb_scene_coour = args.mask_verb_scene_coour
        self.use_coco_panoptic_info = args.use_coco_panoptic_info
        self.use_place365_pred_hier2reclass = args.use_place365_pred_hier2reclass
        if self.use_coco_panoptic_info:
            self.use_num_aswell = args.use_coco_panoptic_num_info

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.img_set == 'train':
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.use_place365_pred_hier2 :
            target['use_place365_pred_hier2d'] =  torch.tensor(img_anno['hier2_pred'])
        if self.use_place365_pred_hier2reclass :
            target['use_place365_pred_hier2_reclass'] =  torch.tensor(img_anno['use_place365_pred_hier2reclass'])
        if self.use_coco_panoptic_info:
            target['panoptic_class_info'] = torch.tensor(img_anno['panoptic_info'][0])
            if self.use_num_aswell:
                target['panoptic_class_num_info'] = torch.tensor(img_anno['panoptic_info'][1])
        elif self.use_place365_pred_hier2 or self.mask_verb_scene_coour:
            target['use_place365_pred_hier2d'] =  torch.tensor(img_anno['hier2_pred'])
        
        if self.use_background:
            target['background'] =  torch.tensor(1.0) if img_anno['background']=='shop or stall' else torch.tensor(0.0)
            target['background'] = torch.as_tensor(target['background'])        
        
        if self.use_place365_pred_hier3:
            target['use_place365_pred_hier3d'] =  torch.tensor(img_anno['hier3_pred'])
            hier3 = torch.tensor(img_anno['hier3_pred'])
            zeros = torch.zeros((512-365))
            hier3 = torch.cat((hier3,zeros),0)
            target['use_place365_pred_hier3d'] =  hier3
        
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)

            target['filename'] = img_anno['file_name']
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['matching_labels'] = torch.zeros((0,), dtype=torch.int64)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
                target['matching_labels'] = torch.ones_like(target['obj_labels'])
        else:
            target['filename'] = img_anno['file_name']
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

        return img, target

    def set_rare_hois(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        counts = defaultdict(lambda: 0)
        for img_anno in annotations:
            hois = img_anno['hoi_annotation']
            bboxes = img_anno['annotations']
            for hoi in hois:
                triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                           self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                           self._valid_verb_ids.index(hoi['category_id']))
                counts[triplet] += 1
        self.rare_triplets = []
        self.non_rare_triplets = []
        for triplet, count in counts.items():
            if count < 10:
                self.rare_triplets.append(triplet)
            else:
                self.non_rare_triplets.append(triplet)

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


def make_hico_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'

    if args.use_place365_pred_hier2 and (args.use_coco_panoptic_info or args.use_panoptic_info_attention):
        print("using all data with hier2 and panoptic information")
        PATHS = {
            'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico_hier2pred_panoptic.json'),
            'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico_hier2pred_panoptic.json')
        }
    
    elif args.use_place365_pred_hier2 or args.mask_verb_scene_coour:
        print("using all data with place365_predicted hier2")
        PATHS = {
            'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico_hier2pred.json'),
            'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico_hier2pred.json')
        }
    elif args.use_place365_pred_hier2reclass:
        print("using all data with hier2 reclassed data")
        PATHS = {
            'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico_hier2reclass_pred.json'),
            'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico_hier2reclass_pred.json')
        }
    elif args.use_place365_pred_hier3:
        print("using all data with place365_predicted hier3")
        PATHS = {
            'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico_hier3_new.json'),
            'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico_hier3_new.json')
        }
    elif args.use_all_data and args.use_background:
        print("using all data with background")
        PATHS = {
            'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico_new.json'),
            'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico_new.json')
        }
    elif args.use_all_data:
        PATHS = {
            'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico.json'),
            'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico.json')
        }
    else:
        print("using part data")
        PATHS = {
            'train': (root / 'images' / 'train2015', root / 'annotations' / 'data_buy.json'),
            'val': (root / 'images' / 'test2015', root / 'annotations' / 'data_buy_test.json')
        }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = HICODetection(image_set, img_folder, anno_file, transforms=make_hico_transforms(image_set),num_queries=args.num_queries,args = args)
    if image_set == 'val':
        dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
