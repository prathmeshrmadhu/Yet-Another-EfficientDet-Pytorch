# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/annotations/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os
import pandas as pd
import numpy as np

import argparse
import torch
import yaml, pdb
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=bool, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=bool, default=False)
ap.add_argument('--override', type=bool, default=False, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights

print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

# valid_df = pd.read_csv(args.valid_csv_path, header=None)
# columns = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
# valid_df.columns = columns
# unique_filenames = np.unique(valid_df['filename'])
# filename_id_dict = {val: i for i, val in enumerate(unique_filenames)}
# cat_id_dict = {val: i for i, val in enumerate(obj_list)}
# valid_df['file_id'] = valid_df['filename'].map(filename_id_dict)
# valid_df['cat_id'] = valid_df['class'].map(cat_id_dict)
# valid_df['id'] = range(len(valid_df))
# VAL_NAMES = valid_df['filename']
# VAL_PATHS = [os.path.join(args.data_path, i.split('latest/')[-1]) for i in VAL_NAMES]
# VAL_PATHS = list(np.unique(VAL_PATHS))[:MAX_IMAGES]

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)


def _eval(coco_gt, image_paths, pred_json_path):
     # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def EFI_COCO(json_path, obj_list):
     # loading annotations
    with open(json_path) as file:
        annotations = json.load(file)
    
    processed_instances = annotations["annotations"]
    n_instances = len(processed_instances)

    for inst in processed_instances:
        xmin, ymin, xmax, ymax = [int(c) for c in inst["bbox"].split(",")]
        coords = [xmin, ymin, xmax - xmin, ymax - ymin]  # converting to x,y,w,h
        inst["bbox"] = coords
    
    
    train_categories = []
    for i, val in enumerate(obj_list):
        one_dict = {}
        one_dict['id'] = i + 1
        one_dict['name'] = val
        train_categories.append(one_dict)
    
    fit_annotations = {
        "annotations": processed_instances,
        "images": annotations["images"],
        "categories": train_categories
    }
    
    # intiializing COCO dataset and fitting the annotations
    coco_dataset = COCO()
    coco_dataset.dataset = fit_annotations
    coco_dataset.createIndex()
    
    return coco_dataset

if __name__ == '__main__':

    
    if 'arthist' in args.project:
        SET_NAME = 'efi_arthist_trainset'
        VAL_GT = 'resources_coco/efi_arthist_train.json'
        VAL_IMGS = '/localhome/prathmeshmadhu/work/EFI/Data/Art_history/latest/'
    else:
        SET_NAME = 'efi_classarch_trainset'
        VAL_GT = 'resources_coco/efi_classarch_valid.json'
        VAL_IMGS = '/localhome/prathmeshmadhu/work/EFI/Data/Classical_Arch/latest/'
    MAX_IMAGES = 100000
        
    
    coco_gt = EFI_COCO(VAL_GT, params['obj_list'])
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    
    if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model.cuda(gpu)

            if use_float16:
                model.half()

        evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model)

    _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')
    
#     SET_NAME = 'efi_arthist_valset'
#     efi_coco_gt = EFI_COCO(obj_list, args.valid_csv_path)
#     image_ids = list(np.unique(valid_df['file_id']))[:MAX_IMAGES]
#     if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
#         model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
#                                      ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
#         model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
#         model.requires_grad_(False)
#         model.eval()

#         if use_cuda:
#             model.cuda(gpu)

#             if use_float16:
#                 model.half()

#         evaluate_coco(args.data_path, SET_NAME, image_ids, efi_coco_gt, model)

#     _eval(efi_coco_gt, VAL_PATHS, f'{SET_NAME}_bbox_results.json')
