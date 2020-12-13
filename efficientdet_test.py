# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import os, pdb
import argparse
import sys

import pandas as pd
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np
import tqdm

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
parser.add_argument('--from_folder', type=bool, default=False, help='test from folder flag otherwise val/train csv file')
parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
parser.add_argument('--folder_path', type=str, help='path to specific folder for testing')
parser.add_argument('--val_csv', '-val_csv', help='path to validation csv')
parser.add_argument('--model_path', '-model_path', required=True, help='path to trained model')
parser.add_argument('-p', '--project', required=True, help='project to be chosen from ["arthist","classarch", "chrisarch"]')
parser.add_argument('-m', '--mode', help='train or valid mode?')
args = vars(parser.parse_args())

PROJECT = args['project']
MODE = args['mode']
MODEL_PATH = args['model_path']

compound_coef = 0
force_input_size = None  # set None to use default size

if not args['from_folder']:
    
    if not args['val_csv']:
        sys.exit('You are testing from a valid/train csv file. Please pass the same')
    if not args['mode']:
        sys.exit('You need to choose a train or valid mode. Please pass the same')
    
    data_folder = args['data_path']
    img_save_folder = 'results/' + PROJECT + '/' + MODE 

    if not os.path.isdir(img_save_folder):
        os.makedirs(img_save_folder)

    validation_df = pd.read_csv(args['val_csv'])
    columns = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    validation_df.columns = columns
    all_images = list(validation_df['filename'].values)
    all_images = [os.path.join(data_folder, i.split('latest/')[-1]) for i in all_images]
else:
    if not args['folder_path']:
        sys.exit('You are testing from a folder. Please pass the path to the folder')
        
    data_folder = args['folder_path']
    img_save_folder = 'results/' + PROJECT + '/' + data_folder.split('/')[-1]
    
    if not os.path.isdir(img_save_folder):
        os.makedirs(img_save_folder)
    
    all_images = [os.path.join(data_folder, i) for i in os.listdir(data_folder)]
    

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.5
iou_threshold = 0.5

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

if PROJECT == 'arthist':
    obj_list = ['putto', 'mary', 'gabriel', 'book', 'column', 'dove', 'bookrest', 'flower', 'angel', 'annunciation', 'flower vase', 'speech scroll', 'god', 'scepter', 'bed', 'basket', 'stool', 'vase', 'jesus child', 'cat', 'window', 'door'] 
elif PROJECT == 'classarch':
    obj_list = ['palmette', 'spear', 'wreath (worn)', 'column', 'lyre', 'dolphin', 'shield', 'altar', 'vessel (kantharos)', 'bow', 'sword', 'cock', 'winged sandal', 'Eros', 'vessel', 'club', 'trident', 'thyrsos', 'kerykeion', 'quiver', 'stick', 'petasos', 'scepter', 'torch', 'tripod', 'arrow', 'lions skin (headdress)', 'aulos', 'phiale', 'fish', 'wreath', 'lion', 'cornucopia', 'door', 'centaur', 'sphinx', 'stephane (bride)', 'dog', 'thunderbolt', 'vessel (oinochoe)', 'hoop', 'vessel (amphora)', 'owl', 'tauros', 'phrygian cap', 'kantharos', 'thymiaterion', 'pomegranate', 'hippocamp', 'basket', 'box', 'lions skin', 'axe', 'chimaira', 'pom', 'ship', 'panther', 'pegasus', 'harp', 'ram', 'octopus', 'griffin', 'vessel (loutrophoros)', 'bed', 'hand-held fan', 'taenia', 'winged sandals', 'harpe']
elif PROJECT == 'chrisarch':
    obj_list = ['basket', 'trousers', 'phrygian cap', 'cape', 'bringing object', 'servant/slave/defeaded', 'fountain', 'coat', 'wool', 'stooping posture']


color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
 
for img_path in tqdm.tqdm(all_images[:25]):
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=anchor_ratios, scales=anchor_scales)
    # model.load_state_dict(torch.load(f'logs/efi_arthist/efficientdet-d{compound_coef}.pth'))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    def display(preds, imgs, imshow=True, imwrite=False):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue

            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)], line_thickness=1)


            if imshow:
                cv2.imshow('img', imgs[i])
                cv2.waitKey(0)

            if imwrite:
                img = img_path.split('/')[-1].split('.')[0]
#                 pdb.set_trace()
                img_name = f'{img}_img_inferred_d{compound_coef}_this_repo_{i}.jpg'
                img_save_path = os.path.join(img_save_folder, img_name)
                print (img_save_path)
                cv2.imwrite(img_save_path, imgs[i])


    out = invert_affine(framed_metas, out)
    display(out, ori_imgs, imshow=False, imwrite=True)

#     print('running speed test...')
#     with torch.no_grad():
#         print('test1: model inferring and postprocessing')
#         print('inferring image for 10 times...')
#         t1 = time.time()
#         for _ in range(10):
#             _, regression, classification, anchors = model(x)

#             out = postprocess(x,
#                               anchors, regression, classification,
#                               regressBoxes, clipBoxes,
#                               threshold, iou_threshold)
#             out = invert_affine(framed_metas, out)

#         t2 = time.time()
#         tact_time = (t2 - t1) / 10
#         print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')
        
        # uncomment this if you want a extreme fps test
        # print('test2: model inferring only')
        # print('inferring images for batch_size 32 for 10 times...')
        # t1 = time.time()
        # x = torch.cat([x] * 32, 0)
        # for _ in range(10):
        #     _, regression, classification, anchors = model(x)
        #
        # t2 = time.time()
        # tact_time = (t2 - t1) / 10
        # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
