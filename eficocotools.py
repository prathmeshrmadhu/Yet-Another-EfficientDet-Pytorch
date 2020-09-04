import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import copy, pdb
import itertools
import os
from collections import defaultdict
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class EFI_COCO:
    def __init__(self, obj_list, annotation_file=None):
        """
        Constructor of Microsoft COCO for EFI helper class for reading and visualizing annotations.
        :param annotation_file (str): location of csv annotation file for bounding boxes
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset_df = pd.read_csv(annotation_file, header=None)
            columns = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
            dataset_df.columns = columns
            assert type(dataset_df)==pd.core.frame.DataFrame, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            unique_filenames = np.unique(dataset_df['filename'])
            filename_id_dict = {val: i for i, val in enumerate(unique_filenames)}
            cat_id_dict = {val: i for i, val in enumerate(obj_list)}
            dataset_df['file_id'] = dataset_df['filename'].map(filename_id_dict)
            dataset_df['cat_id'] = dataset_df['class'].map(cat_id_dict)
            dataset_df['id'] = range(len(dataset_df))
            self.dataset_df = dataset_df
            self.obj_list = obj_list
            self.dataset = self.createCOCOjson()
            self.createIndex()

    def createCOCOjson(self):

        dataset = {}
        annotations = [] # list of dicts
        images = [] # list of dicts
        categories = [] # list of dicts (each category is a dict)
        for i, row in self.dataset_df.iterrows():

            ann = {}
            ann['bbox'] = [row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']]
            ann['id'] = row['id']
            ann['image_id'] = row['file_id']
            ann['category_id'] = row['cat_id']
            ann['iscrowd'] = 0
            annotations.append(ann)

            img = {}
            img['file_name'] = row['filename'].split('latest/')[-1]
            img['id'] = row['file_id']
            images.append(img)

        for i, val in enumerate(self.obj_list):
            cat = {}
            cat['id'] = i
            cat['name'] = val
            categories.append(cat)

        dataset['annotations'] = annotations
        dataset['images'] = images
        dataset['categories'] = categories

        return dataset

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        # pdb.set_trace()

    def getAnnIds(self, imgIds=[], catIds=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]

        ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int or type(ids) == np.int64:
            return [self.imgs[ids]]

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = EFI_COCO(self.obj_list)
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or type(resFile) == unicode:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'

        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            'Results do not correspond to current coco set'

        if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                # if not 'segmentation' in ann:
                #     ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        else:
            print ("Only valid of Bounding box - Object detection")

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert (type(data) == np.ndarray)
        print(data.shape)
        assert (data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i, N))
            ann += [{
                'image_id': int(data[i, 0]),
                'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                'score': data[i, 5],
                'category_id': int(data[i, 6]),
            }]
        return ann
