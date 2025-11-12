import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


if __name__ == '__main__':
    model = YOLO('/home/csyu/CMFADet/ultralytics/cfg/models/multimodal/Multi-Baseline-obb.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    # model = YOLO('/home/csyu/CMFADet/runs/OGSOD/train/exp/weights/last.pt') ## Resume traing setting
    model.train(data='/home/csyu/CMFADet/dataset/data_OGSOD_Multimodel.yaml',
                cache=False,
                # imgsz=640,
                imgsz=512,
                epochs=100,
                batch=8,
                close_mosaic=0,
                workers=4, 
                # device='0,1',
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # Resume setting
                amp=False,
                # fraction=0.2,
                project='runs/OGSOD/train',
                name='Multi-Baseline-obb-512',
                # val = False,
                )
