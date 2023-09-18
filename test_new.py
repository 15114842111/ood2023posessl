import argparse
from datetime import datetime
import logging
import os
import random

import numpy as np
import torch
import pandas as pd
from src.config import load_config
from src.evaluate import pose_error
from src.utils import setup_logging, set_seed
from src.models.utils import construct_class_by_name
from src.models.resnet import ResNetGeneral
# from src.models.resnet import ResNetGenerall
import os
import random
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
from PIL import Image
# from config import args_resnet50
# from utils import AverageMeter, accuracy
import logging
import sys
import timm
CATEGORIES = [
    'aeroplane', 'bicycle', 'boat', 'bus',
    'car', 'chair', 'diningtable', 'motorbike',
    'sofa', 'train']
# NUISANCE=['iid_test','weather','context','texture','pose','shape']
NUISANCE=['pose', 'shape', 'texture', 'context','weather' , 'occlusion']
# from evaluate_pose import get_acc
# theta_mean,elevation_mean,distance_mean,azimuth_mean= 0.00023595481126638268,0.11575889073578172,6.526204760750839,2.708895440086749
# theta_std,elevation_std,distance_std,azimuth_std= 0.13191307591839532,0.1993462425530023,5.318982103694582,2.5564238212543766
def parse_args():
    parser = argparse.ArgumentParser(description='Train baseline 3D pose estimation model for OOD-CV')
    parser.add_argument('--config', type=str, default='resnet.yaml')
    parser.add_argument('--save_dir', type=str, default=f'mistake_oodcv')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# def evaluate(cfg, dataloader, model):
#     pose_errors = []
#     predd=[]
#     for i, sample in enumerate(dataloader):
#         # print("sample",sample)
#         x=[]
#         pred = model.evaluate(sample)
#         x.append(pred)
#         x.append(sample['this_name'])
#         predd.append(x)
#         # print("pred",pred)
#         # print("pred[final]['azimuth']",pred['final'][0]['azimuth'])
#         # print("pred[final]['elevation']",pred['final'][0]['elevation'])
#         # print("pred[final]['theta']", pred['final'][0]['theta'])
#         # print("hhchvgd",pred['final'][0]['distance'])
#     #     _err = pose_error(sample, pred['final'][0])
#     #     pose_errors.append(_err)
#     # pose_errors = np.array(pose_errors)
#     #
#     # acc6 = np.mean(pose_errors < np.pi / 6) * 100
#     # acc18 = np.mean(pose_errors < np.pi / 18) * 100
#     # mederr = np.median(pose_errors) / np.pi * 180
#     # pred.to_csv('output/pred.csv', index=None)
#     return predd

class MytestDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        img_dir = f"/data/gmmm/data/images"
        self.data_list = []
        for name in os.listdir(img_dir):
            img_path = img_dir+'/'+name
            self.data_list.append([img_path])
        self.transform = transform
    def __getitem__(self, index):
        img_path = self.data_list[index][0]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (256, 256),interpolation=cv2.INTER_CUBIC)

        image = self.transform(Image.fromarray(image))
        return image,img_path
    def __len__(self):
        return len(self.data_list)

def test(cfg):
    # train_dataset = construct_class_by_name(**cfg.dataset, data_type='train', category='all')
    # if cfg.dataset.sampler is not None:
    #     train_dataset_sampler = construct_class_by_name(
    #         **cfg.dataset.sampler, dataset=train_dataset, rank=0, num_replicas=1,
    #         seed=cfg.training.random_seed)
    #     shuffle = False
    # else:
    #     train_dataset_sampler = None
    #     shuffle = True
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=cfg.training.batch_size,
    #     shuffle=shuffle,
    #     num_workers=cfg.training.workers,
    #     sampler=train_dataset_sampler)
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # testset = MytestDataset(transform=transform_test)
    # testloader = data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=8)
    test_dataset = construct_class_by_name(**cfg.dataset, data_type='test', category='all')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0)
    # logging.info(f"Number of training images: {len(train_dataset)}")
    # logging.info(f"Number of validation images: {len(val_dataset)}")

    model = construct_class_by_name(
        **cfg.model, cfg=cfg.model, cate='all', mode='test')
    # model = timm.create_model('resnet152', pretrained=True, num_classes=123, checkpoint_path='/data/meng/meng/OOD/resnet152/ckpts/model_90.pth')
    # model=ResNetGenerall(cfg=cfg.model, cate='all', mode='test',checkpoint='/data/meng/meng/OOD/resnet152/ckpts/model_90.pth',training=cfg.model.training)
    # model = nn.DataParallel(model, device_ids=[0])
    # model = model.cuda()
    # device=torch.device('cuda')
    # model=model.to(device)
    # pred = pd.DataFrame()
    # img_list = []
    # print("start")
    # for (inputs, img_path) in testloader:
    #     print("input")
    #     print(inputs)
    #     print("img_path")
    #     print(img_path)
    #     inputs = inputs.cuda()
    #     outputs = model(inputs)
    #     outputs = outputs.detach().cpu().numpy()
        # pred = pd.concat([pred, pd.DataFrame(outputs)])
        # img_list.extend(img_path)

    print("start")
    # results = evaluate(cfg, testloader, model)
    # results=evaluate(cfg,test_dataloader,model)
    # logging.info(
    #             f'[test {epo + 1}] acc@pi/6={results["acc6"]:.2f} acc@pi/18={results["acc18"]:.2f} mederr={results["mederr"]:.2f}')
    # print(results)
    import pandas as pd
    # last_pred=[]
    predd_1 = pd.DataFrame()

    for i, sample in enumerate(test_dataloader):
        predd={}
        # print("+++++++++++++++++++")
        name=sample['this_name'][0].split('/')[-1]
        # print(name)
        predd['img']=name
        # category_1 = name.split('_')[-1]
        # print(category_1)
        # category_2= sample['cad_index']
        # print(category_2)
        ###
        labels=CATEGORIES[sample['label'].numpy()[0]]
        # print(labels)
        predd['labels']=labels
        ###

        # lll=sample['cad_index'].numpy()[0]
        # print(lll)
        # nuisance=NUISANCE[sample['cad_index'].numpy()[0]]
        # print(nuisance)

        pred = model.evaluate(sample)
        # print(pred['final'])
        # predd['azimuth'] = (pred['final'][0]['azimuth'] * azimuth_std) + azimuth_mean
        # predd['elevation'] = (pred['final'][0]['elevation'] * elevation_std) + elevation_mean
        # predd['theta'] = (pred['final'][0]['theta'] * theta_std) + theta_mean
        # predd['distance']=5.0
        # predd['nuisance']='iidtest'

        predd['azimuth'] = pred['final'][0]['azimuth']
        predd['elevation'] = pred['final'][0]['elevation']
        predd['theta'] = pred['final'][0]['theta']
        predd['distance'] = 5.0
        predd['nuisance'] = 'pose'
        # pred=model.evaluate(sample)
        # print(pred['final'])
        # pred['imgs'] = [img.split('/')[-1].split('.')[0] for img in img_list]
        # print(pred['imgs'])
        # pred['labels'] = [img.split('/')[-1].split('.')[0].split('_')[-1] for img in img_list]
        # print("*********************")
        # print(i)
        # print(sample)
        # print("9999999999999")
        # last_pred.append(predd)
        predd_1 = pd.concat([predd_1,pd.DataFrame(predd,index=[0])])
        # print(predd)
        # print("----------------------")
    predd_1.to_csv(r'/data/meng/meng/OOD/mmtest/pred_last.csv',index=False)
    # pred={}
    # pred.columns = ['theta', 'elevation', 'distance', 'azimuth']  # theta,elevation,distance,azimuth
    # pred['theta'] = (pred['theta'] * theta_std) + theta_mean
    # pred['elevation'] = (pred['elevation'] * elevation_std) + elevation_mean
    # # pred['distance'] = (pred['distance'] * distance_std) + distance_mean
    # pred['distance'] = 5.0
    # pred['azimuth'] = (pred['azimuth'] * azimuth_std) + azimuth_mean
    # pred['imgs'] = [img.split('/')[-1] for img in img_list]
    # pred['labels'] = [img.split('/')[-1].split('.')[0].split('_')[-1] for img in img_list]
    # results.to_csv('./test/pred.csv', index=None)






    # # model.step_scheduler()
    # print(results)
    # results.to_csv('output/result.csv', index=None)
def main():
    args = parse_args()

    setup_logging(args.save_dir)
    set_seed(args.seed)

    cfg = load_config(args, load_default_config=False, log_info=False)
    logging.info(args)
    logging.info(cfg)

    test(cfg)


if __name__ == '__main__':
    main()
