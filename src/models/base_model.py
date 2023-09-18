import numpy as np
import torch
import torchvision

from .utils import construct_class_by_name


class BaseModel:
    def __init__(
        self, cfg, cate, mode, checkpoint='/data/meng/meng/OOD/resnet152/ckpts/model_90.pth', transforms=[], loss_names=[], device="cuda:0"
    ):
        self.cfg = cfg
        self.cate = cate
        self.mode = mode
        if checkpoint is None:
            assert (
                mode == "train"
            ), "The checkpoint should not be None in validation or test mode"
            self.checkpoint = None
        else:
            print("我到我们自己的ck了")
            print(checkpoint)
            self.checkpoint = torch.load(checkpoint, map_location=device)
        # print("resnet 输入的trans")
        # print(transforms)
        # print("********")
        self.transforms = torchvision.transforms.Compose(
            [construct_class_by_name(**t) for t in transforms]
        )
        self.loss_names = loss_names
        # print("resnet losses name")
        # print(loss_names)#['loss']
        self.device = device

        self.loss_trackers = {}
        for l in loss_names:
            # print("resnet l")
            # print(l)#{'loss': []}
            self.loss_trackers[l] = []
        # print(self.loss_trackers)
        # print("end")
    def build(self):
        pass

    def get_training_state(self):
        state_msg = f'lr={self.optim.param_groups[0]["lr"]:.5f}'
        for l in self.loss_names:
            state_msg += f' {l}={np.mean(self.loss_trackers[l]):.5f}'
            self.loss_trackers[l] = []
        return state_msg

    def train(self, sample):
        # print("resnet self.trans")
        # print(self.transforms)
        sample = self.transforms(sample)
        raise NotImplementedError

    def evaluate(self, sample):
        sample = self.transforms(sample)
        raise NotImplementedError
    
    def step_scheduler(self):
        self.scheduler.step()
    
    def train(self, sample):
        # print("xiayige train trans")
        # print(self.transforms)
        self.model.train()
        sample = self.transforms(sample)
        return self._train(sample)
    
    def evaluate(self, sample):
        self.model.eval()
        sample = self.transforms(sample)
        return self._evaluate(sample)
