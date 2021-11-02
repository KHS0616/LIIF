"""
LIIF Datasets Code

Writer : KHS0616
Last Update : 2021-11-02
"""
import torch
from torchvision import transforms

from utils import to_pixel_samples

import os
import random
import math
from PIL import Image
import numpy as np

def check_image(file_name):
    """ 이미지 체크 함수 """
    ext = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")
    return True if file_name.endswith(ext) else False

def resize_fn(img, size):
    """ 이미지 크기 변경 함수 """
    return transforms.ToTensor()(transforms.Resize(size, Image.BICUBIC)(transforms.ToPILImage()(img)))

class SRImplicitDownsampledDatasets(torch.utils.data.Dataset):
    """ LIIF 랜덤 Downsample 데이터셋 클래스 """
    def __init__(self):
        # 파일 경로 및 리스트 저장
        self.file_path = "./Image"
        self.file_list = [x for x in os.listdir(self.file_path) if check_image(x)]

        # Random Downsample Scale 범위 지정
        self.scale_min, self.scale_max = 1, 4

        # Crop 사이즈 설정
        self.inp_size = 48

        # random augment 여부 설정
        self.augment = True

        # 학습을 위한 샘플 개수
        self.sample_q = 2304

    def __getitem__(self, idx):
        # 이미지 열기
        file_full_path = os.path.join(self.file_path, self.file_list[idx])
        img_tensor = transforms.ToTensor()(Image.open(file_full_path).convert('RGB'))

        # 랜덤 Downsample Scale 수치 저장
        s = random.uniform(self.scale_min, self.scale_max)

        # Crop 사이즈 지정 여부 확인 후 분기
        # 랜덤 crop 실행
        if self.inp_size is None:
            # math.floor - 내림 연산
            h_lr = math.floor(img_tensor.shape[-2] / s + 1e-9)
            w_lr = math.floor(img_tensor.shape[-1] / s + 1e-9)
            img = img_tensor[:, :round(h_lr * s), :round(w_lr * s)]
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img_tensor.shape[-2] - w_hr)
            y0 = random.randint(0, img_tensor.shape[-1] - w_hr)
            # print("랜덤 좌표 및 스케일 수치 : ", x0, y0, s)
            crop_hr = img_tensor[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        # 랜덤 augment 실행
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # 크롭된 이미지에서 좌표 값, rgb 값 측정
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # 지정된 sample 수 만큼 좌표, rgb 값 선택
        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        # cell decoding 원리에 따른 cell 측정
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {'inp': crop_lr, 'coord': hr_coord, 'cell': cell, 'gt': hr_rgb, 'hinp': crop_hr}

    def __len__(self):
        return len(self.file_list)