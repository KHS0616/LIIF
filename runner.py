"""
LIIF 학습, 추론 코드

Writer : KHS0616
Last Update : 2021-11-02
"""
import torch
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import make_grid, save_image

from .model import LIIF
from .utils import make_coord
from datasets import SRImplicitDownsampledDatasets

from PIL import Image
import numpy as np
import cv2

class Trainer():
    def __init__(self):
        # 초기 설정
        self.setDevice()
        self.setModel()
        self.setDataLoader()
        self.setOptimizer()
        self.setLoss()

    def setDevice(self):
        """ Device 설정 메소드 """
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def setModel(self):
        """ 학습 과정에서 사용할 모델 설정 메소드 """
        self.model = LIIF().to(self.device)
        self.model.load_state_dict(torch.load("./pretrainedModel/rdn-liif.pth")['model']["sd"])

    def setDataLoader(self):
        """ 데이터 셋 및 데이터 로더 설정 메소드 """
        self.datasets = SRImplicitDownsampledDatasets()
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.datasets,
            batch_size=16,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            drop_last=True
        )

    def setOptimizer(self):
        """ Optimizer 설정 메소드 """
        self.optimizer = torch.optim.Adam(params=self.model.state_dict(), lr=1.e-4, betas=(0.9, 0.999))
        self.scheduler = MultiStepLR(optimizer=self.optimizer, milestones=[200, 400, 600, 800], gamma=0.5)

    def setLoss(self):
        """ Loss 함수 설정 메소드 """
        self.loss_func = torch.nn.L1Loss()

    def process(self):
        """ 학습 메소드 """
        # 학습 데이터 정규화 값 설정
        inp_sub = torch.FloatTensor([0.5]).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor([0.5]).view(1, -1, 1, 1).cuda()
        gt_sub = torch.FloatTensor([0.5]).view(1, 1, -1).cuda()
        gt_div = torch.FloatTensor([0.5]).view(1, 1, -1).cuda()

        for epoch in range(1, 1000+1, 1):
            for i, data in enumerate(self.dataloader):
                # 데이터 GPU 등록
                for k, v in data.items():
                    data[k] = v.to(self.device)

                inp = (data['inp'] - inp_sub) / inp_div
                pred = self.model(inp, data['coord'], data['cell'])

                gt = (data['gt'] - gt_sub) / gt_div
                loss = self.loss_func(pred, gt)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()               

                # 100 iterator마다 이미지 저장
                if epoch * (i+1) % 100 == 0:
                    grid_tensor = make_grid(tensor=[pred, data['hinp']], normalize=True)
                    save_image(tensor=grid_tensor, fp=f"{epoch}.png")

                pred = None; loss = None

            # Optimizer Step
            self.scheduler.step()

            # 10 epoch마다 모델 저장
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"LIIF_epoch{i}.pth")

class Tester():
    """ LIFF 추론 클래스 """
    def __init__(self):
        self.setDevice()
        self.setModel()

    def setDevice(self):
        """ Device 설정 메소드 """
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def setModel(self):
        """ 추론 과정에서 사용할 모델 설정 메소드 """
        self.model = LIIF().to(self.device)
        self.model.load_state_dict(torch.load("./pretrainedModel/rdn-liif.pth")['model']["sd"])

    def preprocess(self, img):
        """ 이미지 전처리 메소드 """
        # 텐서로 변환
        if torch.is_tensor(img):
            img = img.to(self.device)
        else:
            img = transforms.ToTensor()(img).to(self.device)

        # 출력 해상도 값 저장
        h, w = 512, 512
        
        # 출력 좌표 맵 생성
        coord = make_coord((h, w)).cuda()

        # cell decoding을 위한 cell 생성
        cell = torch.ones_like(coord).to(self.device)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w

        return ((img - 0.5) / 0.5).unsqueeze(0), coord.unsqueeze(0), cell.unsqueeze(0)

    def process(self, img):
        """ 전체 프로세스 """
        # 이미지 전처리
        img, coord, cell = self.preprocess(img)

        # 옵션 설정
        bsize = 30000

        # 추론
        with torch.no_grad():
            # 인코더를 통한 특징 추출
            self.model.gen_feat(img)
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)

                # LIIF를 통한 RGB 예측 및 결과 저장
                pred = self.model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr

            # 모든 결과를 concat 연산을 통해 하나로 합하여 최종 이미지 생성
            pred = torch.cat(preds, dim=1)

        # 이미지 후처리
        img = self.postprocess(pred[0])

        return img

    def postprocess(self, img):
        """ 이미지 후처리 메소드 """
        img = (img * 0.5 + 0.5).clamp(0, 1).view(512, 512, 3).permute(2, 0, 1)
        img = transforms.ToPILImage()(img.cpu())#.save("output.png")
        numpy_image=np.array(img)
        # img=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return numpy_image
        
if __name__ == '__main__':
    run = Tester()
    run.process(Image.open("../tttt.png").convert('RGB'))