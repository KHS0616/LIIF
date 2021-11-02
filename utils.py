import torch

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
        중앙 좌표 측정
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        # r - 2를 현 w크기의 2배수로 나눔
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    # torch.meshgrid - 입력한 텐서들의 총합 shape 만큼 각각의 텐서를 해당 shape만큼 변형시킨다.(값은 해당 텐서로 이용)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    # print("Flat 이전 : ", ret.shape)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
        이미지를 좌표와 값으로 변환
    """
    # -2는 w값만 가져가겠다는 의미
    # print("입력 이미지 사이즈", img.shape, img.shape[-2:])
    coord = make_coord(img.shape[-2:])
    # print("입력 이미지의 coord 사이즈 : ", coord.shape)
    # 3채널의 rgb값 생성
    rgb = img.view(3, -1).permute(1, 0)
    # print("rgb value 사이즈 : ", rgb.shape)
    return coord, rgb