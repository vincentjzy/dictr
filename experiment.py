import torch
import numpy as np
import cv2
from utils.utils import InputPadder


# Recommend 128×128 or 256×256 size figs for custom test
@torch.no_grad()
def custom(model,
           padding_factor=8,
           attn_splits_list=False,
           corr_radius_list=False,
           prop_radius_list=False,
           ):
    model.eval()

    ref = cv2.imread('/path/to/your/fig', cv2.IMREAD_GRAYSCALE)
    tar = cv2.imread('/path/to/your/fig', cv2.IMREAD_GRAYSCALE)
    ref = torch.Tensor(ref).cuda()
    tar = torch.Tensor(tar).cuda()
    ref = ref.unsqueeze(dim=0).unsqueeze(dim=0)
    tar = tar.unsqueeze(dim=0).unsqueeze(dim=0)

    padder = InputPadder(ref.shape, padding_factor=padding_factor)
    ref, tar = padder.pad(ref, tar)

    results_dict = model(ref, tar,
                         attn_splits_list=attn_splits_list,
                         corr_radius_list=corr_radius_list,
                         prop_radius_list=prop_radius_list,
                         )

    flow_pr = results_dict['flow_preds'][-1].cpu()
    flow_pr = padder.unpad(flow_pr[0]).cpu()

    dispx = flow_pr[0, :, :].numpy()
    dispy = flow_pr[1, :, :].numpy()

    height, width = ref.shape[-2:]

    with open('./test/U.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispx[i, j])
            f.write('\n')

    with open('./test/V.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')


@torch.no_grad()
def rotation_128(model,
             attn_splits_list=False,
             corr_radius_list=False,
             prop_radius_list=False,
             ):
    model.eval()

    ref = cv2.imread('./test/supervised/rotation/rotationREF.bmp', cv2.IMREAD_GRAYSCALE)
    tar = cv2.imread('./test/supervised/rotation/rotationTAR.bmp', cv2.IMREAD_GRAYSCALE)
    ref = torch.Tensor(ref).cuda()
    tar = torch.Tensor(tar).cuda()
    ref = ref.unsqueeze(dim=0).unsqueeze(dim=0)
    tar = tar.unsqueeze(dim=0).unsqueeze(dim=0)

    results_dict = model(ref, tar,
                         attn_splits_list=attn_splits_list,
                         corr_radius_list=corr_radius_list,
                         prop_radius_list=prop_radius_list,
                         )

    flow_pr = results_dict['flow_preds'][-1]
    flow_pr = flow_pr[0].cpu()

    dispx = flow_pr[0, :, :].numpy()
    dispy = flow_pr[1, :, :].numpy()

    height, width = ref.shape[-2:]

    with open('./test/supervised/rotationU.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispx[i, j])
            f.write('\n')

    with open('./test/supervised/rotationV.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')


@torch.no_grad()
def tension(model,
            attn_splits_list=False,
            corr_radius_list=False,
            prop_radius_list=False,
            ):
    model.eval()

    ref = cv2.imread('./test/tension/tensionREF.bmp', cv2.IMREAD_GRAYSCALE)
    tar = cv2.imread('./test/tension/tensionTAR.bmp', cv2.IMREAD_GRAYSCALE)
    ref = torch.Tensor(ref).cuda()
    tar = torch.Tensor(tar).cuda()
    ref = ref.unsqueeze(dim=0).unsqueeze(dim=0)
    tar = tar.unsqueeze(dim=0).unsqueeze(dim=0)

    results_dict = model(ref, tar,
                         attn_splits_list=attn_splits_list,
                         corr_radius_list=corr_radius_list,
                         prop_radius_list=prop_radius_list,
                         )

    flow_pr = results_dict['flow_preds'][-1]
    flow_pr = flow_pr[0].cpu()

    dispx = flow_pr[0, :, :].numpy()
    dispy = flow_pr[1, :, :].numpy()

    height, width = ref.shape[-2:]

    with open('./test/supervised/tensionU.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispx[i, j])
            f.write('\n')

    with open('./test/supervised/tensionV.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')


# Large computation cost at full resolution, Star deformation field with only y-axis displacement is split along the x-axis
@torch.no_grad()
def star5(model,
          attn_splits_list=False,
          corr_radius_list=False,
          prop_radius_list=False,
          ):
    model.eval()

    # v-component calculation in deformed fig with noise
    img1 = cv2.imread('./test/supervised/star5/DIC_Challenge_Star5_Noise_Ref.bmp', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./test/supervised/star5/DIC_Challenge_Star5_Noise_Def.bmp', cv2.IMREAD_GRAYSCALE)
    img1 = torch.Tensor(img1).cuda()
    img2 = torch.Tensor(img2).cuda()
    img1 = img1.unsqueeze(dim=0).unsqueeze(dim=0)
    img2 = img2.unsqueeze(dim=0).unsqueeze(dim=0)

    begin = 0
    end = 127
    interval = 16

    rising = 128 // 2 - interval
    falling = 128 // 2 + interval - 1

    dispy = np.zeros((128, 2048))

    while end < 2048:

        ref = img1[:, :, :, begin:end + 1]
        tar = img2[:, :, :, begin:end + 1]

        results_dict = model(ref, tar,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             )

        flow_pr = results_dict['flow_preds'][-1].cpu()

        if begin == 0:
            dispy[:, begin:begin + falling + 1] = flow_pr[0, 1, :, 0:falling + 1].numpy()
        elif end == 2047:
            dispy[:, begin + rising:begin + 127 + 1] = flow_pr[0, 1, :, rising:127 + 1].numpy()
        else:
            dispy[:, begin + rising:begin + falling + 1] = flow_pr[0, 1, :, rising:falling + 1].numpy()

        begin = begin + interval
        end = begin + 127

    with open('./test/supervised/star5V.csv', 'w') as f:

        for i in range(128):
            for j in range(2048):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')

    # v-component calculation in undeformed fig with noise
    img1 = cv2.imread('./test/supervised/star5/DIC_Challenge_Star5_Noise_Ref.bmp', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./test/supervised/star5/DIC_Challenge_Star5_Noise_Ref2.bmp', cv2.IMREAD_GRAYSCALE)
    img1 = torch.Tensor(img1).cuda()
    img2 = torch.Tensor(img2).cuda()
    img1 = img1.unsqueeze(dim=0).unsqueeze(dim=0)
    img2 = img2.unsqueeze(dim=0).unsqueeze(dim=0)

    begin = 0
    end = 127
    interval = 16

    rising = 128 // 2 - interval
    falling = 128 // 2 + interval - 1

    dispy = np.zeros((128, 2048))

    while end < 2048:

        ref = img1[:, :, :, begin:end + 1]
        tar = img2[:, :, :, begin:end + 1]

        results_dict = model(ref, tar,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             )

        flow_pr = results_dict['flow_preds'][-1].cpu()

        if begin == 0:
            dispy[:, begin:begin + falling + 1] = flow_pr[0, 1, :, 0:falling + 1].numpy()
        elif end == 2047:
            dispy[:, begin + rising:begin + 127 + 1] = flow_pr[0, 1, :, rising:127 + 1].numpy()
        else:
            dispy[:, begin + rising:begin + falling + 1] = flow_pr[0, 1, :, rising:falling + 1].numpy()

        begin = begin + interval
        end = begin + 127

    with open('./test/supervised/star5PseudoV.csv', 'w') as f:

        for i in range(128):
            for j in range(2048):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')


# Large computation cost at full resolution, Star deformation field with only y-axis displacement is split along the x-axis
# Further statistical analysis based on the result from line 64 (csv starting from 1).
@torch.no_grad()
def mei_128(model,
        attn_splits_list=False,
        corr_radius_list=False,
        prop_radius_list=False,
        ):
    model.eval()

    # v-component calculation in deformed fig without noise (spatial resolution)
    img1 = cv2.imread('./test/supervised/mei/DIC_Challenge_Star1_NoiseFree_Ref.bmp', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./test/supervised/mei/DIC_Challenge_Star1_NoiseFree_Def.bmp', cv2.IMREAD_GRAYSCALE)
    img1 = torch.Tensor(img1).cuda()
    img2 = torch.Tensor(img2).cuda()
    img1 = img1.unsqueeze(dim=0).unsqueeze(dim=0)
    img2 = img2.unsqueeze(dim=0).unsqueeze(dim=0)

    begin = 0
    end = 127
    interval = 16

    rising = 128 // 2 - interval
    falling = 128 // 2 + interval - 1

    dispy = np.zeros((128, 2048))

    while end < 2048:

        ref = img1[:, :, :, begin:end + 1]
        tar = img2[:, :, :, begin:end + 1]

        results_dict = model(ref, tar,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             )

        flow_pr = results_dict['flow_preds'][-1].cpu()

        if begin == 0:
            dispy[:, begin:begin + falling + 1] = flow_pr[0, 1, :, 0:falling + 1].numpy()
        elif end == 2047:
            dispy[:, begin + rising:begin + 127 + 1] = flow_pr[0, 1, :, rising:127 + 1].numpy()
        else:
            dispy[:, begin + rising:begin + falling + 1] = flow_pr[0, 1, :, rising:falling + 1].numpy()

        begin = begin + interval
        end = begin + 127

    with open('./test/supervised/V-SR.csv', 'w') as f:

        for i in range(128):
            for j in range(2048):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')

    # v-component calculation in undeformed fig with noise (measurement resolution)
    img1 = cv2.imread('./test/supervised/mei/DIC_Challenge_Star2_Noise_Ref.bmp', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./test/supervised/mei/DIC_Challenge_Star2_Noise_Ref2.bmp', cv2.IMREAD_GRAYSCALE)
    img1 = torch.Tensor(img1).cuda()
    img2 = torch.Tensor(img2).cuda()
    img1 = img1.unsqueeze(dim=0).unsqueeze(dim=0)
    img2 = img2.unsqueeze(dim=0).unsqueeze(dim=0)

    begin = 0
    end = 127
    interval = 16

    rising = 128 // 2 - interval
    falling = 128 // 2 + interval - 1

    dispy = np.zeros((128, 2048))

    while end < 2048:

        ref = img1[:, :, :, begin:end + 1]
        tar = img2[:, :, :, begin:end + 1]

        results_dict = model(ref, tar,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             )

        flow_pr = results_dict['flow_preds'][-1].cpu()

        if begin == 0:
            dispy[:, begin:begin + falling + 1] = flow_pr[0, 1, :, 0:falling + 1].numpy()
        elif end == 2047:
            dispy[:, begin + rising:begin + 127 + 1] = flow_pr[0, 1, :, rising:127 + 1].numpy()
        else:
            dispy[:, begin + rising:begin + falling + 1] = flow_pr[0, 1, :, rising:falling + 1].numpy()

        begin = begin + interval
        end = begin + 127

    with open('./test/V-MR.csv', 'w') as f:

        for i in range(128):
            for j in range(2048):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')


@torch.no_grad()
def realcrack(model,
              attn_splits_list=False,
              corr_radius_list=False,
              prop_radius_list=False,
              ):
    model.eval()

    ref = cv2.imread('./test/supervised/scrack/crackREF.bmp', cv2.IMREAD_GRAYSCALE)
    tar = cv2.imread('./test/supervised/crack/crackTAR.bmp', cv2.IMREAD_GRAYSCALE)
    ref = torch.Tensor(ref).cuda()
    tar = torch.Tensor(tar).cuda()
    ref = ref.unsqueeze(dim=0).unsqueeze(dim=0)
    tar = tar.unsqueeze(dim=0).unsqueeze(dim=0)

    results_dict = model(ref, tar,
                         attn_splits_list=attn_splits_list,
                         corr_radius_list=corr_radius_list,
                         prop_radius_list=prop_radius_list,
                         )

    flow_pr = results_dict['flow_preds'][-1]
    flow_pr = flow_pr[0].cpu()

    dispx = flow_pr[0, :, :].numpy()
    dispy = flow_pr[1, :, :].numpy()

    height, width = ref.shape[-2:]

    with open('./test/supervised/crackU.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispx[i, j])
            f.write('\n')

    with open('./test/supervised/crackV.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')
import time

@torch.no_grad()
def rotation_256(model,
           attn_splits_list=False,
           corr_radius_list=False,
           prop_radius_list=False,
           ):
    model.eval()

    ref = cv2.imread('./test/unsupervised/rotation/rotationREF.bmp', cv2.IMREAD_GRAYSCALE)
    tar = cv2.imread('./test/unsupervised/rotation/rotationTAR.bmp', cv2.IMREAD_GRAYSCALE)

    ref = torch.Tensor(ref).cuda()
    tar = torch.Tensor(tar).cuda()
    ref = ref.unsqueeze(dim=0).unsqueeze(dim=0)
    tar = tar.unsqueeze(dim=0).unsqueeze(dim=0)

    for i in range(1):
        # if 30 warm up
        torch.cuda.synchronize()
        start = time.perf_counter()
        results_dict = model(ref, tar,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            )
        
        end = time.perf_counter()
        torch.cuda.synchronize()
        elapsed = end - start
        height, width = ref.shape[-2:]
        num_poi = height * width
        poi_speed = num_poi / elapsed if elapsed > 0 else float('inf')
        # print(poi_speed)

    flow_pr = results_dict['flow_preds'][-1]
    flow_pr = flow_pr[0].cpu()

    dispx = flow_pr[0, :, :].numpy()
    dispy = flow_pr[1, :, :].numpy()


    with open('./test/unsupervised/rotationU.csv', 'w') as f:
        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispx[i, j])
            f.write('\n')

    with open('./test/unsupervised/rotationV.csv', 'w') as f:
        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')

@torch.no_grad()
def mei_256(model,
        attn_splits_list=False,
        corr_radius_list=False,
        prop_radius_list=False,
        ):
    model.eval()

    # v-component calculation in deformed fig without noise (spatial resolution)
    img1 = cv2.imread('./test/unsupervised/mei/DIC_Challenge_Star1_NoiseFree_Ref.bmp', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./test/unsupervised/mei/DIC_Challenge_Star1_NoiseFree_Def.bmp', cv2.IMREAD_GRAYSCALE)
    img1 = torch.Tensor(img1).cuda()
    img2 = torch.Tensor(img2).cuda()
    img1 = img1.unsqueeze(dim=0).unsqueeze(dim=0)
    img2 = img2.unsqueeze(dim=0).unsqueeze(dim=0)

    begin = 0
    end = 255
    interval = 64

    rising = 256 // 2 - interval
    falling = 256 // 2 + interval - 1

    dispy = np.zeros((256, 1024))

    while end < 1024:

        ref = img1[:, :, :, begin:end + 1]
        tar = img2[:, :, :, begin:end + 1]

        results_dict = model(ref, tar,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             )

        flow_pr = results_dict['flow_preds'][-1].cpu()

        if begin == 0:
            dispy[:, begin:begin + falling + 1] = flow_pr[0, 1, :, 0:falling + 1].numpy()
        elif end == 1023:
            dispy[:, begin + rising:begin + 255 + 1] = flow_pr[0, 1, :, rising:255 + 1].numpy()
        else:
            dispy[:, begin + rising:begin + falling + 1] = flow_pr[0, 1, :, rising:falling + 1].numpy()

        begin = begin + interval
        end = begin + 255

    with open('./test/unsupervised/V-SR.csv', 'w') as f:

        for i in range(256):
            for j in range(1024):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')

    # v-component calculation in undeformed fig with noise (measurement resolution)
    img1 = cv2.imread('./test/unsupervised/mei/DIC_Challenge_Star2_Noise_Ref.bmp', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./test/unsupervised/mei/DIC_Challenge_Star2_Noise_Ref2.bmp', cv2.IMREAD_GRAYSCALE)
    img1 = torch.Tensor(img1).cuda()
    img2 = torch.Tensor(img2).cuda()
    img1 = img1.unsqueeze(dim=0).unsqueeze(dim=0)
    img2 = img2.unsqueeze(dim=0).unsqueeze(dim=0)

    begin = 0
    end = 255
    interval = 64

    rising = 256 // 2 - interval
    falling = 256 // 2 + interval - 1

    dispy = np.zeros((256, 1024))

    while end < 1024:

        ref = img1[:, :, :, begin:end + 1]
        tar = img2[:, :, :, begin:end + 1]

        results_dict = model(ref, tar,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             )

        flow_pr = results_dict['flow_preds'][-1].cpu()

        if begin == 0:
            dispy[:, begin:begin + falling + 1] = flow_pr[0, 1, :, 0:falling + 1].numpy()
        elif end == 1023:
            dispy[:, begin + rising:begin + 255 + 1] = flow_pr[0, 1, :, rising:255 + 1].numpy()
        else:
            dispy[:, begin + rising:begin + falling + 1] = flow_pr[0, 1, :, rising:falling + 1].numpy()

        begin = begin + interval
        end = begin + 255

    with open('./test/V-MR.csv', 'w') as f:

        for i in range(256):
            for j in range(1024):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')


@torch.no_grad()
def shear(model,
              attn_splits_list=False,
              corr_radius_list=False,
              prop_radius_list=False,
              ):
    model.eval()

    ref = cv2.imread('./test/unsupervised/shear/s2901.bmp', cv2.IMREAD_GRAYSCALE)
    tar = cv2.imread('./test/unsupervised/shear/s3200.bmp', cv2.IMREAD_GRAYSCALE)

    ref = torch.Tensor(ref).cuda()
    tar = torch.Tensor(tar).cuda()
    ref = ref.unsqueeze(dim=0).unsqueeze(dim=0)
    tar = tar.unsqueeze(dim=0).unsqueeze(dim=0)

    results_dict = model(ref, tar,
                         attn_splits_list=attn_splits_list,
                         corr_radius_list=corr_radius_list,
                         prop_radius_list=prop_radius_list,
                         )

    flow_pr = results_dict['flow_preds'][-1]
    flow_pr = flow_pr[0].cpu()

    dispx = flow_pr[0, :, :].numpy()
    dispy = flow_pr[1, :, :].numpy()

    height, width = ref.shape[-2:]

    with open('./test/unsupervised/shearU.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispx[i, j])
            f.write('\n')

    with open('./test/unsupervised/shearV.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')
