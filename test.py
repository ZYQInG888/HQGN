import os.path
import logging
import numpy as np
from collections import OrderedDict
import torch
import cv2
from utils import utils_logger
from utils import utils_image as util
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():

    task = 'Color'  # 'Color' or 'Gray'
    quality_factor_list = [10, 20, 30, 40]
    testset_name = 'LIVE1'  # 'LIVE1_color' 'BSDS500_color' 'ICB'
    n_channels = 3            # set 1 for grayscale image, set 3 for color image
    model_name = 'HQGN_color.pth'
    nc = [64, 128, 256, 512]
    nb = 4
    show_img = False                 # default: False
    testsets = 'testset'
    results = 'test_results'

    for quality_factor in quality_factor_list:

        result_name = task + '_' + testset_name + '_' + model_name[:-4]
        H_path = os.path.join(testsets, testset_name)
        E_path = os.path.join(results, result_name, str(quality_factor))   # E_path, for Estimated images
        util.mkdir(E_path)

        model_pool = 'deblocking'  # fixed
        model_path = os.path.join(model_pool, model_name)
        if os.path.exists(model_path):
            print(f'loading model from {model_path}')

        logger_name = result_name + '_qf_' + str(quality_factor)
        utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info('--------------- quality factor: {:d} ---------------'.format(quality_factor))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        border = 0

        # ----------------------------------------
        # load qf pretrained model
        # ----------------------------------------
        from models.qf_pred import resnet34 as qf_pred_net
        qf_predictor = qf_pred_net().to(device)
        pretrained_path = '/home/zyq/HQGN/deblocking/QF_color.pth'  # /home/zyq/HQGN/options/train_hqgn_gray.json

        for param in qf_predictor.parameters():  # 冻结预训练模型的参数
            param.requires_grad = False

        pretrained_dict = torch.load(pretrained_path)
        qf_predictor.load_state_dict(pretrained_dict)  # 加载预训练模型的权重
        qf_predictor.eval()
        logger.info('Qf_predictor model path: {:s}'.format(pretrained_path))

        # ----------------------------------------
        # load model
        # ----------------------------------------

        from models.network import HQGN as net
        model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        logger.info('Model path: {:s}'.format(model_path))

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnrb'] = []

        H_paths = util.get_image_paths(H_path)
        for idx, img in enumerate(H_paths):

            # ------------------------------------
            # (1) img_L
            # ------------------------------------
            img_name, ext = os.path.splitext(os.path.basename(img))
            logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
            img_L = util.imread_uint(img, n_channels=n_channels)

            if n_channels == 3:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)                
            _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            img_L = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 3)
            if n_channels == 3:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)               
            img_L = util.uint2tensor4(img_L)
            img_L = img_L.to(device)

            # ------------------------------------
            # (2) img_E
            # ------------------------------------

            QF, out1, out2, out3 = qf_predictor(img_L)
            img_E, features = model(img_L, QF, out1, out2, out3)
            QF = 1 - QF

            # img_L = util.tensor2uint(img_L)
            img_E = util.tensor2single(img_E)
            img_E = util.single2uint(img_E)
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels).squeeze()

            # --------------------------------
            # PSNR and SSIM, PSNRB
            # --------------------------------

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            psnrb = util.calculate_psnrb(img_H, img_E, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['psnrb'].append(psnrb)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.3f}; PSNRB: {:.2f} dB.'.format(img_name+ext, psnr, ssim, psnrb))
            logger.info('predicted quality factor: {:d}'.format(round(float(QF*100))))

            if show_img:
                concatenated_img = np.concatenate([img_L, img_E, img_H], axis=1)
                concat_path = os.path.join(E_path, 'concat')   # E_path, for Estimated images
                util.mkdir(concat_path)
                # util.imshow(concatenated_img, title='Recovered / Ground-truth')
                util.imsave(concatenated_img, os.path.join(concat_path, img_name + '.png'))
            else:
                None
            util.imsave(img_E, os.path.join(E_path, img_name+'.png'))

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
        logger.info(
             'Average PSNR/SSIM/PSNRB - {} -: {:.2f} | {:.4f} | {:.2f}.'.format(result_name+'_'+str(quality_factor), ave_psnr, ave_ssim, ave_psnrb))


if __name__ == '__main__':
    main()
