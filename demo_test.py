import time
import argparse
import scipy.io
from utils.utils import *
from utils.imresize import *
from model import Net
import numpy as np
import imageio
from einops import rearrange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes_in", type=int, default=2, help="input angular resolution")
    parser.add_argument("--angRes_out", type=int, default=7, help="output angular resolution")
    parser.add_argument("--scene_type", type=str, default='HCI')
    # parser.add_argument("--model_name", type=str, default='DistgASR_Lytro_2x2-7x7')
    parser.add_argument("--model_name", type=str, default='DistgASR_Lytro_2x2-7x7')
    parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--patchsize", type=int, default=128, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument('--input_dir', type=str, default='./input/')
    parser.add_argument('--save_path', type=str, default='./output/')

    return parser.parse_args()

def demo_test(cfg):
    net = Net(2, 7)
    net.to(cfg.device)
    model = torch.load('./log/' + cfg.model_name + '.pth.tar', map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])
    scene_list = os.listdir(cfg.input_dir + cfg.scene_type)

    for scenes in scene_list:
        print('Working on scene: ' + scenes + '...')
        temp = imageio.imread(cfg.input_dir + cfg.scene_type + '/' + scenes + '/view_01_01.png')
        lf_rgb_in = np.zeros(shape=(cfg.angRes_in, cfg.angRes_in, temp.shape[0], temp.shape[1], 3))
        lf_rgb_out = np.zeros(shape=(cfg.angRes_out, cfg.angRes_out, temp.shape[0], temp.shape[1], 3)).astype('float32')

        lf_rgb_in[0, 0, :, :, :] = imageio.imread(cfg.input_dir + cfg.scene_type + '/' + scenes + '/view_01_01.png')
        lf_rgb_in[0, 1, :, :, :] = imageio.imread(cfg.input_dir + cfg.scene_type + '/' + scenes + '/view_01_07.png')
        lf_rgb_in[1, 0, :, :, :] = imageio.imread(cfg.input_dir + cfg.scene_type + '/' + scenes + '/view_07_01.png')
        lf_rgb_in[1, 1, :, :, :] = imageio.imread(cfg.input_dir + cfg.scene_type + '/' + scenes + '/view_07_07.png')

        lf_y_in = (0.256789 * lf_rgb_in[:,:,:,:,0] + 0.504129 * lf_rgb_in[:,:,:,:,1] + 0.097906 * lf_rgb_in[:,:,:,:,2] + 16).astype('float32')
        lf_cb_in = (-0.148223 * lf_rgb_in[:,:,:,:,0] - 0.290992 * lf_rgb_in[:,:,:,:,1] + 0.439215 * lf_rgb_in[:,:,:,:,2] + 128).astype('float32')
        lf_cr_in = (0.439215 * lf_rgb_in[:,:,:,:,0] - 0.367789 * lf_rgb_in[:,:,:,:,1] - 0.071426 * lf_rgb_in[:,:,:,:,2] + 128).astype('float32')

        lf_cb_out = np.zeros(shape=(cfg.angRes_out, cfg.angRes_out, temp.shape[0], temp.shape[1])).astype('float32')
        lf_cr_out = np.zeros(shape=(cfg.angRes_out, cfg.angRes_out, temp.shape[0], temp.shape[1])).astype('float32')
        for h in range(temp.shape[0]):
            for w in range(temp.shape[1]):
                lf_cb_out[:, :, h, w] = imresize(lf_cb_in[:, :, h, w], cfg.angRes_out/cfg.angRes_in)
                lf_cr_out[:, :, h, w] = imresize(lf_cr_in[:, :, h, w], cfg.angRes_out/cfg.angRes_in)

        data = rearrange(lf_y_in, 'u v h w -> (u h) (v w)')
        data = torch.from_numpy(data) / 255.0

        if cfg.crop == False:
            with torch.no_grad():
                outLF = net(data.unsqueeze(0).unsqueeze(0).to(cfg.device))
                lf_y_out =  rearrange(outLF.squeeze(), '(u h) (v w) -> u v h w', u=cfg.angRes_out, v=cfg.angRes_out)
        else:
            patchsize = cfg.patchsize
            stride = patchsize // 2
            uh, vw = data.shape
            h0, w0 = uh // cfg.angRes_in, vw // cfg.angRes_in
            subLFin = LFdivide(data, cfg.angRes_in, patchsize, stride)  # numU, numV, h*angRes, w*angRes
            numU, numV, H, W = subLFin.shape
            subLFout = torch.zeros(numU, numV, cfg.angRes_out * patchsize, cfg.angRes_out * patchsize)

            for u in range(numU):
                for v in range(numV):
                    tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        out = net(tmp.to(cfg.device))
                        subLFout[u, v, :, :] = out.squeeze()
            lf_y_out = LFintegrate(subLFout, cfg.angRes_out, patchsize, stride, h0, w0)

        lf_y_out = 255 * lf_y_out.data.cpu().numpy()
        lf_rgb_out[:, :, :, :, 0] = 1.164383 * (lf_y_out - 16) + 1.596027 * (lf_cr_out - 128)
        lf_rgb_out[:, :, :, :, 1] = 1.164383 * (lf_y_out - 16) - 0.391762 * (lf_cb_out - 128) - 0.812969 * (lf_cr_out - 128)
        lf_rgb_out[:, :, :, :, 2] = 1.164383 * (lf_y_out - 16) + 2.017230 * (lf_cb_out - 128)

        lf_rgb_out = np.clip(lf_rgb_out, 0, 255)
        output_path = cfg.save_path + scenes
        if not (os.path.exists(output_path)):
            os.makedirs(output_path)
        for u in range(cfg.angRes_out):
            for v in range(cfg.angRes_out):
                imageio.imwrite(output_path + '/view_%.2d_%.2d.png' % (u+1, v+1), np.uint8(lf_rgb_out[u, v, :, :]))

        print('Finished! \n')


if __name__ == '__main__':
    cfg = parse_args()
    demo_test(cfg)