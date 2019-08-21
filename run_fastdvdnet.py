#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from fastdvdnet import denoise_seq_fastdvdnet
from models import FastDVDnet
from utils import variable_to_cv2_image, remove_dataparallel_wrapper, get_imagenames, open_image, batch_psnr

NUM_IN_FR_EXT = 5  # temporal size of patch
MC_ALGO = 'DeepFlow'  # motion estimation algorithm
OUTIMGEXT = '.png'  # output images format


def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
    """Saves the denoised and noisy sequences under save_dir
    """
    seq_len = seqnoisy.size()[0]
    for idx in range(seq_len):
        # Build Outname
        fext = OUTIMGEXT
        noisy_name = os.path.join(save_dir,
                                  ('n{}_{}').format(sigmaval, idx) + fext)
        if len(suffix) == 0:
            out_name = os.path.join(save_dir,
                                    ('n{}_FastDVDnet_{}').format(sigmaval, idx) + fext)
        else:
            out_name = os.path.join(save_dir,
                                    ('n{}_FastDVDnet_{}_{}').format(sigmaval, suffix, idx) + fext)

        # Save result
        if save_noisy:
            noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
            cv2.imwrite(noisy_name, noisyimg)

        outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
        cv2.imwrite(out_name, outimg)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
    parser.add_argument("--model_file", type=str,
                        default="./model.pth",
                        help='path to model of the pretrained denoiser')
    parser.add_argument("--read_path", type=str, default="./data/rgb/Kodak24",
                        help='path to sequence to denoise')
    parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
    parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
    parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
    parser.add_argument("--save_path", type=str, default='./results',
                        help='where to save outputs as png')
    parser.add_argument("--gray", action='store_true',
                        help='perform denoising of grayscale images instead of RGB')

    argspar = parser.parse_args()
    # Normalize noises ot [0, 1]
    argspar.noise_sigma /= 255.

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    args = argspar

    # If save_path does not exist, create it
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Sets data type according to CPU or GPU modes
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create models
    print('Loading models ...')
    model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

    # Load saved weights
    state_temp_dict = torch.load(args.model_file)
    if args.cuda:
        device_ids = [0]
        model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
    else:
        # CPU mode: remove the DataParallel wrapper
        state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
    model_temp.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model_temp.eval()
    start_time = time.time()

    with torch.no_grad():
        # Get ordered list of filenames
        files = get_imagenames(args.read_path)

        seq_list = []
        seq_outnames = []
        print("\tOpen sequence in folder: ", args.read_path)
        for fpath in files:
            img, expanded_h, expanded_w = open_image(fpath,
                                                     gray_mode=args.gray,
                                                     expand_if_needed=False,
                                                     expand_axis0=False)
            seq_list.append(img)
            seq_outnames.append(os.path.basename(fpath))
            seq = np.stack(seq_list, axis=0)
            # return seq, expanded_h, expanded_w
            print("Load img:", fpath)

            if len(seq_list) == NUM_IN_FR_EXT:
                print("Infer batch ...")

                seq = torch.from_numpy(seq).to(device)
                seq_time = time.time()

                # Add noise
                noise = torch.empty_like(seq).normal_(mean=0, std=args.noise_sigma).to(device)
                seqn = seq + noise
                noisestd = torch.FloatTensor([args.noise_sigma]).to(device)

                denframes = denoise_seq_fastdvdnet(seq=seq,
                                                   noise_std=noisestd,
                                                   temp_psz=NUM_IN_FR_EXT,
                                                   model_temporal=model_temp)

                # Compute PSNR and log it
                stop_time = time.time()
                psnr = batch_psnr(denframes, seq, 1.)
                psnr_noisy = batch_psnr(seqn.squeeze(), seq, 1.)
                loadtime = (seq_time - start_time)
                runtime = (stop_time - seq_time)
                seq_length = seq.size()[0]

                print("\tDenoised {} frames in {:.3f}s, loaded seq in {:.3f}s".
                      format(seq_length, runtime, loadtime))
                print("\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy, psnr))

                # Save outputs
                seq_len = len(seq_list)
                for idx in range(seq_len):
                    out_name = os.path.join(args.save_path, seq_outnames[idx])
                    print("Saving %s" % out_name)
                    outimg = variable_to_cv2_image(denframes[idx].unsqueeze(dim=0))
                    cv2.imwrite(out_name, outimg)

                seq_list = []
                seq_outnames = []
