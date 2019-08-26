#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import argparse
import os
import time
from multiprocessing import Queue

import cv2
import image2pipe as image2pipe
import numpy as np
import torch
import torch.nn as nn

from fastdvdnet import denoise_seq_fastdvdnet
from models import FastDVDnet
from utils import variable_to_cv2_image, remove_dataparallel_wrapper, get_imagenames, preprocess_img, batch_psnr

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


def yield_from_dir(in_dir):
    files = get_imagenames(in_dir)
    for fn, fpath in enumerate(files):
        if not args.gray:
            # Open image as a CxHxW torch.Tensor
            img = cv2.imread(fpath)
            # from HxWxC to CxHxW, RGB image
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        else:
            # from HxWxC to  CxHxW grayscale image (C=1)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

        img, expanded_h, expanded_w = preprocess_img(img, expand_if_needed=False, expand_axis0=False)
        yield fpath, img


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
    parser.add_argument("--model_file", type=str, default="./model.pth",
                        help='path to model of the pretrained denoiser')
    parser.add_argument("--read_path", required=True, type=str, help='input video file')
    parser.add_argument("--save_path", required=True, type=str, help='output video file')

    parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
    parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
    parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
    parser.add_argument("--gray", action='store_true',
                        help='perform denoising of grayscale images instead of RGB')

    args = parser.parse_args()

    # Normalize noises ot [0, 1]
    args.noise_sigma /= 255.

    print(args)

    # if args.read_path is None or args.save_path is None:
    #     parser.print_usage()
    #     sys.exit(1)

    # use CUDA?
    args.cuda = not args.no_gpu and torch.cuda.is_available()

    frames_q = None
    if args.read_path.find("://") > -1:
        print("Decoding input video %s" % args.read_path)

        frames_q = Queue(maxsize=NUM_IN_FR_EXT * 5)
        if args.read_path.find(":/") > -1:
            probe = image2pipe.ffprobe(args.read_path)
            decoder = image2pipe.images_from_url(frames_q, args.read_path, scale=None)
            decoder.start()

    # If save_path does not exist, create it
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Sets data type according to CPU or GPU modes
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create models
    print('Loading model ...')
    model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)
    print('OK')

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
        if frames_q:
            imgs = image2pipe.utils.yield_from_queue(frames_q)
        else:
            # Get ordered list of filenames
            print("\tOpen sequence in folder: ", args.read_path)
            imgs = yield_from_dir(args.read_path)

        seq_list = []
        seq_outnames = []

        for fn_or_fpath, img in imgs:
            if type(fn_or_fpath) is int:
                fpath = "%06d.png" % fn_or_fpath
                # from HxWxC to CxHxW, RGB image
                img = img.transpose(2, 0, 1)
                img, expanded_h, expanded_w = preprocess_img(img, expand_if_needed=False, expand_axis0=False)
            else:
                fpath = fn_or_fpath

            print("Load img:", fpath, img.shape)

            seq_list.append(img)
            seq_outnames.append(os.path.basename(fpath))
            seq = np.stack(seq_list, axis=0)
            # return seq, expanded_h, expanded_w

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
