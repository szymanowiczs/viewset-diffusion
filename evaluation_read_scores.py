import argparse
import sys
import os

import numpy as np

def main(path):

    dirs = os.listdir(path)
    # Reads all results from distributed evaluation
    dirs = [os.path.join(path, dir_name) for 
            dir_name in dirs if (os.path.isdir(os.path.join(path, dir_name)) 
                                 and dir_name!=".submitit")]

    dirs = sorted(dirs, key=lambda p: int(os.path.basename(p)))

    for top_ns in [1, 5, 10, 20, 100, "mean_sample"]:
        psnrs = []
        lpipses = []
        ssims = []
        psnrs_avg = []
        lpipses_min = []
        ssims_avg = []
        example_ids = []
        psnrs_mean_samples = []
        print("======== Top {} evaluation ========".format(top_ns))
        for dir_name in dirs:
            with open(os.path.join(dir_name, "scores_{}.txt".format(top_ns)), "r") as f:
                lines = f.readlines()
            for i, l in enumerate(lines):
                if len(l.strip('\n').split(' ')) != 7:
                    example_id, psnr, lpips, ssim = l.strip('\n').split(' ')
                else:
                    example_id, psnr, lpips, ssim, psn_avg, lpips_min, ssim_avg = l.strip('\n').split(' ')
                    lpipses.append(float(lpips))
                    ssims.append(float(ssim))
                    psnrs_avg.append(float(psn_avg))
                    lpipses_min.append(float(lpips_min))
                    ssims_avg.append(float(ssim_avg))
                example_ids.append(example_id)
                psnrs.append(float(psnr))
        print("Got {} in total.".format(len(example_ids)))

        print("Max PSNR: {}".format(sum(psnrs) / len(psnrs)))
        print("Avg LPIPS: {}".format(sum(lpipses) / len(lpipses)))
        print("Max SSIM: {}".format(sum(ssims) / len(ssims)))

        print("Avg PSNR: {}".format(sum(psnrs_avg) / len(psnrs)))
        print("Min LPIPS: {}".format(sum(lpipses_min) / len(lpipses)))
        print("Avg SSIM: {}".format(sum(ssims_avg) / len(ssims)))

        # Idxs should be correct because jobs are named in the order of chunks
        # sorting dirs will mean that indices are in order
        lowest_psnrs = np.argsort(psnrs)[:16]
        highest_lpipses = np.argsort(lpipses)[-16:]
        lowest_ssims = np.argsort(ssims)[:16]
        print("Idxs with the lowest psnr: {}".format(lowest_psnrs))
        print("Idxs with the highest LIPIS: {}".format(highest_lpipses))
        print("Idxs with the lowest ssim: {}".format(lowest_ssims))

    return None

if __name__=="__main__":
    path = sys.argv[1]
    main(path)