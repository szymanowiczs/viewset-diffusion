import argparse
import hydra
import os
import sys

from omegaconf import DictConfig, OmegaConf

from evaluation.generator import Generator
from evaluation.metricator import Metricator

import torch
import torchvision.utils as tv_uils
import numpy as np

from utils import set_seed

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups

"""
Runs quantitative evaluation of a selected model. Dataset is inferred from the
model config. Saves results in a series of jsons. Results can be read using
evaluation_read_scores if distributed evaluation is used.
"""
@hydra.main(version_base=None, config_path='configs_eval', config_name="default_config")
def main(cfg: DictConfig):

    split = cfg.eval.split

    set_seed(0)
        
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(torch.cuda.current_device()))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")  

    generator = Generator(cfg.model_path, device, seed=cfg.seed,
                          deterministic=cfg.eval.deterministic)
    # Sets the correct indices for evaluation
    generator.update_dataset(cfg.N_clean, cfg.eval.split, cfg.cf_guidance,
                             with_index_selection=True)
    metricator = Metricator()

    if "minens" in cfg.model_path:
        length = 200
    elif "co3d" in cfg.model_path:
        length = 100
    else:
        length = len(generator.dataset)

    if "srn" in cfg.model_path:
        non_one_length = True
    else:
        non_one_length = False

    chunks = num_to_chunks(length, cfg.eval.n_devices)
    start_idx = sum(chunks[:cfg.eval.device_idx])
    chunk_start = sum(chunks[:cfg.eval.device_idx])
    chunk_end = sum(chunks[:cfg.eval.device_idx+1])

    batches = num_to_groups(chunk_end - start_idx, 
            # a heuristic for how many images will fit on the GPU
            32 // (cfg.N_clean + cfg.N_noisy))

    if "co3d" in cfg.model_path:
        length = 100
        batches = num_to_groups(chunk_end - start_idx, 
            # a heuristic for how many images will fit on the GPU
            100)

    print('evaluating examples {} to {}'.format(start_idx, chunk_end))

    # repeat the following evaluation cfg.eval.n_samples_per_ex times
    # for each sample compute the top PSNR in the first n samples
    # where n = 1, 5, 10, 20

    all_psnrs = []
    all_lpipses = []
    all_ssims = []
    example_ids = []
    psnrs_average_across_samples = []
    ssims_average_across_samples = []
    lpipses_average_across_samples = []

    for batch in batches:
        batch_generated_samples = []
        for ex_idx in range(start_idx, start_idx+batch):
            example_ids.append(generator.dataset.get_example_id(ex_idx))
            for a in all_psnrs, all_lpipses, all_ssims:
                a.append([])
        for sample_idx in range(cfg.eval.n_samples_per_ex):
            generated_samples, gt_data = generator.generate_samples(
                                    [i for i in range(start_idx, start_idx+batch)],
                                    cfg.N_clean,
                                    cfg.N_noisy,
                                    split=split,
                                    use_testing_protocol=True)
            batch_generated_samples.append(generated_samples)
            psnrs, lpipses, ssims = \
                metricator.measure_metrics(generated_samples, 
                                            gt_data,
                                            cfg.N_noisy,
                                            cfg.N_clean,
                                            non_one_length)
            for ex_idx in range(start_idx, start_idx+batch):
                all_psnrs[ex_idx - chunk_start].append(psnrs[ex_idx - start_idx])
                all_lpipses[ex_idx - chunk_start].append(lpipses[ex_idx - start_idx])
                all_ssims[ex_idx - chunk_start].append(ssims[ex_idx - start_idx])

            if sample_idx == 0 and cfg.save_output:
                for ex_idx in range(start_idx, start_idx+batch):
                    example_id = generator.dataset.get_example_id(ex_idx)
                    out_dir_name = os.path.join(os.getcwd(), example_id)
                    assert cfg.N_noisy != 0, "N_noisy must be 0 for saving output"
                    if not os.path.isdir(out_dir_name):
                        os.mkdir(out_dir_name)
                    N_test_start = cfg.N_clean + cfg.N_noisy
                    for rot_idx, output_frame in enumerate(generated_samples[ex_idx - start_idx][N_test_start:]):
                        # save output frames
                        tv_uils.save_image(output_frame,
                            os.path.join(out_dir_name, "{}_out.png".format(rot_idx)),
                            padding=0,n_row=1)
        # Measures the PSNR of an average sample - this will be blurrier but more often
        # than not will give a higher PSNR than average PSNR of a single sample.
        # Shows that PSNR(average(samples)) > average(PSNR(samples)), illustrating that
        # PSNR puts generative models at a disadvantage. 
        average_generated_samples = torch.stack(batch_generated_samples).mean(dim=0, keepdim=False)
        psnr_average_across_samples, ssim_average_across_samples, lpips_average_across_samples = \
            metricator.measure_metrics(average_generated_samples, 
                                        gt_data,
                                        cfg.N_noisy,
                                        cfg.N_clean,
                                        non_one_length)
        for ex_idx in range(start_idx, start_idx+batch):
            psnrs_average_across_samples.append(psnr_average_across_samples[ex_idx - start_idx])
            ssims_average_across_samples.append(ssim_average_across_samples[ex_idx - start_idx])
            lpipses_average_across_samples.append(lpips_average_across_samples[ex_idx - start_idx])

        for top_n in [1, 5, 10, 20, 100]:
            if top_n > len(all_psnrs[0]):
                print("WARNING: did not collect {} samples".format(top_n))
            print("Top {} Batch Max PSNR mean: {}".format(top_n, sum(
                [max(all_psnrs[ex_idx - chunk_start][:top_n]) for ex_idx in range(start_idx, start_idx+batch)]) / batch))
            print("Top {} Batch Average LPIPs mean: {}".format(top_n, sum(
                [sum(all_lpipses[ex_idx - chunk_start][:top_n]) / top_n for ex_idx in range(start_idx, start_idx+batch)]) / batch))
            print("Top {} Batch Max SSIM mean: {}".format(top_n, sum(
                [max(all_ssims[ex_idx - chunk_start][:top_n]) for ex_idx in range(start_idx, start_idx+batch)]) / batch))
        print("PSNR( average( samples ) ): {}".format(sum(psnrs_average_across_samples) / len(psnrs_average_across_samples)))
        print("SSIM( average( samples ) ): {}".format(sum(ssims_average_across_samples) / len(ssims_average_across_samples)))
        print("LPIPS( average ( samples ) ): {}".format(sum(lpipses_average_across_samples) / len(lpipses_average_across_samples)))

        for top_n in [1, 5, 10, 20, 100]:
            if top_n > len(all_psnrs[0]):
                print("WARNING: did not collect {} samples".format(top_n))
            print("Top {} Running Max PSNR mean: {}".format(top_n, sum(
                [max(all_psnrs[ex_idx][:top_n]) for ex_idx in range(len(all_psnrs))]) / len(all_psnrs)))
            print("Top {} Running Average LPIPs mean: {}".format(top_n, sum(
                [sum(all_lpipses[ex_idx][:top_n]) / top_n for ex_idx in range(len(all_lpipses))]) / len(all_lpipses)))
            print("Top {} Running SSIM mean: {}".format(top_n, sum(
                [max(all_ssims[ex_idx][:top_n]) for ex_idx in range(len(all_ssims))]) / len(all_ssims)))

        start_idx += batch

        print("Done {} out of {} examples".format(len(all_psnrs), len(generator.dataset)))

    vis_dir = os.getcwd()
    for top_n in [1, 5, 10, 20, 100]:
        with open(os.path.join(vis_dir, 'scores_{}.txt'.format(top_n)), 'w+') as f:
            for i, output_id in enumerate(example_ids):
                f.write(str(output_id) + ' ' + \
                        str(max(all_psnrs[i][:top_n])) + ' ' + \
                        str(sum(all_lpipses[i][:top_n]) / top_n) + ' ' + \
                        str(max(all_ssims[i][:top_n])) + ' ' + \
                        str(sum(all_psnrs[i][:top_n]) / top_n) + ' ' + \
                        str(min(all_lpipses[i][:top_n])) + ' ' + \
                        str(sum(all_ssims[i][:top_n]) / top_n) + '\n')
    with open(os.path.join(vis_dir, "scores_mean_sample.txt"), 'w+') as f:
        for i, output_id in enumerate(example_ids):
            f.write(str(output_id) + ' ' + \
                    str(psnrs_average_across_samples[i]) +  ' ' + \
                    str(ssims_average_across_samples[i]) +  ' ' + \
                    str(lpipses_average_across_samples[i]) +  ' ' + \
                          '\n')

def num_to_chunks(num, groups):
    remainder = num % groups
    divisor = num // groups
    arr = [divisor] * groups
    if remainder > 0:
        arr[-1] += remainder
    return arr

if __name__ == "__main__":
    main()