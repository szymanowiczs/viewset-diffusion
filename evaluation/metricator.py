import torch
import torch.nn as nn

import torchvision.transforms as transforms

import lpips
import skimage.metrics

class Metricator():
    def __init__(self):
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')

    @torch.no_grad()
    def measure_metrics(self, output_all_samples, gt_data,
                        N_noisy, N_clean, non_one_length):
        """
        args:
            output_all_samples: renders of all samples from generator. Should contain
                N_clean renders from conditioning cameras, N_noisy from denoised views and 
                gt_data["test_images"].shape[1] renders from testing viewpoints
            gt_data: ground truth images from testing viewpoints
            N_noisy: number of noisy images in viewset
            N_clean: number of clean conditioning images in viewset
            non_one_length: whether the testing sequence is of length 1. True for CO3D and Minens,
                False for SRN Cars
        """
        if N_noisy != 0:
            assert output_all_samples.shape[1] == N_clean + N_noisy + gt_data["test_imgs"].shape[1], \
                "Wrong number of frames"
            N_test_start = N_clean + N_noisy
        else:
            raise NotImplementedError # check if this is corrected and remove if needed
            if non_one_length:
                rendered_non_test = N_clean 
            else:
                rendered_non_test = N_clean + 1
            assert output_all_samples.shape[1] == rendered_non_test + gt_data["test_imgs"].shape[1], \
                "Wrong number of frames"
            N_test_start = rendered_non_test
        if non_one_length:
            assert gt_data["test_imgs"].shape[1] == 251 - N_clean, "Unexpected number of gt_frames"
        else:
            assert gt_data["test_imgs"].shape[1] == 1, "Unexpected number of gt_frames"
        if output_all_samples.shape[3] != gt_data["test_imgs"].shape[3]:
            print('resizing to {}, images are at {}'.format(gt_data["test_imgs"].shape[3],
                                                            output_all_samples.shape[3]))
            resizing = transforms.Resize(gt_data["test_imgs"].shape[3], 
                                         interpolation=transforms.InterpolationMode.BILINEAR)
        else:
            resizing = nn.Identity()
        output_test_frames = output_all_samples[:, N_test_start:, ...]
        
        l2_criterion = nn.MSELoss(reduction='none')
        psnrs = []
        lpipses = []
        ssims = []
        for sample_in_batch_idx in range(len(output_all_samples)):
            # ================ PSNR measurement ================
            # don't average across frames
            l2_loss = l2_criterion(resizing(output_test_frames[sample_in_batch_idx]), 
                                   gt_data["test_imgs"][sample_in_batch_idx]
                                   ).mean(dim=[1, 2, 3])
            psnr = -10 * torch.log10(l2_loss)
            # average PSNR across frames
            psnrs.append(torch.mean(psnr).item())
            # ================ LPIPS measurement ================ 
            lpips = self.loss_fn_vgg(resizing(output_test_frames[sample_in_batch_idx].cpu()) * 2 - 1,
                                        gt_data["test_imgs"][sample_in_batch_idx].cpu() * 2 - 1
                                        )
            lpipses.append(torch.mean(lpips).item())
            # ================ SSIM measurement =============
            ssim_batch = []
            for view_idx in range(output_test_frames.shape[1]):
                ssim = skimage.metrics.structural_similarity(
                    resizing(output_test_frames[sample_in_batch_idx, view_idx]).detach().cpu().numpy(),
                    gt_data["test_imgs"][sample_in_batch_idx, view_idx].detach().cpu().numpy(),
                    data_range=1,
                    channel_axis=0
                )
                ssim_batch.append(ssim)
            ssims.append(sum(ssim_batch) / len(ssim_batch))

        return psnrs, lpipses, ssims