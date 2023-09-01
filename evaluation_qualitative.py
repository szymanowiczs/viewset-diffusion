import argparse
import sys
from evaluation.generator import Generator

import torch

from utils import set_seed
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

"""
Generates qualitative results: denoises a multi-view image set,
saves and produces renderings from the conditioning, noised and orbit
viewpoints. Choose which examples to reconstruct using samples_to_generate .
Setting samples_to_generate = [0, 1, 1, 4] will reconstruct examples
0, 1 twice and 4 from the selected dataset. You can also change which
set you want to use: train, val or test.
"""

def main(args):

    set_seed(args.seed)

    split = 'test'
    if args.dataset_name == "minens":
        # Choose which samples to generate / reconstruct - choose any indices in [0, 10000]
        samples_to_generate = [i*2 for i in range(args.seed*16, args.seed*16 + 16)]
        if args.N_clean == 2:
            samples_to_generate = [s // 2 for s in samples_to_generate]

        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(2))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")  

    elif args.dataset_name == "srn":
        random_order = torch.randperm(704)
        samples_to_generate = random_order[args.seed*16: args.seed*16 + 16]

        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(0))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu") 

    elif args.dataset_name == "hydrant":
        samples_to_generate = torch.randperm(4990)[:16]

        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(0))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu") 

    elif args.dataset_name == "teddybear":
        # each of these samples comes from a different sequence
        samples_to_generate = [0, 202, 403, 605, 807, 1009, 1211, 1413, 1615, 1816, 2018, 2219, 
           2421, 2623, 2825, 3027, 3229, 3431, 3632, 3834, 4036, 4238, 4440, 
           4642, 4844, 5045, 5244, 5446, 5648, 5850, 6052, 6254, 6456, 6658, 
           6860, 7062, 7264, 7466, 7668, 7868, 8070, 8272, 8474, 8676, 8878, 
           9080, 9282, 9484, 9686, 9888, 10090, 10292, 10494, 10696, 10898, 
           11100][args.seed*16: args.seed*16 + 16]
        if args.N_clean == 2:
            samples_to_generate = [s // 2 for s in samples_to_generate]

        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(0))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu") 

    elif args.dataset_name == "plant":
        samples_to_generate = [0, 202, 404, 606, 808, 1010, 1211, 1413, 1615, 1816, 2014, 2216, 
           2418, 2620, 2822, 3024, 3226, 3428, 3630, 3832, 4034, 4236, 4438, 
           4640, 4842, 5044, 5246, 5448, 5650, 5852, 6054, 6256, 6458, 6658, 
           6860, 7062, 7264, 7466, 7668, 7870, 8072, 8274, 8476, 8678, 8880, 
           9082, 9284, 9486, 9688, 9890, 10092, 10289, 10491, 10693, 10895, 
           11097, 11299, 11501, 11703, 11905, 12107, 12309, 12511, 12713, 
           12915, 13117, 13319, 13521, 13723, 13925, 14127, 14329, 14531, 
           14733, 14935, 15137, 15339, 15541, 15743][args.seed*16: args.seed*16 + 16]
        split = 'test'
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(0))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu") 


    elif args.dataset_name == "vase":
        samples_to_generate = [0, 202, 404, 606, 808, 1010, 1202, 1404, 1606, 
        1808, 2010, 2212, 2414, 2616, 2818, 3018, 3220, 
        3422, 3624, 3826, 4028, 4229, 4430, 4632, 4834, 
        5036, 5238, 5440, 5642, 5844, 6046, 6240, 6442, 
        6642, 6844, 7046, 7248, 7450, 7652, 7854, 8056, 
        8258, 8460, 8662, 8863, 9065, 9267, 9469, 9671, 
        9873, 10043, 10245, 10447, 10648, 10850, 11052, 
        11254, 11455, 11657, 11856, 12058, 12260, 12462, 
        12664, 12866, 13068, 13270, 13472, 13674, 13876, 
        14078, 14280, 14482, 14684, 14886, 15088, 15290, 
        15492, 15694, 15896, 16098, 16300, 16502, 16704, 
        16906, 17108, 17310][args.seed*16: args.seed*16 + 16]
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(0))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu") 

    generator = Generator(args.experiment_path, device, seed=args.seed,
                          deterministic = args.N_noisy==0)

    samples, gt_data = generator.generate_samples(
                                         samples_to_generate,
                                         N_clean=args.N_clean,
                                         N_noisy=args.N_noisy,
                                         cf_guidance=args.cf_guidance,
                                         split=split)

    generator.reshape_and_save_samples(samples, gt_data, args.N_clean, 
                                       args.N_noisy, split, args.cf_guidance,
                                       seed=args.seed)

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="Path to experiment folder")
    parser.add_argument("--dataset_name", type=str, help="One of [minens, srn, hydrant, plant, teddybear, vase]")
    parser.add_argument("--N_clean", type=int, help="Number of clean conditioning images")
    parser.add_argument("--N_noisy", type=int, help="Number of noisy images in viewset, at least 1")
    parser.add_argument("--cf_guidance", type=float, help="Strength of classifier-free guidance")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args(args)

    return args

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)