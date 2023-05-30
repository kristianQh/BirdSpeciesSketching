import sys
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import multiprocessing as mp
import os
import subprocess as sp
from shutil import copyfile

import numpy as np
import torch
from IPython.display import Image as Image_colab
from IPython.display import display, SVG, clear_output

parser = argparse.ArgumentParser()
parser.add_argument("--target_file", type=str,
                    help="target image file, located in <target_images>")
parser.add_argument("--num_strokes", type=int, default=64,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")
parser.add_argument("--num_iter", type=int, default=1300,
                    help="number of iterations")
parser.add_argument("--fix_scale", type=int, default=0,
                    help="if the target image is not squared, it is recommended to fix the scale")
parser.add_argument("--mask_object", type=int, default=0,
                    help="if the target image contains background, it's better to mask it out")
parser.add_argument("--num_sketches", type=int, default=2,
                    help="it is recommended to draw 3 sketches and automatically chose the best one")
parser.add_argument("--multiprocess", type=int, default=0,
                    help="recommended to use multiprocess if your computer has enough memory")
parser.add_argument('-colab', action='store_true')
parser.add_argument('-cpu', action='store_true')
parser.add_argument('-display', action='store_true')
parser.add_argument('--gpunum', type=int, default=0)

args = parser.parse_args()
multiprocess = not args.colab and args.num_sketches > 1 and args.multiprocess

abs_path = os.path.abspath(os.getcwd())

target = f"../../target_images/{args.target_file}"
assert os.path.isfile(target), f"{target} does not exists!"

# Download U2net weights
if not os.path.isfile(f"../../U2Net_/saved_models/u2net.pth"):
    sp.run(["gdown", "https://drive.google.com/u/0/uc?id=135ED-KO3wCANFKxmeZOkwxo1OD1OmQ89&export=download",
           "-O", "U2Net_/saved_models/"])

test_name = os.path.splitext(args.target_file)[0]
output_dir = f"{abs_path}/output_sketches/{test_name}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_interval = 30
use_gpu = not args.cpu

if not torch.cuda.is_available():
    use_gpu = False
    print("CUDA is not configured with GPU, running with CPU instead.")
    print("Note that this will be very slow, it is recommended to use colab.")

if args.colab:
    print("=" * 50)
    print(f"Processing [{args.target_file}] ...")
    if args.colab or args.display:
        img_ = Image_colab(target)
        display(img_)
        print(f"GPU: {use_gpu}, {torch.cuda.current_device()}")
    print(f"Results will be saved to \n[{output_dir}] ...")
    print("=" * 50)

seeds = list(range(0, args.num_sketches * 1000, 1000))

def run(seed, wandb_name, output_dir, lossess_best_normalised, losses_eval_sum):
    exit_code = sp.run(["python", "painterly_rendering.py", target,
                            "--num_paths", str(args.num_strokes),
                            "--output_dir", output_dir,
                            "--wandb_name", wandb_name,
                            "--num_iter", str(args.num_iter),
                            "--save_interval", str(save_interval),
                            "--seed", str(seed),
                            "--img_id", args.target_file,
                            "--use_gpu", str(int(use_gpu)),
                            "--fix_scale", str(args.fix_scale),
                            "--mask_object", str(args.mask_object),
                            "--mask_object_attention", str(
                                args.mask_object),
                            "--display_logs", str(int(args.colab)),
                            "--display", str(int(args.display))])
    if exit_code.returncode:
        sys.exit(1)

    config = np.load(f"{output_dir}/{wandb_name}/config.npy",
                     allow_pickle=True)[()]
    loss_eval = np.array(config['loss_eval'])
    inds = np.argsort(loss_eval)
    losses_eval_sum[wandb_name] = loss_eval[inds][0]


if __name__ == "__main__":
    mp.set_start_method("spawn")
    exit_codes = []
    manager = mp.Manager()
    # losses that are not normalised, for the regular run (no simp)
    losses_eval_sum = manager.dict()
    # save the best normalised loss (for visual simplification)
    losses_best_normalised = manager.dict()

    print("multiprocess", multiprocess)
    if multiprocess:
        ncpus = 10
        P = mp.Pool(ncpus)  # Generate pool of workers

    for j in range(args.num_sketches):
        seed = seeds[j]
        wandb_name = f"{test_name}_{args.num_strokes}strokes_seed{seed}"
        if multiprocess:
            # run simulation and ISF analysis in each process
            P.apply_async(run, (seed, wandb_name, output_dir,
                          losses_best_normalised, losses_eval_sum))
        else:
            run(seed, wandb_name, output_dir,
                losses_best_normalised, losses_eval_sum)

    if multiprocess:
        P.close()
        P.join()  # start processes