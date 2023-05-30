import matplotlib.pyplot as plt
from IPython.display import display, SVG
from models.painter_params import Painter, PainterOptimizer
from models.loss import Loss
import sketch_utils as utils
import config
from tqdm.auto import tqdm, trange
from torchvision import models, transforms
from PIL import Image
import wandb
import torch.nn.functional as F
import torch.nn as nn
import torch
import PIL
import numpy as np
import traceback
import time
import sys
import os
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

sys.stdout.flush()


warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_renderer(args, target_im=None, im_size=None, mask=None, hg_model=None, label_emb=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       im_size=im_size,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = PIL.Image.open(args.target)
    im_size = target.size
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_, mask, im_size


def main(args):
    split_dir = args.output_dir.split("/")
    im_name = split_dir[-2]
    inp = "92_64s_bw/" + im_name + "/" + split_dir[-1] + "/" + "best_iter.svg"

    inputs, mask, im_size = get_target(args)
    loss_func = Loss(args)
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    renderer = load_renderer(args, inputs, im_size,
                             mask)

    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss, best_fc_loss, best_num_strokes = 100, 100, args.num_paths
    best_iter, best_iter_fc = 0, 0
    min_delta = 1e-7
    terminate = False

    renderer.set_random_noise(0)
    renderer.init_image(stage=0, path_svg=inp)
    renderer.save_svg(
        f"{args.output_dir}/svg_logs", f"init_svg")  # this is the inital random strokes
    optimizer.init_optimizers()

    # not using tdqm for jupyter demo
    args.display = args.display
    if args.display:
        epoch_range = range(args.num_iter)
    else:
        epoch_range = tqdm(range(args.num_iter))

    with torch.no_grad():
        init_sketches = renderer.get_image("init").to(args.device)
        renderer.save_svg(
            f"{args.output_dir}", f"init")

    for epoch in epoch_range:
        if not args.display:
            epoch_range.refresh()
        start = time.time()
        optimizer.zero_grad_()
        sketches = renderer.get_image().to(args.device)
        losses_dict_weighted, loss_dict_norm = loss_func(sketches, inputs, counter, renderer.get_widths(
        ), renderer, optimizer, mode="train", width_opt=renderer.width_optim)
        loss = sum(list(losses_dict_weighted.values()))
        loss.backward()
        optimizer.step_()

        if epoch % 20 == 0:
            utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                             use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")

        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                losses_dict_weighted_eval, losses_dict_norm_eval = loss_func(
                    sketches, inputs, counter, renderer.get_widths(), renderer, optimizer, mode="eval", width_opt=renderer.width_optim)
                loss_eval = sum(list(losses_dict_weighted_eval.values()))
                configs_to_save["loss_eval"].append(loss_eval.item())
                if "num_strokes" not in configs_to_save.keys():
                    configs_to_save["num_strokes"] = []
                configs_to_save["num_strokes"].append(
                    renderer.get_strokes_count())
                for k in losses_dict_norm_eval.keys():
                    original_name, gradnorm_name, final_name = k + \
                        "_original_eval", k + "_gradnorm_eval", k + "_final_eval"
                    if original_name not in configs_to_save.keys():
                        configs_to_save[original_name] = []
                    if gradnorm_name not in configs_to_save.keys():
                        configs_to_save[gradnorm_name] = []
                    if final_name not in configs_to_save.keys():
                        configs_to_save[final_name] = []

                    configs_to_save[gradnorm_name].append(
                        losses_dict_norm_eval[k].item())
                    if k in losses_dict_weighted_eval.keys():
                        configs_to_save[final_name].append(
                            losses_dict_weighted_eval[k].item())

                cur_delta = loss_eval.item() - best_loss
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = epoch
                        best_num_strokes = renderer.get_strokes_count()
                        terminate = False

                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_loss_fc"] = best_fc_loss
                    wandb.run.summary["num_strokes"] = renderer.get_strokes_count(
                    )
                    wandb_dict = {"delta": cur_delta,
                                  "loss_eval": loss_eval.item()}
                    for k in losses_dict_weighted_eval.keys():
                        wandb_dict[k +
                                   "_eval"] = losses_dict_weighted_eval[k].item()
                    wandb.log(wandb_dict, step=counter)

        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                             args.use_wandb, "{}/{}.jpg".format(
                                 args.output_dir, "attention_map"),
                             args.saliency_model, args.display_logs)

        counter += 1
    np.save(f"{args.output_dir}/num_strokes.npy",
            renderer.get_strokes_count().cpu())
    return configs_to_save


if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()
