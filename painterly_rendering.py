from models.painter_params import Painter, PainterOptimizer
from models.loss import Loss
from tqdm.auto import tqdm
from torchvision import transforms
import sketch_utils as utils
import config
import wandb
import torch
import PIL
import numpy as np
import traceback
import sys
import os
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_renderer(args, target_im=None, im_size=None, mask=None):
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
    inputs, mask, im_size = get_target(args)
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    # Used to setup Painter class for attention map, initial points, etc.
    renderer = load_renderer(args, inputs, im_size,
                            mask)
    optimizer = PainterOptimizer(args, renderer)

    def run_stage(stage=0, path_svg="none", num_iters=args.num_iter):
        
        loss_func = Loss(args)

        counter = 0
        configs_to_save = {"loss_eval": []}
        best_loss, best_fc_loss = 100, 100
        best_iter = 0
        min_delta = 1e-5
        terminate = False

        renderer.set_random_noise(0)
        renderer.init_image(stage, path_svg)
        optimizer.init_optimizers(stage)

        # not using tdqm for jupyter demo
        if args.display:
            epoch_range = range(num_iters)
        else:
            epoch_range = tqdm(range(num_iters))

        for epoch in epoch_range:
            if not args.display:
                epoch_range.refresh()
            renderer.set_random_noise(epoch)
            if args.lr_scheduler:
                optimizer.update_lr(counter)

            optimizer.zero_grad_()
            # Forward pass
            # Render image, compose img with white background, NHWC -> NCHW
            sketches = renderer.get_image().to(args.device)
            losses_dict = loss_func(
                sketches, inputs.detach(),
                renderer.get_color_parameters(), renderer,
                counter, optimizer
            )
            loss = sum(list(losses_dict.values()))

            # Propogate gradients
            loss.backward()
            # Take gradient descent step
            optimizer.step_(stage)

            # Logging, intermediate renders and results
            if epoch % args.save_interval == 0:
                utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                                use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
                renderer.save_svg(
                    f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
            if epoch % args.eval_interval == 0:
                with torch.no_grad():
                    losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(
                    ), renderer.get_points_parans(), counter, optimizer, mode="eval")
                    loss_eval = sum(list(losses_dict.values()))
                    configs_to_save["loss_eval"].append(loss_eval.item())
                    for k in losses_dict_eval.keys():
                        if k not in configs_to_save.keys():
                            configs_to_save[k] = []
                        configs_to_save[k].append(losses_dict_eval[k].item())
                    if args.clip_fc_loss_weight:
                        if losses_dict_eval["fc"].item() < best_fc_loss:
                            best_fc_loss = losses_dict_eval["fc"].item(
                            ) / args.clip_fc_loss_weight
                            best_iter_fc = epoch

                    cur_delta = loss_eval.item() - best_loss
                    # Save best iteration result
                    if abs(cur_delta) > min_delta:
                        if cur_delta < 0:
                            best_loss = loss_eval.item()
                            best_iter = epoch
                            terminate = False
                            utils.plot_batch(
                                inputs, sketches, args.output_dir, counter, use_wandb=args.use_wandb, title="best_iter.jpg")
                            renderer.save_svg(args.output_dir, "best_iter")
                            if epoch > 20:
                                testit = renderer.shapes
                                np.save(args.output_dir +
                                        "/best_shapes.npy", testit)
                                testit2 = renderer.shape_groups
                                np.save(args.output_dir +
                                        "/best_shapes_groups.npy", testit2)

                    if args.use_wandb:
                        wandb.run.summary["best_loss"] = best_loss
                        wandb.run.summary["best_loss_fc"] = best_fc_loss
                        wandb_dict = {"delta": cur_delta,
                                    "loss_eval": loss_eval.item()}
                        for k in losses_dict_eval.keys():
                            wandb_dict[k + "_eval"] = losses_dict_eval[k].item()
                        wandb.log(wandb_dict, step=counter)

                    if abs(cur_delta) <= min_delta:
                        if terminate:
                            break
                        terminate = True

            if counter == 0:
                utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                                args.use_wandb, "{}/{}.eps".format(
                                    args.output_dir, "attention_map"),
                                args.saliency_model, args.attention_text)

            if args.use_wandb:
                wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
                for k in losses_dict.keys():
                    wandb_dict[k] = losses_dict[k].item()
                wandb.log(wandb_dict, step=counter)

            counter += 1

        output_name = "stage" + stage + "_best_iter.svg"
        path_svg = os.path.join(args.output_dir, "best_iter.svg")
        utils.log_sketch_summary_final(
            path_svg, args.use_wandb, args.device, best_iter, best_loss, "best total")

        return configs_to_save

    configs_to_save = run_stage()

    if args.stage2_color:
        args.clip_conv_loss = 0
        args.percep_loss = "l2"
        run_stage(stage=1, path_svg=f"{args.output_dir}/stage0_best_iter.svg", num_iters=args.color_iters)

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
