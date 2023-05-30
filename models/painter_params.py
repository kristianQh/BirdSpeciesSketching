import random
import CLIP_.clip as clip
import numpy as np
import pydiffvg
import sketch_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class Painter(torch.nn.Module):
    def __init__(self, args,
                 num_strokes=4,
                 num_segments=4,
                 imsize=224,
                 device=None,
                 target_im=None,
                 im_size=None,
                 mask=None):
        super(Painter, self).__init__()
        self.args = args

        self.im_width, self.im_height = im_size

        self.img_id = args.img_id

        self.output_dir = args.output_dir
        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = args.width
        self.control_points_per_seg = args.control_points_per_seg
        self.opacity_optim = 0
        self.width_optim = args.optim_width
        # training stages, you can train x strokes, then freeze them and train another x strokes etc.
        self.num_stages = args.num_stages
        self.add_random_noise = "noise" in args.augemntations
        self.noise_thresh = args.noise_thresh
        self.softmax_temp = args.softmax_temp

        self.shapes = []
        self.shape_groups = []
        self.device = device
        self.canvas_width, self.canvas_height = imsize, imsize
        self.points_vars = []
        self.color_vars = []
        self.color_vars_threshold = args.color_vars_threshold

        self.path_svg = args.path_svg
        self.strokes_per_stage = self.num_paths
        self.optimize_flag = []

        # attention related for strokes initialisation
        self.attention_init = args.attention_init
        self.attention_dist = args.attention_dist
        self.attention_text = args.attention_text
        self.kp_init = args.kp_init
        self.target_path = args.target
        self.saliency_model = args.saliency_model
        self.xdog_intersec = args.xdog_intersec
        self.mask_object = args.mask_object_attention

        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-B/32", args.device, jit=False)

        self.clip_normalize_transform = transforms.Compose([
            self.clip_preprocess.transforms[0],  # Resize
            self.clip_preprocess.transforms[1],  # CenterCrop
            self.clip_preprocess.transforms[-1],  # Normalize
        ])

        self.text_target = args.text_target  # for clip gradients
        self.saliency_clip_model = args.saliency_clip_model
        # load clip and transform input/target image
        self.genus = args.genus
        self.define_attention_input(target_im)
        self.set_inds_keypoints(target_im) if self.kp_init else None
        self.mask = mask
        self.attention_map = self.set_attention_map() if self.attention_init else None
        self.thresh = self.set_attention_threshold_map(
            target_im) if self.attention_init else None
        self.strokes_counter = 0  # counts the number of calls to "get_path"
        self.epoch = 0
        self.final_epoch = args.num_iter - 1

    def init_image(self, stage=0, path_svg="none"):
        if stage > 0:
            if path_svg != "none":
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = utils.load_svg(
                    path_svg)

            self.optimize_flag = [True for i in range(len(self.shapes))]
            for i, path in enumerate(self.shapes):

                path.points.requires_grad = True
        else:
            num_paths_exists = 0
            if path_svg != "none":
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = utils.load_svg(
                    path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_paths_exists = len(self.shapes)
            for i in range(num_paths_exists, self.num_paths):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                                 fill_color=None,
                                                 stroke_color=stroke_color)
                self.shape_groups.append(path_group)
            self.optimize_flag = [True for i in range(len(self.shapes))]

        #Given a list of shapes, convert them to a linear list of argument, so that we can use it in PyTorch.
        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img
        # utils.imwrite(img.cpu(), '{}/init.png'.format(args.output_dir), gamma=args.gamma, use_wandb=args.use_wandb, wandb_name="init")

    def get_image(self):
        img = self.render_warp()
        # Compose img with white background
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def get_path(self):
        points = []
        self.num_control_points = torch.zeros(
            self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        # Initial control points/indices are from clip attention layers
        if self.kp_init or self.attention_init:
            p0 = self.inds_normalised[self.strokes_counter]
        else:
            p0 = (random.random(), random.random())
        points.append(p0)
        # number of segments for each stroke, each stroke is a bezier curve with 4 control points
        for j in range(self.num_segments):
            radius = 0.05
            # Running for 3 control points
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5),
                      p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height

        path = pydiffvg.Path(num_control_points=self.num_control_points,
                             points=points,
                             stroke_width=torch.tensor(self.width),
                             is_closed=False)
        self.strokes_counter += 1
        return path

    def classify_images(self, processed_img, model, label_emb):
        # encode image to image embedding
        with torch.no_grad():
            img_emb = model.get_image_features(processed_img)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
        img_emb = img_emb.detach().cpu().numpy()
        score = np.dot(img_emb, label_emb.T)
        return score

    def load_best_save(self):

        best_shapes = np.load(
            self.output_dir + "/best_shapes.npy", allow_pickle=True)
        best_shape_groups = np.load(
            self.output_dir + "/best_shapes_groups.npy", allow_pickle=True)
        self.shapes = best_shapes
        self.shape_groups = best_shape_groups

    def clip_abstract(self, abstraction_level):
        sketch = self.get_image().to(self.device)
        x_normed = self.clip_normalize_transform(sketch)
        best_cs = self.classify_images(x_normed, self.hg_model, self.label_emb)

        orig_widths = []
        new_widths = []
        for path in self.shapes:
            orig_widths.append(path.stroke_width.data)
            new_widths.append(path.stroke_width.data)
        np.save(self.output_dir + "/orig_widths.npy", orig_widths)
        np.save(self.output_dir + "/new_widths.npy", new_widths)
        orig_widths = np.load(self.output_dir + "/orig_widths.npy")
        new_widths = np.load(self.output_dir + "/new_widths.npy")

        for i in range(len(self.shapes)):
            self.shapes[i].stroke_width.data.fill_(0.0)
            _render = pydiffvg.RenderFunction.apply
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
            new_sketch = _render(self.canvas_width,  # width
                                 self.canvas_height,  # height
                                 2,   # num_samples_x
                                 2,   # num_samples_y
                                 0,   # seed
                                 None,
                                 *scene_args)
            opacity = new_sketch[:, :, 3:4]
            new_sketch = opacity * new_sketch[:, :, :3] + torch.ones(
                new_sketch.shape[0], new_sketch.shape[1], 3, device=self.device) * (1 - opacity)
            new_sketch = new_sketch[:, :, :3]
            # Convert img from HWC to NCHW
            new_sketch = new_sketch.unsqueeze(0)
            new_sketch = new_sketch.permute(
                0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
            new_sketch_norm = self.clip_normalize_transform(new_sketch)
            new_sketch_norm.to(self.device)
            cs = self.classify_images(
                new_sketch_norm, self.hg_model, self.label_emb)
            cs_diff = best_cs - cs
            if cs_diff <= abstraction_level:
                best_cs = cs
                new_widths[i] = self.shapes[i].stroke_width.data
                np.save(self.output_dir + "/new_widths.npy", new_widths)
                self.shapes[i].stroke_width.data.fill_(orig_widths[i])
            else:
                self.shapes[i].stroke_width.data.fill_(orig_widths[i])

        new_widths = np.load(self.output_dir + "/new_widths.npy")
        for i in range(len(self.shapes)):
            self.shapes[i].stroke_width.data.fill_(new_widths[i])

    def render_warp(self):
        if self.opacity_optim:
            for group in self.shape_groups:
                # group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
                # NOTE: no alpha channel is always 1, that is we ignore the opacity
                group.stroke_color.data[-1].clamp_(1., 1.)
                # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
        _render = pydiffvg.RenderFunction.apply

        if self.width_optim:
            for path in self.shapes:
                path.stroke_width.data.clamp_(0.5, 3.0)

        # uncomment if you want to add random noise
        if self.add_random_noise:
            if random.random() > self.noise_thresh:
                eps = 0.01 * min(self.canvas_width, self.canvas_height)
                for path in self.shapes:
                    path.points.data.add_(eps * torch.randn_like(path.points))
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = _render(self.canvas_width,  # width
                      self.canvas_height,  # height
                      2,   # num_samples_x
                      2,   # num_samples_y
                      0,   # seed
                      None,
                      *scene_args)
        return img

    def parameters(self):
        self.points_vars = []
        # storkes' location optimization
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.requires_grad = True
                self.points_vars.append(path.points)
        return self.points_vars

    def get_points_parans(self):
        return self.points_vars

    def set_color_parameters(self):
        # for storkes' color optimization (opacity)
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if self.optimize_flag[i]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)
        return self.color_vars

    def set_width_parameters(self):
        self.width_vars = []
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.stroke_width.requires_grad = True
                self.width_vars.append(path.stroke_width)
        return self.width_vars

    def get_color_parameters(self):
        return self.color_vars

    def save_svg(self, output_dir, name):
        pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width,
                          self.canvas_height, self.shapes, self.shape_groups)

    # saliency_clip_model is currently ViT-B/32
    def define_attention_input(self, target_im):
        model, preprocess = clip.load(
            self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([
            preprocess.transforms[-1],
        ])
        self.image_input_attn_clip = data_transforms(target_im).to(self.device)

    def clip_attn(self):
        model, _ = clip.load(
            self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        text_input = clip.tokenize([self.text_target]).to(self.device)
        text_prompts = self.attention_text

        if "RN" in self.saliency_clip_model:
            saliency_layer = "layer4"
            attn_map = gradCAM(
                model.visual,
                self.image_input_attn_clip,
                model.encode_text(text_input).float(),
                getattr(model.visual, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / \
                (attn_map.max() - attn_map.min())

        else:
            attn_maps = interpret(self.image_input_attn_clip,
                                  text_prompts, model, self.genus, device=self.device)

        del model
        return attn_maps

    def set_attention_map(self):
        return self.clip_attn()

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()

    def get_img_id_bbox(self, img_name):
        with open("images.txt") as f:
            for line in f:
                pieces = line.strip().split()
                if pieces[1].split("/")[1] == img_name:
                    img_id = pieces[0]

        with open("bounding_boxes.txt") as f:
            for line in f:
                pieces = line.strip().split()
                if pieces[0] == img_id:
                    data = pieces

        return img_id, data

    def read_part_loc(self, img_id, filename):
        data = []
        with open(filename) as f:
            for line in f:
                pieces = line.strip().split()
                if pieces[0] == img_id:
                    data.append(pieces)
        return data

    def get_kps(self, img_name, target_im):
        img_id, coords = self.get_img_id_bbox(img_name)
        loc_data = self.read_part_loc(
            img_id, f"part_locs.txt")
        kps = []
        for row in loc_data:
            if row[4] == '1':
                x, y = int(float(row[2])), int(float(row[3]))
                # scale according to bounding box
                x = (x - int(coords[1]))
                y = (y - int(coords[2]))
                rescaled_x = int((x / self.im_width) * self.canvas_width)
                rescaled_y = int((y / self.im_height) * self.canvas_height)
                kps.append([rescaled_y, rescaled_x])
        num_kp = len(kps)
        return kps, num_kp

    def set_inds_keypoints(self, target_im):
        kps, num_kp = self.get_kps(self.img_id, target_im)
        self.inds = np.zeros((num_kp, 2))
        for i, kp in enumerate(kps): #TODO recheck this
            self.inds[i] = kp

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()


    def get_attn_inds(self, attention_map, xdog_intersec=False, tau=0.1):
        attn_map = (attention_map - attention_map.min()) / \
            (attention_map.max() - attention_map.min())

        if xdog_intersec:
            xdog = XDoG_()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(
                1, 2, 0).cpu().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map

        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(
            attn_map[attn_map > 0], tau)

        k = self.num_stages * self.num_paths
        inds = np.random.choice(range(
            attn_map.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        inds = np.array(np.unravel_index(inds, attn_map.shape)).T

        return inds, attn_map_soft

    def update_inds(self, distribution, inds):
        prev_idx = 0
        for i in range(len(self.attention_map)-1):
            idx_dist = distribution[i]
            upd_inds, _ = self.get_attn_inds(self.attention_map[i+1], tau=0.1)
            inds[prev_idx:idx_dist+prev_idx] = upd_inds[prev_idx:idx_dist+prev_idx]
            prev_idx = idx_dist+prev_idx

        return inds

    def set_inds_clip(self, target_im):
        #initially set all indices to image relevancy/attention
        self.inds, attn_map_soft = self.get_attn_inds(self.attention_map[0], self.xdog_intersec, self.softmax_temp)

        distribution = self.attention_dist
        self.inds = self.update_inds(distribution, self.inds)

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()

        return attn_map_soft

    def set_attention_threshold_map(self, target_im):
        return self.set_inds_clip(target_im)

    def get_attn(self):
        return self.attention_map

    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds

    def get_mask(self):
        return self.mask

    def set_random_noise(self, epoch):
        if epoch % self.args.save_interval == 0:
            self.add_random_noise = False
        else:
            self.add_random_noise = "noise" in self.args.augemntations


class PainterOptimizer:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr
        self.color_lr = args.color_lr
        self.args = args
        self.optim_color = args.optim_color
        self.optim_width = args.optim_width

    def init_optimizers(self, stage=0):
        if stage == 0:
            self.points_optim = torch.optim.Adam(
                self.renderer.parameters(), lr=self.points_lr)
            if self.optim_color:
                self.color_optim = torch.optim.Adam(
                    self.renderer.set_color_parameters(), lr=self.color_lr)
            if self.optim_width:
                self.width_optim = torch.optim.Adam(
                    self.renderer.set_width_parameters(), lr=0.01)
        else:
            self.points_optim = torch.optim.Adam(
                self.renderer.parameters(), lr=self.points_lr)
            self.optim_color = 1
            self.color_optim = torch.optim.Adam(
                self.renderer.set_color_parameters(), lr=self.color_lr)


    def update_lr(self, counter):
        new_lr = utils.get_epoch_lr(counter, self.args)
        for param_group in self.points_optim.param_groups:
            param_group["lr"] = new_lr

    def zero_grad_(self):
        self.points_optim.zero_grad()
        if self.optim_color:
            self.color_optim.zero_grad()
        if self.optim_width:
            self.width_optim.zero_grad()

    def step_(self, stage=0):
        if stage == 0:
            self.points_optim.step()
            if self.optim_color:
                self.color_optim.step()
            if self.optim_width:
                self.width_optim.step()
        else:
            self.color_optim.step()

    def get_lr(self):
        return self.points_optim.param_groups[0]['lr']


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


def get_output_heat(preds):
    output_heat = torch.sigmoid(preds)
    output_heat = output_heat.unsqueeze(0)
    output_heat = torch.nn.functional.interpolate(
        output_heat, size=224, mode='bilinear')
    output_heat = output_heat.reshape(224, 224).data.cpu().numpy()
    output_heat = (output_heat - output_heat.min()) / \
        (output_heat.max() - output_heat.min())
    return output_heat

def interpret(image, text_prompts, model, genus, device):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images, mode='saliency')
    model.zero_grad()
    image_attn_blocks = list(
        dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens,
                  dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = []  # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach()  # attn_probs shape is 12, 50, 50
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1)  # mean of the 12 something
        cams.append(cam)
        R = R + torch.bmm(cam, R)

    cams_avg = torch.cat(cams)  # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:]  # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    # dimension depends on model usage, e.g. 7 for ViT-B/32, 14 for ViT-B/16, 16 for ViT-L/14
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / \
        (image_relevance.max() - image_relevance.min())

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined")

    # Text attention focus
    input_img = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    inputs = processor(text=text_prompts, images=[
        input_img] * len(text_prompts), padding="max_length", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    preds = outputs.logits.unsqueeze(1)
    output_heat1 = get_output_heat(preds[0])
    output_heat2 = get_output_heat(preds[1])
    output_heat3 = get_output_heat(preds[2])

    return [image_relevance, output_heat1, output_heat2, output_heat3]

# Reference: https://arxiv.org/abs/1610.02391


def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma = 0.98
        self.phi = 200
        self.eps = -0.1
        self.sigma = 0.8
        self.binarize = True

    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0 + (imdiff >= self.eps) * \
            (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff