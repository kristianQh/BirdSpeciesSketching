import random
import sys
sys.path.append('../..')
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
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
        self.opacity_optim = 1
        self.gumbel_temp = 0.2
        # training stages, you can train x strokes, then freeze them and train another x strokes etc.
        self.num_stages = args.num_stages
        self.add_random_noise = "noise" in args.augemntations
        self.noise_thresh = args.noise_thresh
        self.softmax_temp = args.softmax_temp
        self.optimize_points = args.optimize_points
        self.optimize_points_global = args.optimize_points

        self.shapes = []
        self.shape_groups = []
        self.device = device
        self.canvas_width, self.canvas_height = imsize, imsize
        self.points_vars = []
        self.points_init = []  # for mlp training
        self.color_vars = []
        self.color_vars_threshold = args.color_vars_threshold

        self.path_svg = args.path_svg
        self.strokes_per_stage = self.num_paths
        self.optimize_flag = []

        # attention related for strokes initialisation
        self.attention_init = args.attention_init
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
        self.mlp_train = args.mlp_train
        self.width_optim = args.width_optim
        self.width_optim_global = args.width_optim

        self.mlp_points_weights_path = "none"

        if self.width_optim:
            self.init_widths = torch.ones((self.num_paths)).to(device) * 1.5
            self.mlp_width = WidthMLP(num_strokes=self.num_paths, num_cp=self.control_points_per_seg, width_optim=self.width_optim).to(device)
            self.mlp_width_weights_path = "none"
            self.mlp_width_weight_init()

        self.mlp = MLP(num_strokes=self.num_paths, num_cp=self.control_points_per_seg, width_optim=self.width_optim).to(device)
        self.mlp_points_weight_init()
        self.out_of_canvas_mask = torch.ones((self.num_paths)).to(self.device)

    def mlp_points_weight_init(self):
            if self.mlp_points_weights_path != "none":
                checkpoint = torch.load(self.mlp_points_weights_path)
                self.mlp.load_state_dict(checkpoint['model_state_dict'])
                print("mlp checkpoint loaded from ", self.mlp_points_weights_path)


    def mlp_width_weight_init(self):
        if self.mlp_width_weights_path == "none":
            self.mlp_width.apply(init_weights)
        else:
            checkpoint = torch.load(self.mlp_width_weights_path)
            self.mlp_width.load_state_dict(checkpoint['model_state_dict'])
            print("mlp checkpoint loaded from ", self.mlp_width_weights_path)


    def init_image(self, stage=0, path_svg="none"):
        if stage > 0:
            self.optimize_flag = [False for i in range(len(self.shapes))]
            for i in range(self.strokes_per_stage):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)
                self.optimize_flag.append(True)

        else:
            num_paths_exists = 0
            if path_svg != "none":
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = utils.load_svg(path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_paths_exists = len(self.shapes)
                # if self.width_optim:
                for path in self.shapes:
                    self.points_init.append(path.points)

            for i in range(num_paths_exists, self.num_paths):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)        
            self.optimize_flag = [True for i in range(len(self.shapes))]

    def get_image(self, mode="train"):
        if self.mlp_train:
            img = self.mlp_pass(mode)
        else:
            img = self.render_warp(mode)
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img
    
    def mlp_pass(self, mode, eps=1e-4):
        """
        update self.shapes etc through mlp pass instead of directly (should be updated with the optimizer as well).
        """
        if self.optimize_points_global:
            points_vars = self.points_init
            # reshape and normalise to [-1,1] range
            points_vars = torch.stack(points_vars).unsqueeze(0).to(self.device)
            points_vars = points_vars / self.canvas_width
            points_vars = 2 * points_vars - 1
            if self.optimize_points:
                points = self.mlp(points_vars)
            else:
                with torch.no_grad():
                    points = self.mlp(points_vars)
            
        else:
            points = torch.stack(self.points_init).unsqueeze(0).to(self.device)

        if self.width_optim and mode != "init": #first iter use just the location mlp
            widths_  = self.mlp_width(self.init_widths).clamp(min=1e-8)
            mask_flipped = (1 - widths_).clamp(min=1e-8)
            v = torch.stack((torch.log(widths_), torch.log(mask_flipped)), dim=-1)
            hard_mask = torch.nn.functional.gumbel_softmax(v, self.gumbel_temp, False)
            self.stroke_probs = hard_mask[:, 0] * self.out_of_canvas_mask
            self.widths = self.stroke_probs * self.init_widths            
        
        # normalize back to canvas size [0, 224] and reshape
        all_points = 0.5 * (points + 1.0) * self.canvas_width
        all_points = all_points + eps * torch.randn_like(all_points)
        all_points = all_points.reshape((-1, self.num_paths, self.control_points_per_seg, 2))

        if self.width_optim_global and not self.width_optim:
            self.widths = self.widths.detach()
            # all_points = all_points.detach()
        
        # define new primitives to render
        shapes = []
        shape_groups = []
        for p in range(self.num_paths):
            width = torch.tensor(self.width)
            if self.width_optim_global and mode != "init":
                width = self.widths[p]
            path = pydiffvg.Path(
                num_control_points=self.num_control_points, points=all_points[:,p].reshape((-1,2)),
                stroke_width=width, is_closed=False)
            if mode == "init":
                # do once at the begining, define a mask for strokes that are outside the canvas
                is_in_canvas_ = self.is_in_canvas(self.canvas_width, self.canvas_height, path)
                if not is_in_canvas_:
                    self.out_of_canvas_mask[p] = 0
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=torch.tensor([0,0,0,1]))
            shape_groups.append(path_group)
        
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, shapes, shape_groups)
        img = _render(self.canvas_width, # width
                    self.canvas_height, # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    0,   # seed
                    None,
                    *scene_args)
        self.shapes = shapes.copy()
        self.shape_groups = shape_groups.copy()
        return img
    
    def get_path(self):
        points = []
        p0 = self.inds_normalised[self.strokes_counter] if self.attention_init else (random.random(), random.random())
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        self.points_init.append(points)

        path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.width).to(self.device),
                                is_closed = False)
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

    def render_warp(self, mode):
        if not self.mlp_train:
            if self.opacity_optim:
                for group in self.shape_groups:
                    group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
                    group.stroke_color.data[-1].clamp_(0., 1.) # opacity
                    # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
            # uncomment if you want to add random noise
            if self.add_random_noise:
                if random.random() > self.noise_thresh:
                    eps = 0.01 * min(self.canvas_width, self.canvas_height)
                    for path in self.shapes:
                        path.points.data.add_(eps * torch.randn_like(path.points))
        
        if self.width_optim and mode != "init":
            widths_  = self.mlp_width(self.init_widths).clamp(min=1e-8)
            mask_flipped = 1 - widths_
            v = torch.stack((torch.log(widths_), torch.log(mask_flipped)), dim=-1)
            hard_mask = torch.nn.functional.gumbel_softmax(v, self.gumbel_temp, False)
            self.stroke_probs = hard_mask[:, 0] * self.out_of_canvas_mask
            self.widths = self.stroke_probs * self.init_widths  
        
        if self.optimize_points:
            _render = pydiffvg.RenderFunction.apply
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
            img = _render(self.canvas_width, # width
                        self.canvas_height, # height
                        2,   # num_samples_x
                        2,   # num_samples_y
                        0,   # seed
                        None,
                        *scene_args)
        else:
            points = torch.stack(self.points_init).unsqueeze(0).to(self.device)
            shapes = []
            shape_groups = []
            for p in range(self.num_paths):
                width = torch.tensor(self.width)
                if self.width_optim:
                    width = self.widths[p]
                path = pydiffvg.Path(
                    num_control_points=self.num_control_points, points=points[:,p].reshape((-1,2)),
                    stroke_width=width, is_closed=False)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=None,
                    stroke_color=torch.tensor([0,0,0,1]))
                shape_groups.append(path_group)
            
            _render = pydiffvg.RenderFunction.apply
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                self.canvas_width, self.canvas_height, shapes, shape_groups)
            img = _render(self.canvas_width, # width
                        self.canvas_height, # height
                        2,   # num_samples_x
                        2,   # num_samples_y
                        0,   # seed
                        None,
                        *scene_args)
            self.shapes = shapes.copy()
            self.shape_groups = shape_groups.copy()

        return img

    def parameters(self):
        if self.optimize_points:
            if self.mlp_train:
                self.points_vars = self.mlp.parameters()
            else:
                self.points_vars = []
                # storkes' location optimization
                for i, path in enumerate(self.shapes):
                    if self.optimize_flag[i]:
                        path.points.requires_grad = True
                        self.points_vars.append(path.points)
                        self.optimize_flag[i] = False

        if self.width_optim:
            return self.points_vars, self.mlp_width.parameters()    
        return self.points_vars

    def get_mlp(self):
        return self.mlp

    def get_width_mlp(self):
        if self.width_optim_global:
            return self.mlp_width
        else:
            return None

    def get_points_parans(self):
        return dict(list(self.mlp.named_parameters()))
        # return self.points_vars

    def set_color_parameters(self):
        print("color optimizer running")
        # for strokes' color optimization (opacity)
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            # if self.optimize_flag[i]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)
        return self.color_vars

    def set_width_parameters(self):
        self.width_vars = []
        for i, path in enumerate(self.shapes):
            # if self.optimize_flag[i]:
                path.stroke_width.requires_grad = True
                self.width_vars.append(path.stroke_width)
        return self.width_vars

    def get_color_parameters(self):
        return self.color_vars

    def get_widths(self):
        if self.width_optim_global:
            return self.stroke_probs
        return None

    def get_strokes_in_canvas_count(self):
        return self.out_of_canvas_mask.sum()

    def get_strokes_count(self):
        if self.width_optim_global:
            with torch.no_grad():
                return torch.sum(self.stroke_probs)
        return self.num_paths

    def is_in_canvas(self, canvas_width, canvas_height, path):
        shapes, shape_groups = [], []
        stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                            fill_color = None,
                                            stroke_color = stroke_color)
        shape_groups.append(path_group) 
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        img = _render(canvas_width,  # width
                    canvas_height,  # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    0,   # seed
                    None,
                    *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + \
            torch.ones(img.shape[0], img.shape[1], 3,
                    device=self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3].detach().cpu().numpy()
        return (1 - img).sum()

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
                                  text_input, model, self.genus, device=self.device)

        del model
        return attn_maps

    def set_attention_map(self):
        return self.clip_attn()

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()

    def read_part_loc(self, img_id, filename):
        data = []
        with open(filename) as f:
            for line in f:
                pieces = line.strip().split()
                if pieces[0] == img_id:
                    data.append(pieces)
        return data

    def get_kps(self, img_id, target_im):
        loc_data = self.read_part_loc(
            img_id, f"CUB_200_2011/parts/part_locs.txt")
        kps = []
        for row in loc_data:
            if row[4] == '1':
                x, y = int(float(row[2])), int(float(row[3]))
                rescaled_x = int((x / self.im_width) * self.canvas_width)
                rescaled_y = int((y / self.im_height) * self.canvas_height)
                kps.append([rescaled_y, rescaled_x])
        num_kp = len(kps)
        return kps, num_kp

    def set_inds_keypoints(self, target_im):
        kps, _ = self.get_kps(self.img_id, target_im)

        xdog = XDoG_()
        im_xdog = (1-xdog(np.squeeze(target_im).permute(1, 2, 0).cpu(), k=5))
        attn_map_soft = np.copy(im_xdog)
        attn_map_soft[im_xdog > 0] = self.softmax(
            im_xdog[im_xdog > 0], tau=self.softmax_temp)
        k = self.num_stages * self.num_paths
        self.inds = np.random.choice(range(
            im_xdog.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        self.inds = np.array(np.unravel_index(
            self.inds, target_im.shape)).T[:, 2:]
        for i, kp in enumerate(kps):
            self.inds[i] = kp

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()

    def set_inds_clip(self, target_im):
        attn_map1 = (self.attention_map[0] - self.attention_map[0].min()) / \
            (self.attention_map[0].max() - self.attention_map[0].min())
        if self.xdog_intersec:
            xdog = XDoG_()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(
                1, 2, 0).cpu().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map1
            attn_map1 = intersec_map

        attn_map_soft = np.copy(attn_map1)
        attn_map_soft[attn_map1 > 0] = self.softmax(
            attn_map1[attn_map1 > 0], tau=self.softmax_temp)

        k = self.num_stages * self.num_paths
        self.inds = np.random.choice(range(
            attn_map1.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        self.inds = np.array(np.unravel_index(self.inds, attn_map1.shape)).T

        attn_map2 = (self.attention_map[1] - self.attention_map[1].min()) / \
            (self.attention_map[1].max() - self.attention_map[1].min())

        attn_map_soft = np.copy(attn_map2)
        attn_map_soft[attn_map2 > 0] = self.softmax(
            attn_map2[attn_map2 > 0], tau=0.1)

        k = self.num_stages * self.num_paths
        inds2 = np.random.choice(range(
            attn_map2.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        inds2 = np.array(np.unravel_index(inds2, attn_map2.shape)).T

        attn_map3 = (self.attention_map[2] - self.attention_map[2].min()) / \
            (self.attention_map[2].max() - self.attention_map[2].min())

        attn_map_soft = np.copy(attn_map3)
        attn_map_soft[attn_map3 > 0] = self.softmax(
            attn_map3[attn_map3 > 0], tau=0.1)

        k = self.num_stages * self.num_paths
        inds3 = np.random.choice(range(
            attn_map3.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        inds3 = np.array(np.unravel_index(inds3, attn_map3.shape)).T

        attn_map4 = (self.attention_map[3] - self.attention_map[3].min()) / \
            (self.attention_map[3].max() - self.attention_map[3].min())

        attn_map_soft = np.copy(attn_map4)
        attn_map_soft[attn_map4 > 0] = self.softmax(
            attn_map4[attn_map4 > 0], tau=0.1)

        k = self.num_stages * self.num_paths
        inds4 = np.random.choice(range(
            attn_map3.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        inds4 = np.array(np.unravel_index(inds4, attn_map4.shape)).T

        # self.inds[0:18] = inds2[0:18]
        # self.inds[18:23] = inds3[18:23]
        # self.inds[23:28] = inds4[23:28]

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
        self.optim_color = args.force_sparse
        self.width_optim = args.width_optim
        self.width_optim_global = args.width_optim
        self.width_lr = args.width_lr
        self.optimize_points = args.optimize_points
        self.optimize_points_global = args.optimize_points
        self.points_optim = None
        self.width_optimizer = None
        self.mlp_width_weights_path = args.mlp_width_weights_path
        self.mlp_points_weights_path = args.mlp_points_weights_path
        self.load_points_opt_weights = args.load_points_opt_weights
        # self.only_width = args.only_width

    def turn_off_points_optim(self):
        self.optimize_points = False

    def switch_opt(self):
        self.width_optim = not self.width_optim
        self.optimize_points = not self.optimize_points

    def init_optimizers(self):
        if self.width_optim:
            points_params, width_params = self.renderer.parameters()
            self.width_optimizer = torch.optim.Adam(
                width_params, lr=self.width_lr)
            if self.mlp_width_weights_path != "none":
                checkpoint = torch.load(self.mlp_width_weights_path)
                self.width_optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                print("optimizer checkpoint loaded from ",
                      self.mlp_width_weights_path)
        else:
            points_params = self.renderer.parameters()

        if self.optimize_points:
            self.points_optim = torch.optim.Adam(
                points_params, lr=self.points_lr)
            if self.mlp_points_weights_path != "none" and self.load_points_opt_weights:
                checkpoint = torch.load(self.mlp_points_weights_path)
                self.points_optim.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                print("optimizer checkpoint loaded from ",
                      self.mlp_points_weights_path)

        if self.optim_color:
            self.color_optim = torch.optim.Adam(
                self.renderer.set_color_parameters(), lr=self.color_lr)

    def update_lr(self, counter):
        if self.optimize_points:
            new_lr = utils.get_epoch_lr(counter, self.args)
            for param_group in self.points_optim.param_groups:
                param_group["lr"] = new_lr

    def zero_grad_(self):
        if self.optimize_points:
            self.points_optim.zero_grad()
        if self.width_optim:
            self.width_optimizer.zero_grad()
        if self.optim_color:
            self.color_optim.zero_grad()

    def step_(self):
        if self.optimize_points:
            self.points_optim.step()
        if self.width_optim:
            self.width_optimizer.step()
        if self.optim_color:
            self.color_optim.step()

    def get_lr(self, optim="points"):
        if optim == "points" and self.optimize_points_global:
            return self.points_optim.param_groups[0]['lr']
        if optim == "width" and self.width_optim_global:
            return self.width_optimizer.param_groups[0]['lr']
        else:
            return None

    def get_points_optim(self):
        return self.points_optim

    def get_width_optim(self):
        return self.width_optimizer

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


def interpret(image, texts, model, genus, device):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images, mode='saliency')
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = [] # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
        cams.append(cam)
        R = R + torch.bmm(cam, R)

    cams_avg = torch.cat(cams) # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    # dimension depends on model usage, e.g. 7 for ViT-B/32, 14 for ViT-B/16, 16 for ViT-L/14
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())

    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    prompts = ["head beak", "tail", "legs"]

    input_img = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    inputs = processor(text=prompts, images=[
                        input_img] * len(prompts), padding="max_length", return_tensors="pt")
    # predict
    with torch.no_grad():
        outputs = model(**inputs)

    preds = outputs.logits.unsqueeze(1)
    output_heat1 = torch.sigmoid(preds[0])
    output_heat1 = output_heat1.unsqueeze(0)
    output_heat1 = torch.nn.functional.interpolate(output_heat1, size=224, mode='bilinear')
    output_heat1 = output_heat1.reshape(224, 224).data.cpu().numpy()
    output_heat1 = (output_heat1 - output_heat1.min()) / (output_heat1.max() - output_heat1.min())

    output_heat2 = torch.sigmoid(preds[1])
    output_heat2 = output_heat2.unsqueeze(0)
    output_heat2 = torch.nn.functional.interpolate(output_heat2, size=224, mode='bilinear')
    output_heat2 = output_heat2.reshape(224, 224).data.cpu().numpy()
    output_heat2 = (output_heat2 - output_heat2.min()) / (output_heat2.max() - output_heat2.min())

    output_heat3 = torch.sigmoid(preds[2])
    output_heat3 = output_heat3.unsqueeze(0)
    output_heat3 = torch.nn.functional.interpolate(output_heat3, size=224, mode='bilinear')
    output_heat3 = output_heat3.reshape(224, 224).data.cpu().numpy()
    output_heat3 = (output_heat3 - output_heat3.min()) / (output_heat3.max() - output_heat3.min())

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

class MLP(nn.Module):
    def __init__(self, num_strokes, num_cp, width_optim=False):
        super().__init__()
        outdim = 1000
        self.width_optim = width_optim
        self.layers_points = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_strokes * num_cp * 2, outdim),
            nn.SELU(inplace=True),
            nn.Linear(outdim, outdim),
            nn.SELU(inplace=True),
            nn.Linear(outdim, num_strokes * num_cp * 2),
        )


    def forward(self, x, widths=None):
        '''Forward pass'''
        deltas = self.layers_points(x)
        return x.flatten() + 0.1 * deltas

class WidthMLP(nn.Module):
    def __init__(self, num_strokes, num_cp, width_optim=False):
        super().__init__()
        outdim = 1000
        self.width_optim = width_optim

        self.layers_width = nn.Sequential(
            nn.Linear(num_strokes, outdim),
            nn.SELU(inplace=True),
            nn.Linear(outdim, outdim),
            nn.SELU(inplace=True),
            nn.Linear(outdim, num_strokes),
            nn.Sigmoid()
        )

    def forward(self, widths=None):
        '''Forward pass'''
        return self.layers_width(widths)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)