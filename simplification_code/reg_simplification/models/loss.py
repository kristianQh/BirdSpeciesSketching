
import os
import numpy as np
import collections
import CLIP_.clip as clip
import torch
import torch.nn as nn
from torchvision import models, transforms
import timm as timm
import re


def compute_grad_norm_losses(losses_dict, model, points_mlp):
    '''
    Balances multiple losses by weighting them inversly proportional
    to their overall gradient contribution.
    
    Args:
        losses: A dictionary of losses.
        model: A PyTorch model.
    Returns:
        A dictionary of loss weights.
    '''
    grad_norms = {}
    for loss_name, loss in losses_dict.items():
        grads = torch.autograd.grad(
            loss, model.parameters(), create_graph=True)
        grad_sum = sum([w.abs().sum().item() for w in grads if w is not None])
        num_elem = sum([w.numel() for w in grads if w is not None])
        grad_norms[loss_name] = grad_sum / num_elem

    epsilon = 1e-30  # to avoid division by zero
    grad_norms_total = sum(grad_norms.values())
    loss_weights = {}
    for loss_name, loss in losses_dict.items():
        weight = (grad_norms_total -
                  grad_norms[loss_name]) / (((len(losses_dict) - 1) * grad_norms_total) + epsilon)
        loss_weights[loss_name] = weight

    return loss_weights


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.percep_loss = args.percep_loss

        self.train_with_clip = args.train_with_clip
        self.clip_weight = args.clip_weight
        self.start_clip = args.start_clip

        self.clip_score = args.clip_score

        self.width_optim = args.width_optim

        self.clip_conv_loss = args.clip_conv_loss
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.clip_text_guide = args.clip_text_guide
        self.clip_conv_layer_weights = args.clip_conv_layer_weights

        self.force_sparse = args.force_sparse
        self.losses_to_apply = self.get_losses_to_apply()
        self.gradnorm = args.gradnorm
        self.width_loss_weight = args.width_loss_weight
        if self.gradnorm:
            self.new_weights = {}

        self.loss_mapper = {}
        if self.clip_conv_loss:
            self.loss_mapper["clip_conv_loss"] = CLIPConvLoss(args)
        if self.width_optim:
            self.loss_mapper["width_loss"] = WidthLoss(args)

    def get_losses_to_apply(self):
        losses_to_apply = []
        if self.percep_loss != "none":
            losses_to_apply.append(self.percep_loss)
        if self.train_with_clip and self.start_clip == 0:
            losses_to_apply.append("clip")
        if self.clip_conv_loss:
            losses_to_apply.append("clip_conv_loss")
        if self.clip_text_guide:
            losses_to_apply.append("clip_text")
        if self.width_optim:
            losses_to_apply.append("width_loss")
        print(losses_to_apply)
        return losses_to_apply

    def update_losses_to_apply(self, epoch, width_opt=None, mode="train"):
        if "clip" not in self.losses_to_apply:
            if self.train_with_clip:
                if epoch > self.start_clip:
                    self.losses_to_apply.append("clip")

        # for width loss switch
        if width_opt is not None:
            if self.width_optim and "width_loss" not in self.losses_to_apply and mode == "eval":
                self.losses_to_apply.append("width_loss")
            if width_opt and "width_loss" not in self.losses_to_apply:
                self.losses_to_apply.append("width_loss")
            if not width_opt and "width_loss" in self.losses_to_apply and mode == "train":
                self.losses_to_apply.remove("width_loss")

    def forward(self, sketches, targets, epoch, widths=None, renderer=None, optimizer=None, mode="train", width_opt=None):
        loss = 0
        self.update_losses_to_apply(epoch, width_opt, mode)

        losses_dict = {}
        loss_coeffs = {}
        if self.width_optim:
            loss_coeffs["width_loss"] = self.width_loss_weight

        clip_loss_names = []
        for loss_name in self.losses_to_apply:
            if loss_name in ["clip_conv_loss", "clip_mask_loss"]:
                conv_loss = self.loss_mapper[loss_name](
                    sketches, targets, mode)
                for layer in conv_loss.keys():
                    losses_dict[layer] = conv_loss[layer]
                    layer_w_index = int(re.findall(r'\d+', layer)[0])
                    losses_dict[layer] = conv_loss[layer]
                    loss_coeffs[layer] = self.clip_conv_layer_weights[layer_w_index]
                    clip_loss_names.append(layer)
            elif loss_name == "width_loss":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    widths, renderer.get_strokes_in_canvas_count())
            elif loss_name == "l2":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets).mean()
            else:
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets, mode).mean()
            # loss = loss + self.loss_mapper[loss_name](sketches, targets).mean() * loss_coeffs[loss_name]

        losses_dict_orig = losses_dict.copy()
        if self.gradnorm:
            if mode == "train":
                if self.width_optim:
                    self.new_weights = compute_grad_norm_losses(
                        losses_dict, renderer.get_width_mlp(), renderer.get_mlp())
                else:
                    self.new_weights = compute_grad_norm_losses(
                        losses_dict, renderer.get_mlp(), renderer.get_mlp())

            for key in losses_dict.keys():
                losses_dict[key] = losses_dict[key] * self.new_weights[key]

        losses_dict_copy = {}  # return the normalised losses before weighting
        for k_ in losses_dict.keys():
            losses_dict_copy[k_] = losses_dict[k_].clone().detach()
        for key in losses_dict.keys():
            # loss = loss + losses_dict[key] * loss_coeffs[key]
            if loss_coeffs[key] == 0:
                losses_dict[key] = losses_dict[key].detach() * loss_coeffs[key]
            else:
                losses_dict[key] = losses_dict[key] * loss_coeffs[key]

        return losses_dict, losses_dict_copy

class LPIPS(torch.nn.Module):
    def __init__(self, pretrained=True, normalize=True, pre_relu=True, device=None):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(LPIPS, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        self.normalize = normalize
        print("Running LPIPS")
        self.pretrained = pretrained
        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        self.augment_trans = transforms.Compose(augemntations)
        self.feature_extractor = LPIPS._FeatureExtractor(
            pretrained, pre_relu).to(device)

    def _l2_normalize_features(self, x, eps=1e-10):
        nrm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        return x / (nrm + eps)

    def forward(self, pred, target, mode="train"):
        """Compare VGG features of two inputs."""

        # Get VGG features

        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0)
        ys = torch.cat(img_augs, dim=0)

        pred = self.feature_extractor(xs)
        target = self.feature_extractor(ys)

        # L2 normalize features
        if self.normalize:
            pred = [self._l2_normalize_features(f) for f in pred]
            target = [self._l2_normalize_features(f) for f in target]

        # TODO(mgharbi) Apply Richard's linear weights?

        if self.normalize:
            diffs = [torch.sum((p - t) ** 2, 1)
                     for (p, t) in zip(pred, target)]
        else:
            # mean instead of sum to avoid super high range
            diffs = [torch.mean((p - t) ** 2, 1)
                     for (p, t) in zip(pred, target)]

        # Spatial average
        diffs = [diff.mean([1, 2]) for diff in diffs]

        return sum(diffs)

    class _FeatureExtractor(torch.nn.Module):
        def __init__(self, pretrained, pre_relu):
            super(LPIPS._FeatureExtractor, self).__init__()
            vgg_pretrained = models.vgg16(pretrained=pretrained).features

            self.breakpoints = [0, 4, 9, 16, 23, 30]
            if pre_relu:
                for i, _ in enumerate(self.breakpoints[1:]):
                    self.breakpoints[i + 1] -= 1

            # Split at the maxpools
            for i, b in enumerate(self.breakpoints[:-1]):
                ops = torch.nn.Sequential()
                for idx in range(b, self.breakpoints[i + 1]):
                    op = vgg_pretrained[idx]
                    ops.add_module(str(idx), op)
                # print(ops)
                self.add_module("group{}".format(i), ops)

            # No gradients
            for p in self.parameters():
                p.requires_grad = False

            # Torchvision's normalization: <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>
            self.register_buffer("shift", torch.Tensor(
                [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("scale", torch.Tensor(
                [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            feats = []
            x = (x - self.shift) / self.scale
            for idx in range(len(self.breakpoints) - 1):
                m = getattr(self, "group{}".format(idx))
                x = m(x)
                feats.append(x)
            return feats


class L2_(torch.nn.Module):
    def __init__(self):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(L2_, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)
        # LOG.warning("LPIPS is untested")

    def forward(self, pred, target, mode="train"):
        """Compare VGG features of two inputs."""

        # Get VGG features

        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        pred = torch.cat(sketch_augs, dim=0)
        target = torch.cat(img_augs, dim=0)
        diffs = [torch.square(p - t).mean() for (p, t) in zip(pred, target)]
        return sum(diffs)


class ModelConvLoss(nn.Module):
    def __init__(self, args):
        super(ModelConvLoss, self).__init__()

        self.args = args

        self.clip_conv_loss_type = args.clip_conv_loss_type
        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        # TODO CLIP should not be loaded here
        self.clip, clip_preprocess = clip.load(
            args.clip_model_name, args.device, jit=False)

        self.img_size = clip_preprocess.transforms[1].size
        self.clip.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # TODO clip normalization?
        # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.rnn_transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        self.device = args.device
        self.num_augs = self.args.num_aug_clip

        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
            augemntations.append(transforms.RandomAffine(
                degrees=(0, 50), translate=(0.1, 0.8), scale=(0.3, 1.1)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.clip_fc_loss_weight = args.clip_fc_loss_weight

        # Load model used for convolution loss
        self.model = timm.create_model(
            args.model_name, pretrained=True, features_only=True)
        self.model.to(args.device)
        self.model.eval()

    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        conv_loss_dict = {}
        x = sketch.to(self.device)
        y = target.to(self.device)

        # TODO look into this normalization and augmentation
        sketch_augs, img_augs = [self.normalize_transform(x)], [
            self.normalize_transform(y)]

        if mode == "train":
            for n in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([x, y]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        xs_features = self.model(xs)
        xs_fc_features, xs_conv_features = 0, xs_features
        ys_features = self.model(ys)
        ys_fc_features, ys_conv_features = 0, ys_features

        # By default distance metrics is L2
        # TODO model_name "RN"?
        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, "RN")

        for layer, w in enumerate(self.args.clip_conv_layer_weights):
            if w:
                conv_loss_dict[f"conv_loss_layer{layer}"] = conv_loss[layer] * w

        # TODO currently not using any fc / semantic loss
        if self.clip_fc_loss_weight:
            #w_semantic * l_semantic
            fc_loss = (1 - torch.cosine_similarity(xs_fc_features,
                       ys_fc_features, dim=1)).mean()
            conv_loss_dict["fc"] = fc_loss * 0

        return conv_loss_dict


def load_class_names(dataset_path=''):
  names = []
  with open(os.path.join(dataset_path, 'classes.txt')) as f:
    for line in f:
      pieces = line.strip().split()[1]
      split = pieces.split(".")
      spiece_name = split[1].replace("_", " ")
      names.append(spiece_name)
  return names

# def classify_images(processed_img, model, label_emb, device):
#     # encode image to image embedding
#     # with torch.no_grad():
#     #   img_emb = model.get_image_features(processed_img)
#     #   img_emb /= img_emb.norm(dim=-1, keepdim=True)
#     # img_emb = img_emb.detach().cpu().numpy()
#     # score = np.dot(img_emb, label_emb.T)

#     with torch.no_grad():
#       img_features = model.encode_image(processed_img)
#       text_features = label_emb

#     score = torch.cosine_similarity(text_features, img_features, dim=1)

#     # top_5 = np.argsort(scores, axis=1)[:, -5:]
#     # pred = np.argmax(scores, axis=1)
#     return score


class AlphaLoss(torch.nn.Module):
    def __init__(self, args):
        super(AlphaLoss, self).__init__()
        self.width_loss_type = "L1"
        self.width_loss_weight = float(0)

    def forward(self, alphas):
        sum_w = torch.sum(alphas)
        return sum_w / 64


class WidthLoss(torch.nn.Module):
    def __init__(self, args):
        super(WidthLoss, self).__init__()

    def forward(self, widths, strokes_in_canvas_count):
        sum_w = torch.sum(widths)
        return sum_w / strokes_in_canvas_count


class CLIPScoreLoss(nn.Module):
    def __init__(self, args):
        super(CLIPScoreLoss, self).__init__()
        self.args = args
        self.device = args.device

        self.clip_model, clip_preprocess = clip.load(
            "ViT-B/32", args.device, jit=False)

        self.preprocess = transforms.Compose(
            [clip_preprocess.transforms[-1]])  # clip normalisation

    def forward(self, sketch, target, mode="train"):
        x = sketch.to(self.device)
        sketch_augs = self.preprocess(x)

        text_input = clip.tokenize(
            "a black and white photo of Belted Kingfisher").to(self.device)
        image_features = self.clip_model.encode_image(sketch_augs)
        label_features = self.clip_model.encode_text(text_input)

        return -torch.cosine_similarity(label_features, image_features, dim=1)


class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps


def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    if "RN" in clip_model_name:
        return [torch.square(x_conv, y_conv, dim=1).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


class CLIPConvLoss(torch.nn.Module):
    def __init__(self, args):
        super(CLIPConvLoss, self).__init__()
        self.clip_model_name = "RN101"
        assert self.clip_model_name in [
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "ViT-B/32",
            "ViT-B/16",
        ]
        # clip_conv_loss is the geometric loss
        self.clip_conv_loss_type = args.clip_conv_loss_type
        # clip_fc_loss is the semantic loss
        self.clip_fc_loss_type = "Cos"  # args.clip_fc_loss_type
        assert self.clip_conv_loss_type in [
            "L2", "Cos", "L1",
        ]
        assert self.clip_fc_loss_type in [
            "L2", "Cos", "L1",
        ]

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, clip_preprocess = clip.load(
            self.clip_model_name, args.device, jit=False)

        if self.clip_model_name.startswith("ViT"):
            self.visual_encoder = CLIPVisualEncoder(self.model)

        else:
            self.visual_model = self.model.visual
            layers = list(self.model.visual.children())
            init_layers = torch.nn.Sequential(*layers)[:8]
            self.layer1 = layers[8]
            self.layer2 = layers[9]
            self.layer3 = layers[10]
            self.layer4 = layers[11]
            self.att_pool2d = layers[12]

        self.args = args

        self.img_size = clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.device = args.device
        self.num_augs = self.args.num_aug_clip

        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.clip_fc_layer_dims = None  # self.args.clip_fc_layer_dims
        self.clip_conv_layer_dims = None  # self.args.clip_conv_layer_dims
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.counter = 0

    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        #         y = self.target_transform(target).to(self.args.device)
        conv_loss_dict = {}
        x = sketch.to(self.device)
        y = target.to(self.device)
        sketch_augs, img_augs = [self.normalize_transform(x)], [
            self.normalize_transform(y)]
        if mode == "train":
            for n in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([x, y]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        if self.clip_model_name.startswith("RN"):
            xs_conv_features = self.forward_inspection_clip_resnet(
                xs.contiguous())
            ys_conv_features = self.forward_inspection_clip_resnet(
                ys.detach())

        else:
            xs_fc_features, xs_conv_features = self.visual_encoder(xs)
            ys_fc_features, ys_conv_features = self.visual_encoder(ys)

        # By default distance metrics is L2
        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, self.clip_model_name)

        # default="0,0,1.0,1.0,0"
        for layer, w in enumerate(self.args.clip_conv_layer_weights):
            if w:
                conv_loss_dict[f"clip_conv_loss_layer{layer}"] = conv_loss[layer] * w

        if self.clip_fc_loss_weight:
            # l_geometric + w_semantic * l_semantic
            fc_loss = (1 - torch.cosine_similarity(xs_fc_features,
                       ys_fc_features, dim=1)).mean()
            conv_loss_dict["fc"] = fc_loss * self.clip_fc_loss_weight

        self.counter += 1
        return conv_loss_dict

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x
        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x, x1, x2, x3, x4]
