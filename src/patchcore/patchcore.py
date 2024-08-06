"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
from torchvision.transforms import ToPILImage
from PIL import Image
LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        
        

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer_normal = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        
        self.anomaly_scorer_abnormal = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        
        self.support_scorer_normal = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        self.support_scorer_abnormal = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        
        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                # print(image.shape, image["mask"].shape, image["query"].shape)
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        # print(f'features: {features.shape}')

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, normal_data, abnormal_data):
        """PatchCore training with separate normal and abnormal memory banks."""
        self._fill_memory_bank(normal_data, abnormal_data)

    def _fill_memory_bank(self, normal_data, abnormal_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        def _compute_features(data, flag=0):
            features = []
            with tqdm.tqdm(
                data, desc="Computing support features...", position=1, leave=False
            ) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        # print(torch.unique(image['mask']))
                        # print(image['image'].shape, image["mask"].shape, image["query"].shape)
                        if flag == 0:
                            image = image["normal_img"]
                        else:
                            image = image["abnormal_img"]

                        to_pil = ToPILImage()
                        image_pil = to_pil(image[0].squeeze().cpu())
                        image_pil.save(os.path.join('/home/waylon/HD/AD', 'image.png'))
                        
                    features.append(_image_to_features(image))
            return np.concatenate(features, axis=0)

        features_normal = _compute_features(normal_data,0)
        features_normal = self.featuresampler.run(features_normal)
        features_abnormal = _compute_features(abnormal_data,1)
        features_abnormal = self.featuresampler.run(features_abnormal)
        
        self.anomaly_scorer_normal.fit(detection_features=[features_normal])
        self.anomaly_scorer_abnormal.fit(detection_features=[features_abnormal])

    def fill_memory_bank_support(self, support_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        def _compute_features(data, flag):
            features = []
            with tqdm.tqdm(
                data, desc="Computing support features...", position=1, leave=False
            ) as data_iterator:
                i = 0
                for image in data_iterator:
                    i +=1
                    if isinstance(image, dict):
                        # print(torch.unique(image['mask']))
                        # print(image['image'].shape, image["mask"].shape, image["query"].shape)
                        if flag == 0:
                            image = image["normal_img"]
                            to_pil = ToPILImage()
                            image_pil = to_pil(image[0].squeeze().cpu())
                            image_pil.save(os.path.join('/home/waylon/HD/AD', 'normal_image' + str(i) + '.png'))
                        else:
                            image = image["abnormal_img"]

                            to_pil = ToPILImage()
                            image_pil = to_pil(image[0].squeeze().cpu())
                            image_pil.save(os.path.join('/home/waylon/HD/AD', 'abnormal_image' + str(i) + '.png'))
                        
                    features.append(_image_to_features(image))
            return np.concatenate(features, axis=0)

        features_normal = _compute_features(support_data,0)
        features_normal = self.featuresampler.run(features_normal)
        features_abnormal = _compute_features(support_data,1)
        features_abnormal = self.featuresampler.run(features_abnormal)
        
        self.support_scorer_normal.fit(detection_features=[features_normal])
        self.support_scorer_abnormal.fit(detection_features=[features_abnormal])
    
    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            self.fill_memory_bank_support(data)
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    # image = image["normal_img"]
                _scores, _masks = self._predict(image["normal_img"], image["abnormal_img"])
                for idx, (score, mask) in enumerate(zip(_scores, _masks)):
                    # print(image["query"][idx].unsqueeze(0).shape, image["query"].shape)
                    if score < 0.2:
                        score, mask = self.predict_from_all(image["query"][idx].unsqueeze(0), image["query"][idx].unsqueeze(0))
                    else:
                        score, mask = self.predict_from_support(image["query"][idx].unsqueeze(0), image["query"][idx].unsqueeze(0))
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, normal_images, abnormal_images):
        """Infer score and mask for a batch of images."""
        normal_images = normal_images.to(torch.float).to(self.device)
        abnormal_images = abnormal_images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = normal_images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(normal_images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores_normal = self.anomaly_scorer_normal.predict([features])[0]

            features, patch_shapes = self._embed(abnormal_images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores_abnormal = self.anomaly_scorer_abnormal.predict([features])[0]

            # Combine the scores from normal and abnormal memory banks
            patch_scores = np.maximum(patch_scores_normal - patch_scores_abnormal, 0)

            image_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    def predict_from_support(self, normal_images, abnormal_images):
        """Infer score and mask for a batch of images."""
        normal_images = normal_images.to(torch.float).to(self.device)
        abnormal_images = abnormal_images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        print(f'normal_images shape: {normal_images.shape}')
        
        batchsize = normal_images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(normal_images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores_normal = self.support_scorer_normal.predict([features])[0]

            features, patch_shapes = self._embed(abnormal_images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores_abnormal = self.support_scorer_abnormal.predict([features])[0]

            # Combine the scores from normal and abnormal memory banks
            patch_scores = np.maximum(patch_scores_normal, 0)
            # print(f'scores: {patch_scores_normal}, {patch_scores_abnormal}')
            # print(f'patch_scores: {patch_scores}')  

            image_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]
    
    def predict_from_all(self, normal_images, abnormal_images):
        """Infer score and mask for a batch of images."""
        normal_images = normal_images.to(torch.float).to(self.device)
        abnormal_images = abnormal_images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = normal_images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(normal_images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores_normal = self.support_scorer_normal.predict([features])[0]

            features, patch_shapes = self._embed(abnormal_images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores_abnormal = self.support_scorer_abnormal.predict([features])[0]

            # Combine the scores from normal and abnormal memory banks
            patch_scores = np.maximum(patch_scores_normal - patch_scores_abnormal, 0)
            
            features, patch_shapes = self._embed(normal_images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores_normal = self.anomaly_scorer_normal.predict([features])[0]

            features, patch_shapes = self._embed(abnormal_images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores_abnormal = self.anomaly_scorer_abnormal.predict([features])[0]

            patch_scores = np.maximum(patch_scores_normal - patch_scores_abnormal, patch_scores)
            

            image_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]
    
    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))