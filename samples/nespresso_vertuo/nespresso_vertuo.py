"""
Mask R-CNN
Train on a Greppy Metaverse generated dataset and infer depth.

Copyright (c) 2018 Achille, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Matthew Moore

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 nespresso_vertuo.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 nespresso_vertuo.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 nespresso_vertuo.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Run inference on an image
    python3 nespresso_vertuo.py infer --weights=/path/to/weights/file.h5 --image=<URL or path to file> --depth=<URL or path to file>
"""

import os
import fnmatch
import sys
import json
import datetime
import numpy as np
import skimage.draw
import OpenEXR, Imath

# Nespresso part classes
NESPRESSO_PART_CLASSES = ["User:5649391675244544/nespresso-vertuo-coffee-machine/components/button","User:5649391675244544/nespresso-vertuo-coffee-machine/components/coffee-spout","User:5649391675244544/nespresso-vertuo-coffee-machine/components/cup-platform","User:5649391675244544/nespresso-vertuo-coffee-machine/components/new-capsule-basin","User:5649391675244544/nespresso-vertuo-coffee-machine/components/new-capsule-open-close-latch","User:5649391675244544/nespresso-vertuo-coffee-machine/components/platform-for-water-tank","User:5649391675244544/nespresso-vertuo-coffee-machine/components/used-capsule-basin","User:5649391675244544/nespresso-vertuo-coffee-machine/components/water-tank-basin","User:5649391675244544/nespresso-vertuo-coffee-machine/components/water-tank-lid"]

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class NespressoVertuoConfig(Config):
    """Configuration for training on the Nespresso dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "nespresso_vertuo"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(NESPRESSO_PART_CLASSES)  # Background + components @ http://metaverse.greppy.co/things/5718607623356416

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class NespressoVertuoDataset(utils.Dataset):

    def subset_prefixes_list(self, dataset_dir):
        print("Loading dataset ", dataset_dir)
        dataset_prefixes = []
        for root, dirs, files in os.walk(dataset_dir):
            # one mask json file per scene so we can get the prefixes from them
            for filename in fnmatch.filter(files, '*masks.json'):
                dataset_prefixes.append(filename[0:0-len('-masks.json')])
        dataset_prefixes.sort()
        return dataset_prefixes

    # file_kind = componentMasks-left or variantMasks-left
    def load_exr(self, prefix_dir, prefix, file_kind, expected_height, expected_width):
        exr_file = OpenEXR.InputFile(os.path.join(prefix_dir,prefix+"-"+file_kind+".exr"))
        cm_dw = exr_file.header()['dataWindow']
        exr_data = np.fromstring(
            exr_file.channel('R', Imath.PixelType(Imath.PixelType.HALF)),
            dtype=np.float16
        )
        exr_data.shape = (cm_dw.max.y - cm_dw.min.y + 1, cm_dw.max.x - cm_dw.min.x + 1) # rows, cols
        if exr_data.shape[0] != expected_height:
            print("[ERROR] ", prefix, file_kind, " != expected image height", exr_data.shape[0], expected_height)
        if exr_data.shape[1] != expected_width:
            print("[ERROR] ", prefix, file_kind, " width != image width", exr_data.shape[1], expected_width)
        return exr_data

    def load_subset(self, dataset_dir, subset):
        """Load a subset of the generated dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: 'training' or 'validation'
        """
        # Add classes wiith their ids for all the components
        for i, component_uri in enumerate(NESPRESSO_PART_CLASSES):
            self.add_class(NespressoVertuoConfig.NAME, i+1, component_uri)

        # Train or validation dataset?
        assert subset in ["training", "validation"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # TODO FIXME only doing the left images
        # TODO FIXME not doing depth

        dataset_prefixes = self.subset_prefixes_list(dataset_dir)
        assert len(dataset_prefixes) > 0

        for prefix in dataset_prefixes:
            image_filename = prefix+'-rgb-left.jpg'
            image_path = os.path.join(dataset_dir, image_filename)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                NespressoVertuoConfig.NAME,
                image_id=image_filename,  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                prefix=prefix,
                prefix_dir=dataset_dir
            )


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If image is not from this dataset, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != NespressoVertuoConfig.NAME:
            return super(self.__class__, self).load_mask(image_id)

        # the json file has the information about all the possible pixels
        masks_json = json.load(open(os.path.join(image_info['prefix_dir'], image_info['prefix']+'-masks.json')))

        variant_data = self.load_exr(
            image_info['prefix_dir'],
            image_info['prefix'],
            "variantMasks-left",
            image_info['height'],
            image_info['width']
        )
        component_data = self.load_exr(
            image_info['prefix_dir'],
            image_info['prefix'],
            "componentMasks-left",
            image_info['height'],
            image_info['width']
        )

        # For each variant in the scene, for each component of it, we might have an instance
        # Or, you might not (it could be "not in view" for this variant).  Loop and only add the variant
        # instances that are in the scene.
        class_ids = []
        masks_bool = []
        for variant_pixel_val_str, instance in masks_json["variant_masks"]["pixel_values"].items():
            variant_pixel_val = float(int(variant_pixel_val_str))
            variant_data_copy = np.copy(variant_data)
            variant_data_copy[variant_data_copy != variant_pixel_val] = 0
            variant_data_copy[variant_data_copy == variant_pixel_val] = 1
            for component_pixel_val_str, component_mask in masks_json["component_masks"].items():
                # Filter to only the pixel values where the variants line up
                if component_mask['variant_uri'] == instance['variant_uri']:
                    # Run intersection on this variant with this component
                    component_pixel_val = float(int(component_pixel_val_str))
                    component_data_copy = np.copy(component_data)
                    component_data_copy_sum_test = (component_data_copy == 106).sum()
                    component_data_copy[component_data_copy != component_pixel_val] = 0
                    component_data_copy[component_data_copy == component_pixel_val] = 1
                    intersected_data = np.bitwise_and(variant_data_copy.astype(np.bool), component_data_copy.astype(np.bool))
                    # intersection actually exists on this one
                    if np.any(intersected_data):
                        masks_bool.append(intersected_data)
                        component_class_id = NESPRESSO_PART_CLASSES.index(component_mask['component_uri']) + 1
                        class_ids.append(component_class_id)

        # Convert generate bitmap masks of all components in the image
        # shape" [height, width, instance_count]
        mask = np.zeros([image_info["height"], image_info["width"], 0], dtype=np.bool)
        if len(masks_bool) > 0:
            mask = np.stack(masks_bool, axis=-1)

        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == NespressoVertuoConfig.NAME:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################
def train(model, dataset):
    """Train the model."""
    # Training dataset.
    dataset_train = NespressoVertuoDataset()
    dataset_train.load_subset(dataset, "training")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NespressoVertuoDataset()
    dataset_val.load_subset(dataset, "validation")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


############################################################
#  Inference
############################################################
def draw_objects_and_depth(image, r):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_infer_depth(model, dataset_dir, image_path=None, depth_path=None):
    assert image_path and depth_path

    # Run model detection and generate the color splash effect
    print("Running on {} with dataset {}".format(image_path, dataset_dir))

    dataset = NespressoVertuoDataset()
    dataset.load_subset(dataset_dir, "validation")
    dataset.prepare()

    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    # Save image with masks
    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'],
        dataset.class_names, r['scores'],
        show_bbox=True, show_mask=True,
        title="Predictions")
    # annotated = draw_objects_and_depth(image, r['masks'])
    # Save output
    # file_name = "depth_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    # skimage.io.imsave(file_name, annotated)
    # TODO FIXME update to also read depth
    print("Saved to ", file_name)



############################################################
#  Main
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect nespressos.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'infer'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/generated/dataset/",
                        help='Directory of the generated dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to predict on')
    parser.add_argument('--depth', required=False,
                        metavar="path or URL to depth image exr",
                        help='Accompanying depth file to predict on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "infer":
        assert args.image and args.depth,\
               "Provide --image and --depth to run inference"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NespressoVertuoConfig()
    else:
        class InferenceConfig(NespressoVertuoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset)
    elif args.command == "infer":
        detect_and_infer_depth(model, args.dataset, image_path=args.image,
                                depth_path=args.depth)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'infer'".format(args.command))
