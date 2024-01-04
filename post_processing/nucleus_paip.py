"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
    python3 nucleus_paip.py train --dataset=/mnt/disk6/nancy/paip_2023/data/ --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last
    python3 nucleus_paip.py train --dataset=/mnt/disk6/nancy/paip_2023/data/ --subset=train --weights=/mnt/disk6/nancy/paip_2023/code/Mask_RCNN/logs/nucleus20230126T2248/mask_rcnn_nucleus_1227.h5

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""



# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = [
"tr_c003_5","tr_c003_6","tr_p001_2","tr_p001_7","tr_p001_9","tr_p002_1","tr_p002_5",
"tr_p003_5","tr_p003_7","tr_p005_1","tr_p005_8","tr_p006_2","tr_p007_3","tr_p007_7",
"tr_p008_9","tr_p009_9","tr_p010_1","tr_p010_9","tr_p011_1","tr_p012_1","tr_p012_9",
"tr_p013_1","tr_p013_2","tr_p013_3","tr_p013_4","tr_p014_5","tr_p014_7","tr_p015_9",
"tr_p016_5","tr_p017_2","tr_p017_6","tr_p018_2","tr_p018_4","tr_p018_6","tr_p018_7",
"tr_p019_1","tr_p019_5","tr_p019_6","tr_p020_5","tr_p020_7","tr_p020_8","tr_p022_4",
"tr_p023_8","tr_p024_1","tr_p024_2","tr_p024_3","tr_p026_4","tr_p026_9","tr_p028_4",
"tr_p028_5","tr_p028_6","tr_p029_4","tr_p029_7","tr_p030_3","tr_p030_8","tr_p031_2",
"tr_p031_3","tr_p032_4","tr_p032_6","tr_p033_2","tr_p033_7","tr_p034_2","tr_p034_6",
"tr_p034_8","tr_p035_3","tr_p035_6","tr_p036_2","tr_p036_8","tr_p036_9","tr_p037_2",
"tr_p037_6","tr_p038_1","tr_p039_5","tr_p041_3","tr_p041_5","tr_p041_8","tr_p041_9",
"tr_p042_9","tr_p043_3","tr_p043_9","tr_p044_4","tr_p044_7","tr_p045_3","tr_p046_1",
"tr_p046_6","tr_p047_4","tr_p047_7","tr_p047_9","tr_p048_2","tr_p048_6","tr_p048_8",
"tr_p049_1","tr_p049_9","tr_p050_2","tr_p050_4","tr_p050_9","tr_c003_5_n","tr_c003_6_n",
"tr_p001_2_n","tr_p001_7_n","tr_p001_9_n","tr_p002_1_n","tr_p002_5_n","tr_p003_5_n","tr_p003_7_n",
"tr_p005_1_n","tr_p005_8_n","tr_p006_2_n","tr_p007_3_n","tr_p007_7_n","tr_p008_9_n","tr_p009_9_n",
"tr_p010_1_n","tr_p010_9_n","tr_p011_1_n","tr_p012_1_n","tr_p012_9_n","tr_p013_1_n","tr_p013_2_n",
"tr_p013_3_n","tr_p013_4_n","tr_p014_5_n","tr_p014_7_n","tr_p015_9_n","tr_p016_5_n","tr_p017_2_n",
"tr_p017_6_n","tr_p018_2_n","tr_p018_4_n","tr_p018_6_n","tr_p018_7_n","tr_p019_1_n","tr_p019_5_n",
"tr_p019_6_n","tr_p020_5_n","tr_p020_7_n","tr_p020_8_n","tr_p022_4_n","tr_p023_8_n","tr_p024_1_n",
"tr_p024_2_n","tr_p024_3_n","tr_p026_4_n","tr_p026_9_n","tr_p028_4_n","tr_p028_5_n","tr_p028_6_n",
"tr_p029_4_n","tr_p029_7_n","tr_p030_3_n","tr_p030_8_n","tr_p031_2_n","tr_p031_3_n","tr_p032_4_n",
"tr_p032_6_n","tr_p033_2_n","tr_p033_7_n","tr_p034_2_n","tr_p034_6_n","tr_p034_8_n","tr_p035_3_n",
"tr_p035_6_n","tr_p036_2_n","tr_p036_8_n","tr_p036_9_n","tr_p037_2_n","tr_p037_6_n","tr_p038_1_n",
"tr_p039_5_n","tr_p041_3_n","tr_p041_5_n","tr_p041_8_n","tr_p041_9_n","tr_p042_9_n","tr_p043_3_n",
"tr_p043_9_n","tr_p044_4_n","tr_p044_7_n","tr_p045_3_n","tr_p046_1_n","tr_p046_6_n","tr_p047_4_n",
"tr_p047_7_n","tr_p047_9_n","tr_p048_2_n","tr_p048_6_n","tr_p048_8_n","tr_p049_1_n","tr_p049_9_n",
"tr_p050_2_n","tr_p050_4_n","tr_p050_9_n",
]


############################################################
#  Configurations
############################################################

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + nucleus **(tumor/non_tumor)

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (858 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU 
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.7

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2048 #** training階段所生成的box數量
    POST_NMS_ROIS_INFERENCE = 3072 #** testing階段所生成的box數量

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9 #NMS第一階段篩選最後篩出的機率

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 4096 #** 單張影像所生成的anchors於datagen中(data_gen階段)

    # Image mean (RGB)
    MEAN_PIXEL = np.array([217.81, 198.47, 217.69])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (28, 28)  # (height, width) of the mini-mask **

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 1024 #rpn生成的anchor數量

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1024 #** 最大生成anchor的數量

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 2048 #** 利用DETECTION_NMS_THRESHOLD的nms方式保留的最大數量


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "none"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class NucleusDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "tumor")
        self.add_class("nucleus", 2, "non_tumor")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = next(os.walk(dataset_dir))[1]
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images
        for image_id in image_ids:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "image/{}.png".format(image_id))) #**

    def load_detect_nucleus(self, dataset_dir, subset):
        self.add_class("nucleus", 1, "tumor")
        self.add_class("nucleus", 2, "non_tumor")
        
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        dataset_dir_all = glob.glob(dataset_dir + '/*.png')
        print()
        image_ids = []
        for dataset_dirs in dataset_dir_all:
            image_ids.append(dataset_dirs.split('/')[-1]) 
        
        for image_id in image_ids:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, "{}".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir  = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "tumor")
        mask_dir2  = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "non_tumor")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        brown = len(mask)
        for f2 in next(os.walk(mask_dir2))[2]:
            if f2.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir2,f2)).astype(np.bool)
#                print(mask_dir2,f2)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        label = np.ones([mask.shape[-1]], dtype=np.int32)
        label[brown:]=2
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, label.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

import warnings
warnings.filterwarnings("ignore")

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleusDataset()
    dataset_train.load_nucleus(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.load_nucleus(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***
    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    # print("Train network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=20,
    #             augmentation=augmentation,
    #             layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5000,
                augmentation=augmentation,
                layers='all')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_detect_nucleus(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        # plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"].split('.')[0]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleusConfig()
    else:
        config = NucleusInferenceConfig()
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
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
