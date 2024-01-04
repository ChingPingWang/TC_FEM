
# README

Improving Tumor Cellularity Estimation with Ensemble Mask R-CNN: Insights from the PAIP 2023 Challenge

An ensemble Mask R-CNN model with data augmentation was proposed to identify the tumor and non-tumor nucleus. 


## Set Up Environment
This algorithm was implemented by using python 3.6.11 using Tensorflow 1.10.0 and Keras 2.2.4 on Linux system with 1 NVIDIA GeForce GTX 1080 Ti GPU.

## Running the code

### Pre-Processing
```crop_image.py``` is capable of dividing the image into 1/8 overlapping images, which is a patch sampling way.
- Set path to the image directory
- Set path to save the visualized image directory 

```normaling.py``` is capable of normalizng the image into same color staining by using the Macenko's method [[1]](https://ieeexplore.ieee.org/document/5193250). I set the maximum stain concentrations to [1.5 1.3].
- Set path to the image directory
- Set path to save the normalized image directory 


### Training
```
cd PAIP2023_maskrcnn/Mask_RCNN/samples/nucleus/
python3 nucleus_paip.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
```
The training parameter showed in ```nucleus_paip.py```.  The input size of training is set to 512×512. The pre-trained weight of ImageNet ILSVRC 2012 dataset had used in the training phase. The SGD optimizer was used with learning rate 0.001. 


### Inference
```
python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
```
- Load a custom model
- Set path to the input image directory
- Set path to the saved image directory 

### Post-Processing
```count_YOLO_npy.py``` is capable of counting the predicted tumor cells and estimating the TC values.

```distribute_data.py``` is capable of transforming the prediction format. The first step moves the image in the right file. The second step produces corresponding annotation txt (For Detection model). The thrid step produces corresponding annotation txt (For Instance segmentation model).

```draw_YOLO_npy.py``` is capable of drawing the contour of instances on H&E stained images, which is ```.npy``` format. The yellow instances represent the positive tumor cells, and blue instances represent the negative tumor cells.

```vis_YOLO8_instance.py``` is a diffeent way to draw the contour of instances on H&E stained images, which is ```.txt``` format with different color.

<<<<<<< HEAD
=======
## Result
| IMAGE                                           | GT                                             | PREDICTION                                      |
| ----------------------------------------------- | ---------------------------------------------- | ----------------------------------------------- |
| <img src="example/tr_c001_4_img.png" width="200" height="200"> | <img src="example/tr_c001_4_gt.png" width="200" height="200"> | <img src="example/tr_c001_4_result.png" width="200" height="200"> |
| ----------------------------------------------- | ---------------------------------------------- | ----------------------------------------------- |
| <img src="example/tr_p010_3_img.png" width="200" height="200"> | <img src="example/tr_p010_3_gt.png" width="200" height="200"> | <img src="example/tr_p010_3_result.png" width="200" height="200"> |


>>>>>>> ea5ecef (Add images and resize them)
## Color Reference

| Color             | RGB                                                                |
| ----------------- | ------------------------------------------------------------------ |
| Postive tumor cell |![Yellow Color](https://via.placeholder.com/10x10/FFFF00/000000?text=+) |
| Negative tumor cell |![Blue Color](https://via.placeholder.com/10x10/0000FF/000000?text=+)

## Reference
```
[1] Macenko, M., Niethammer, M., Marron, J. S., Borland, D., Woosley, J. T., Guan, X., ... & Thomas, N. E. (2009, June). A method for normalizing histology slides for quantitative analysis. In 2009 IEEE international symposium on biomedical imaging: from nano to macro (pp. 1107-1110). IEEE.
```
## Authors

- [@ChingPingWang](https://github.com/ChingPingWang)

