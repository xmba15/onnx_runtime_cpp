# small c++ library to quickly use [onnxruntime](https://github.com/microsoft/onnxruntime) to deploy deep learning models #

Thanks to [cardboardcode](https://github.com/cardboardcode), we have [the documentation](https://onnx-runtime-cpp.readthedocs.io/en/latest/index.html) for this small library.
Hope that they both are helpful for your work.

## TODO

- [x] Support inference of multi-inputs, multi-outputs
- [x] Examples for famous models, like yolov3, mask-rcnn, [ultra-light-weight face detector](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB), [yolox](https://github.com/Megvii-BaseDetection/YOLOX). Might consider supporting more if requested
- [ ] Batch-inference

## Installation ##
- build onnxruntime from source with the following script
```bash
    sudo bash ./scripts/install_onnx_runtime.sh
```

## How to build ##
***

```bash
make default

# build examples
make apps
```

## How to test apps ##
***

### Image Classification With Squeezenet ###
***

```bash
# after make apps
./build/examples/TestImageClassification ./data/squeezenet1.1.onnx ./data/images/dog.jpg
```
the following result can be obtained
```
264 : Cardigan, Cardigan Welsh corgi : 0.391365
263 : Pembroke, Pembroke Welsh corgi : 0.376214
227 : kelpie : 0.0314975
158 : toy terrier : 0.0223435
230 : Shetland sheepdog, Shetland sheep dog, Shetland : 0.020529
```

### Object Detection With Tiny-Yolov2 trained on VOC dataset (with 20 classes) ###
***
- Download model from onnx model zoo: [HERE](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov2)

- The shape of the output would be
```text
    OUTPUT_FEATUREMAP_SIZE X OUTPUT_FEATUREMAP_SIZE * NUM_ANCHORS * (NUM_CLASSES + 4 + 1)
    where OUTPUT_FEATUREMAP_SIZE = 13; NUM_ANCHORS = 5; NUM_CLASSES = 20 for the tiny-yolov2 model from onnx model zoo
```
- Test tiny-yolov2 inference apps
```bash
# after make apps
./build/examples/tiny_yolo_v2 [path/to/tiny_yolov2/onnx/model] ./data/images/dog.jpg
```
- Test result

![tinyyolov2 test result](./data/images/result.jpg)

### Object Instance Segmentation With MaskRCNN trained on MS CoCo Dataset (80 + 1(background) clasess) ###
***
- Download model from onnx model zoo: [HERE](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/mask-rcnn)

- As also stated in the url above, there are four outputs: boxes(nboxes x 4), labels(nboxes), scores(nboxes), masks(nboxesx1x28x28)
- Test mask-rcnn inference apps
```bash
# after make apps
./build/examples/mask_rcnn [path/to/mask_rcnn/onnx/model] ./data/images/dogs.jpg
```

- Test results:

![dogs maskrcnn result](./data/images/dogs_maskrcnn_result.jpg)

![indoor maskrcnn result](./data/images/indoor_maskrcnn_result.jpg)

### Yolo V3 trained on Ms CoCo Dataset ###
***

- Download model from onnx model zoo: [HERE](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3)

- Test yolo-v3 inference apps
```bash
# after make apps
./build/examples/yolov3 [path/to/yolov3/onnx/model] ./data/images/no_way_home.jpg
```

- Test result

<p align="center">
  <img width="1000" height="667" src="./data/images/no_way_home_result.jpg">
</p>


### [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) ###
***

- App to use onnx model trained with famous light-weight [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
- Sample weight has been saved [./data/version-RFB-640.onnx](./data/version-RFB-640.onnx)
- Test inference apps
```bash
# after make apps
./build/examples/ultra_light_face_detector ./data/version-RFB-640.onnx ./data/images/endgame.jpg
```

- Test results:
![ultra light weight face result](./data/images/endgame_result.jpg)

### [YoloX: high-performance anchor-free YOLO by Megvii](https://github.com/Megvii-BaseDetection/YOLOX)
***

- Download onnx model trained on COCO dataset from [HERE](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime)
```bash
# this app tests yolox_l model but you can try with other yolox models also.
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.onnx -O ./data/yolox_l.onnx
```

- Test inference apps
```bash
# after make apps
./build/examples/yolox ./data/yolox_l.onnx ./data/images/matrix.jpg
```
- Test results:
![yolox result](./data/images/matrix_result.jpg)

### [Semantic Segmentation Paddle Seg](https://github.com/PaddlePaddle/PaddleSeg)
***

- Download PaddleSeg's bisenetv2 trained on cityscapes dataset that has been converted to onnx [HERE](https://drive.google.com/file/d/1e-anuWG_ppDXmoy0sQ0sgrdutCTGlk95/view?usp=sharing) and copy to [./data directory](./data)

<details>
<summary>You can also convert your own PaddleSeg with following procedures</summary>

*  [export PaddleSeg model](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.3/docs/model_export.md)
*  convert exported model to onnx format with [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)

</details>

- Test inference apps

```bash
./build/examples/semantic_segmentation_duc ./data/bisenetv2_cityscapes.onnx ./sample_city_scapes.png
./build/examples/semantic_segmentation_duc ./data/bisenetv2_cityscapes.onnx ./odaiba.jpg
```

- Test results:
    +  test result on sample image of cityscapes dataset (this model is trained on cityscapes dataset)
![paddleseg city scapes](./data/images/sample_city_scapes_result.jpg)

    +  test result on a new scene at Odaiba, Tokyo, Japan
![paddleseg odaiba](./data/images/odaiba_result.jpg)
