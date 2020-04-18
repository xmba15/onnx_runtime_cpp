# small c++ library to quickly use [onnxruntime](https://github.com/microsoft/onnxruntime) to deploy deep learning models #

## Installation ##
- build onnxruntime from source with the following script
```bash
    sudo bash ./scripts/install_onnx_runtime.sh
```

## How to build ##
```bash
make all

# build apps
make apps
```

## How to test apps ##
- Image Classification With Squeezenet
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
