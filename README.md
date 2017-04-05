# Fast-RCNN as a module for torch7

[Fast-RCNN](https://github.com/rbgirshick/fast-rcnn) implementation for Torch7 as a package with methods for training and testing an object detector network.


## Features

- Fast R-CNN as a package with simple API methods for training, testing, detecting and visualizing objects in images.
- Multi-threaded data loading/processing;
- Multi-GPU support;
- Common data augmentation techniques (color jitter, scaling, etc.);
- Pascal VOC / MS COCO mAP evaluation schemes.


## Package installation

### Requirements

- [Torch7](http://torch.ch/docs/getting-started.html)
- NVIDIA GPU with compute capability 3.5+ (2GB+ ram)
- tds
- matio
- cudnn
- inn
- torchnet


### Installation

To install this package you need to have [Torch7](http://torch.ch/docs/getting-started.html) installed on your machine and some other packages. To install this packages, simply do:

```bash
luarocks install tds
luarocks install cudnn
luarocks install inn
luarocks install matio
luarocks install torchnet
```

Finally, to install this package do the following:

```bash
git clone https://github.com/farrajota/fast-rcnn-torch
cd fast-rcnn-torch && luarocks make rocks/*
```


## Usage

To call this package simply do:

```lua
local fastrcnn = require("fastrcnn")
```

This loads a table with the necessary methods for creating, training and testing a Fast R-CNN network. Also, it contains a method for detecting objects in images and for visualizing the detections with a window frame (requires `qt` to work).


<a name="train"></a>
### train ###

```lua
fastrcnn.train(dataLoadTable, rois, model, modelParameters, opts)
```

Trains a model on a given dataset with some proposals.

#### Parameters

- `dataLoadTable`: Table with methods for loading data. (*type=table*)
- `rois`: Region-of-Interest bounding box proposals. (*type=table*)
- `model`: A Fast R-CNN network. (*type=table*)
- `modelParameters`: The model's parameters (color space, meanstd, pixel_scale and stride). (*type=table*)
- `opts`: Training options. (*type=table*)


<a name="test"></a>
### test ###

```lua
fastrcnn.test(dataLoadTable, rois, model, modelParameters, opt)
```

Test a model on a dataset (mAP score).

#### Parameters

- `dataLoadTable`: Table with methods for loading data. (*type=table*)
- `rois`: Region-of-Interest bounding box proposals. (*type=table*)
- `model`: A Fast R-CNN network. (*type=table*)
- `modelParameters`: The model's parameters (color space, meanstd, pixel_scale and stride). (*type=table*)
- `opts`: Testing options. (*type=table*)


<a name="detector"></a>
### detector ###

```lua
imdetector = fastrcnn.Detector(model, modelParameters, opt)
```

Object detector class.

#### Parameters

- `model`: A Fast R-CNN network. (*type=table*)
- `modelParameters`: The model's parameters (color space, meanstd, pixel_scale and stride). (*type=table*)
- `opts`: Testing options. (*type=table*)

```lua
scores, bboxes = imdetector:detect(im, proposals)
```

Receives an image and region proposals as input and outputs scores and bounding boxes.

#### Parameters

- `im`: Image tensor. (*type=torch.Tensor*)
- `proposals`: Region-of-Interest bounding box proposals (*type=torch.Tensor*)


<a name="utils"></a>
### utils ###

This package contains several utility methods for creating models, loading roi proposals from file or visualizing object detection with a window frame.

- model: methods for storing, loading and setup a Fast R-CNN model.

- nms: non-maximum suppression methods.

- load: load roi proposals from file (ony Matlab files atm).

- visualize_detections: visualize detections with a window frame.

> Note: visualize_detections requires `qt` to work. For that, you need to use the `qlua` interpreter.


## Demos

This [repo](https://github.com/farrajota/fast-rcnn-examples) contains code examples on how to train+test an object detector using this module for the Pascal VOC 2007 dataset.


## License

MIT license (see the LICENSE file)


## Acknowledges

This package was heavily inspired by the following repositories: [Fast-RCNN](https://github.com/rbgirshick/fast-rcnn), [Fast-RCNN for Torch7](https://github.com/mahyarnajibi/fast-rcnn-torch) and [facebook/multipathnet](https://github.com/facebookresearch/multipathnet).
