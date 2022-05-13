# GPTR: Gestalt-perception Transformer for Diagram Object Detection

## Introduction



The DETR-based approaches apply the
transformer encoder-decoder architecture to nature image object detection
and achieve promising performance. In this paper,
we propose a novel gestalt-perception transformer model for diagram object detection.
Our approach is motivated by that <b>the human visual system tends to perceive patches
in an image that are similar, close or connected without abrupt directional changes as a perceptual whole object.</b>



## Installation

### Requirements
- Python >= 3.7, CUDA >= 10.1
- PyTorch >= 1.7.0, torchvision >= 0.6.1
- Cython, COCOAPI, scipy, termcolor

The code is developed using Python 3.8 with PyTorch 1.7.0.
First, clone the repository locally:
```shell
git clone https://github.com/AIProCode/GPTR.git
```
Then, install PyTorch and torchvision:
```shell
conda install pytorch=1.7.0 torchvision=0.6.1 cudatoolkit=10.1 -c pytorch
```
Install other requirements:
```shell
cd GPTR
pip install -r requirements.txt
```

## Usage

### Data preparation (MSCOCO)

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
├── annotations/  # annotation json files
└── images/
    ├── train2017/    # train images
    ├── val2017/      # val images
    └── test2017/     # test images
```

### Data preparation (AI2D*)

The newly proposed dataset AI2D* is developed by ourselves to support the novel research on diagram object detection. Thanks to Kembhavi A, Salvato M, Kolve E, et al for offering the original AI2D dataset designed for diagram understanding and TQA tasks.

You can click on the links below to download the AI2D* :

BaiduyunDrive: [https://pan.baidu.com/s/1tijw2ti29J8rSAinFkFTVw](https://pan.baidu.com/s/1tijw2ti29J8rSAinFkFTVw), the password for the link is **axby** and the password for the zip file is **gptr**.

### Training

```shell
python -m torch.distributed.launch \
    --nproc_per_node=num \
    --use_env \
    main.py \
    --resume auto \
    --coco_path /path/to/coco \
    --output_dir output/<output_path>
```

### Evaluation

```shell
python -m torch.distributed.launch \
    --nproc_per_node=num \
    --use_env \
    main.py \
    --batch_size 2 \
    --eval \
    --resume <checkpoint.pth> \
    --coco_path /path/to/coco \
    --output_dir output/<output_path>
```

The experimental results show that our model achieves the best results in the diagram object detection task.


<table>
  <thead>
    <tr style="text-align: center;">
      <th>Models</th>
      <th>Epoch</th>
      <th>Layer</th>
      <th>Head</th>
      <th>Params</th>
      <th>AP</th>
      <th>APS</th>
      <th>APM</th>
      <th>APL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CenterNet</td>
      <td>500</td>
      <td>/</td>
      <td>/</td>
      <td>50.39M</td>
      <td>0.086</td>
      <td>0.107</td>
      <td>0.134</td>
      <td>0.125</td>
    </tr>
    <tr>
      <td>RetinaNet</td>
      <td>100</td>
      <td>/</td>
      <td>/</td>
      <td>29.86M</td>
      <td>0.105</td>
      <td>0.060</td>
      <td>0.128</td>
      <td>0.149</td>
    </tr>
    <tr>
      <td>DETR</td>
      <td>1000</td>
      <td>4</td>
      <td>4</td>
      <td>28.93M</td>
      <td>0.105</td>
      <td>0.066</td>
      <td>0.137</td>
      <td>0.139</td>
    </tr>
    <tr>
      <td>ConditionDETR</td>
      <td>50</td>
      <td>4</td>
      <td>4</td>
      <td>29.22M</td>
      <td>0.100</td>
      <td>0.102</td>
      <td>0.140</td>
      <td>0.146</td>
    </tr>
    <tr>
      <td><b>GPTR(Ours)</b></td>
      <td>50</td>
      <td>4</td>
      <td>4</td>
      <td>29.22M</td>
      <td><b>0.132</b></td>
      <td><b>0.111</b></td>
      <td><b>0.166</b></td>
      <td><b>0.181</b></td>
    </tr>
    <tr>
      <td>ConditionDETR</td>
      <td>100</td>
      <td>4</td>
      <td>4</td>
      <td>29.22M</td>
      <td>0.118</td>
      <td>0.107</td>
      <td>0.164</td>
      <td>0.167</td>
    </tr>
    <tr>
      <td><b>GPTR(Ours)</b></td>
      <td>100</td>
      <td>4</td>
      <td>4</td>
      <td>30.56M</td>
      <td><b>0.141</b></td>
      <td><b>0.122</b></td>
      <td><b>0.175</b></td>
      <td><b>0.202</b></td>
    </tr>
    <tr>
      <td>SMCA-DETR</td>
      <td>300</td>
      <td>6</td>
      <td>8</td>
      <td>39.66M</td>
      <td>0.151</td>
      <td>0.091</td>
      <td>0.200</td>
      <td>0.198</td>
    </tr>
    <tr>
      <td>SAM-DETR</td>
      <td>200</td>
      <td>6</td>
      <td>8</td>
      <td>47.08M</td>
      <td>0.152</td>
      <td>0.122</td>
      <td>0.190</td>
      <td>0.204</td>
    </tr>
    <tr>
      <td><b>GPTR(Ours)</b></td>
      <td>120</td>
      <td>6</td>
      <td>8</td>
      <td>33.44M</td>
      <td><b>0.161</b></td>
      <td><b>0.162</b></td>
      <td><b>0.211</b></td>
      <td><b>0.208</b></td>
    </tr>
  </tbody>
</table>


