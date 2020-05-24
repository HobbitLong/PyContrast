## Transfer Learning: Object Detection

### Running

1. Install [Detectron2](https://github.com/facebookresearch/detectron2).

2. Convert pre-trained models to Detectron2 models:
```
python convert_pretrained.py model.pth det_model.pkl
```

3. Set up data folders following Detectron2's [datasets instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

4. Go to Detectron2's folder, and run:
```
python tools/train_net.py \
  --num-gpus 8 \
  --config-file /path/to/config/config.yaml \
  MODEL.WEIGHTS /path/to/model/det_model.pkl
```
where `config.yaml` is the config file listed under the [configs](configs) folder.

### Results
See [paper](https://arxiv.org/abs/2005.10243).

### TODO: add logs and pre-trained detector models