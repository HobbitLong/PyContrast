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

**(1) Results with ResNet-50 (200 epochs pre-training)**

Mask-RCNN, FPN: 
<table><tbody>

<th valign="bottom">Pretrain</th>
<th valign="bottom">Arch</th>
<th valign="bottom">Detector</th>
<th valign="bottom">lr <br> sched</th>
<th valign="bottom">box <br> AP</th>
<th valign="bottom">mask <br> AP</th>
<th valign="bottom">download</th>

<!-- 1x schedule -->

<tr>
<td align="center">No (rand init.)</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">1x</td>
<td align="center">32.8</td>
<td align="center">29.9</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_1x_rand.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/szuzbuxmbglnnrj/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/n2gq6s8digc034z/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">Supervised</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">1x</td>
<td align="center">39.7</td>
<td align="center">35.9</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_1x.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/vunziha5hd11qki/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/2lsv2knlzj54ynz/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">InstDis</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">1x</td>
<td align="center">38.8</td>
<td align="center">35.2</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_1x_infomin.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/bc131tpru4iw7n1/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/xsndddym1gjm0mi/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">PIRL</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">1x</td>
<td align="center">38.6</td>
<td align="center">35.1</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_1x_infomin.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/li01ay7j9z7cwj3/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/00tf21z5sqvzslh/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">MoCo v1</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">1x</td>
<td align="center">39.4</td>
<td align="center">35.6</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_1x_infomin.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/ljs130qyj9w3zhs/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/1qp52p6hahbwxq4/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">InfoMin Aug.</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">1x</td>
<td align="center">40.6</td>
<td align="center">36.7</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_1x_infomin.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/trn58ixmgb4ecnh/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/g78b0awygmmnohd/log.txt?dl=0">log</a>
</td>
</tr>

<!-- 2x schedule -->

<tr>
<td align="center">No (rand init.)</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">2x</td>
<td align="center">38.4</td>
<td align="center">34.7</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_2x_rand.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/s1be6i6aosvcgxp/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/8qyahtv1h6hfbgy/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">Supervised</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">2x</td>
<td align="center">41.6</td>
<td align="center">37.6</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_2x.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/6ekidu2z19fhe1v/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/xhpefhr5gkq6yg3/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">InstDis</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">2x</td>
<td align="center">41.3</td>
<td align="center">37.3</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_2x_infomin.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/m9tsvdxx9h5qa5a/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/ot4lzwwicz7is4x/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">PIRL</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">2x</td>
<td align="center">41.2</td>
<td align="center">37.4</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_2x_infomin.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/riiqvsnx1qk0ec4/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/iuj0wqkb925ply2/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">MoCo v1</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">2x</td>
<td align="center">41.7</td>
<td align="center">37.5</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_2x_infomin.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/eofvp6t9e17b9wq/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/80pgilcus418iri/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">MoCo v2</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">2x</td>
<td align="center">41.7</td>
<td align="center">37.6</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_2x_infomin.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/hcoastwec4al1nq/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/berheqqzzxy9kos/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">InfoMin Aug.</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">2x</td>
<td align="center">42.5</td>
<td align="center">38.4</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_2x_infomin.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/oh0ke3sdil40kge/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/qoidmj93k95rqbd/log.txt?dl=0">log</a>
</td>
</tr>

<!-- 6x schedule -->

<tr>
<td align="center">No (rand init.)</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">6x</td>
<td align="center">42.7</td>
<td align="center">38.6</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_6x_rand.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/3t0ez4d5z06ks6y/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/z5g8xigbmm05z26/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">Supervised</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">6x</td>
<td align="center">42.6</td>
<td align="center">38.5</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_6x.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/6hotgcfrbluk8a4/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/06d1bfz75wuqply/log.txt?dl=0">log</a>
</td>
</tr>

<tr>
<td align="center">InfoMin Aug.</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, FPN</td>
<td align="center">6x</td>
<td align="center">43.6</td>
<td align="center">39.2</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_FPN_6x_infomin.yaml">config</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/7xrlpd3klx0hwnj/model_final.pth?dl=0">model</a>
&nbsp;|&nbsp;
<a href="https://www.dropbox.com/s/1rrwanmhzihtl1m/log.txt?dl=0">log</a>
</td>
</tr>

</tbody></table>

Mask-RCNN, C4:
<table><tbody>

<th valign="bottom">Pretrain</th>
<th valign="bottom">Arch</th>
<th valign="bottom">Detector</th>
<th valign="bottom">lr <br> sched</th>
<th valign="bottom">box <br> AP</th>
<th valign="bottom">mask <br> AP</th>
<th valign="bottom">download</th>

<!-- C4 1x schedule -->

<tr>
<td align="center">Supervised</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, C4</td>
<td align="center">1x</td>
<td align="center">38.2</td>
<td align="center">33.3</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_C4_1x.yaml">config</a>
</td>
</tr>

<tr>
<td align="center">MoCo</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, C4</td>
<td align="center">1x</td>
<td align="center">38.5</td>
<td align="center">33.6</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_C4_1x_infomin.yaml">config</a>
</td>
</tr>

<tr>
<td align="center">InfoMin Aug.</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, C4</td>
<td align="center">1x</td>
<td align="center">39.0</td>
<td align="center">34.1</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_C4_1x_infomin.yaml">config</a>
</td>
</tr>

<!-- C4 2x schedule -->
<tr>
<td align="center">Supervised</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, C4</td>
<td align="center">2x</td>
<td align="center">40.0</td>
<td align="center">34.7</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_C4_2x.yaml">config</a>
</td>
</tr>

<tr>
<td align="center">MoCo</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, C4</td>
<td align="center">2x</td>
<td align="center">40.7</td>
<td align="center">35.6</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_C4_2x_infomin.yaml">config</a>
</td>
</tr>

<tr>
<td align="center">InfoMin Aug.</td>
<td align="center">R-50</td>
<td align="center">Mask-RCNN, C4</td>
<td align="center">2x</td>
<td align="center">41.3</td>
<td align="center">36.0</td>
<td align="center">
<a href="https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/detection/configs/R_50_C4_2x_infomin.yaml">config</a>
</td>
</tr>

</tbody></table>

**(2) Results with other architecture**

See [paper](https://arxiv.org/abs/2005.10243).

### TODO: add logs and pre-trained detector models