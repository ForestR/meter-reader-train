
我们将数据集的标注和维护管道迁移到ultralytics的[官方平台](https://platform.ultralytics.com/)，并使用NDJSON文件作为数据集管理工具。

```bash
(/backup/conda/venvs/torch_env) [d.yin@linuxconfig meter-reader-train]$ ls -la data/stage1_dial_seg/
总用量 720
drwxr-xr-x. 2 d.yin d.yin     73  4月  1 14:14 .
drwxr-xr-x. 6 d.yin d.yin    110  4月  1 14:18 ..
-rw-r--r--. 1 d.yin d.yin   1829  4月  1 14:14 INFO.md
-rw-r--r--. 1 d.yin d.yin 226437  4月  1 14:13 meter-panel.ndjson
-rw-r--r--. 1 d.yin d.yin 500949  4月  1 14:13 water-meter.ndjson
```

NDJSON文件的相关说明详见[docs/repository/YOLO/ndjson.md](docs/repository/YOLO/ndjson.md)。

为了便于重复训练，我们将NDJSON文件中的图片下载到本地并转换为YOLO格式，详见[yolo_ndjson_symlink_mix_and_cache.md](docs/repository/YOLO/yolo_ndjson_symlink_mix_and_cache.md)。


