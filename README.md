1. Put images in Images folder and images for evaluation in the EvalImages folder and annotations (Pascal VOC format) for them in the annotations and evalannotations folder.
2. Generate the tfrecord files by running `generate_tfrecord.py`. Configure the paths where necessary.
3. From the `slim/` directory, run `pip install -e .`
5. From the `CubeTracker/` directory, run `python setup.py install`
6. Install `cocoapi` [non-windows](https://github.com/cocodataset/cocoapi) [windows](https://github.com/philferriere/cocoapi)
7. Download [protoc](https://github.com/protocolbuffers/protobuf/releases) and run  
`protoc object_detection/protos/*.proto --python_out=.` or  
`for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.` on windows.
8. Download a starting model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), extract it and put the folder in the `CubeTracker/` directory.
9. If using a new `pipeline.config` file, change `num_classes` to 1, and change the paths to the correct ones. It is easier if the pipeline.config file is moved to the CubeTracker directory.
10. Run `./train.sh` to train the model. Make sure to configure the `PIPELINE_CONFIG_PATH` to point to the correct `pipeline.config` file and the `MODEL_DIR` to point to the directory with the downloaded `model.ckpt` files. Also remove the line with `max_evals` and change `num_examples` to 20.
11. Check the training statistics with `tensorboard` by running `tensorboard --logdir=train/`
12. When done training, run `./export.sh` again configuring the paths correctly, and also the name of the checkpoint correctly.
13. Finally, run `python predict.py` to test it on some videos. 