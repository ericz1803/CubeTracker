# Cube Tracker  

# Installation
1. `pip install  -r requirements.txt`.
2. (optional) Also install `tensorflow-gpu` ([instructions](https://www.tensorflow.org/install/gpu)) and `tensorboard` via pip
3. Use [protoc](https://github.com/protocolbuffers/protobuf/releases) to compile `.proto` files inside the `object_detection`.  
Linux: `protoc object_detection/protos/*.proto --python_out=.`  
Windows (from cmd): `for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.` (might require path to protoc.exe instead of protoc)
4. Install `object_detection` by running `python setup.py install` from the `CubeTracker/` directory.
5. Install TF-slim by running `pip install -e .` from the slim/ directory.
6. Install `cocoapi`. ([non-windows instructions](https://github.com/cocodataset/cocoapi)), ([windows instructions](https://github.com/philferriere/cocoapi))

## Usage
1. Put images in Images folder and images for evaluation in the EvalImages folder and annotations (Pascal VOC format) for them in the annotations and evalannotations folder.
2. Generate the tfrecord files by running `generate_tfrecord.py`. Configure the paths where necessary.
3. Download a starting model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), extract it and put the folder in the `CubeTracker/` directory.
4. If using a new `pipeline.config` file, change `num_classes` to 1, and change the paths to the correct ones. It is easier if the pipeline.config file is moved to the CubeTracker directory.
5. Run `./train.sh` to train the model. Make sure to configure the `PIPELINE_CONFIG_PATH` to point to the correct `pipeline.config` file and the `MODEL_DIR` to point to the directory with the downloaded `model.ckpt` files. Also remove the line with `max_evals` and change `num_examples` to 20. If it is saying that variables could not be found in the checkpoint file, add `from_detection_checkpoint: true` after the line `fine_tune_checkpoint`.
6. Check the training statistics with `tensorboard` by running `tensorboard --logdir=train/`
7. When done training, run `./export.sh` again configuring the paths correctly, and also the name of the checkpoint correctly.
8. Finally, run `python predict.py` to test it on some videos (put them in the `videos/` folder). 

## Using `real_time.py`
### Uses IP Webcam to provide data to the computer.
1. Download IP Webcam (on phone).
2. Set Video and Picture resolution to a low resolution (such as 720x480 or 480x360 for performance reasons).
3. Press Start Server and make sure the IP address listed matches the one in the `real_time.py` file.
4. Run `python real_time.py`.

# Samples
View them here: [google drive](https://drive.google.com/open?id=1MbbqhxA971yuk8MLRdDtXqKKvDo5kbgG)