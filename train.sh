PIPELINE_CONFIG_PATH=pipeline.config
MODEL_DIR=ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18/
TRAIN_DIR=train/
NUM_TRAIN_STEPS=100000
SAMPLE_1_OF_N_EVAL_EXAMPLES=5
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${TRAIN_DIR} \
    --train_dir=${TRAIN_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES}