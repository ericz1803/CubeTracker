INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=pipeline.config
TRAINED_CKPT_PREFIX=train/model.ckpt-100000
EXPORT_DIR=export/
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}