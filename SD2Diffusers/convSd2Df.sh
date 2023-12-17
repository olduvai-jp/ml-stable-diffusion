# https://huggingface.co/docs/diffusers/using-diffusers/other-formats

SD_CHECKPOINT=/Users/daiki/Documents/projects/sd-models/checkpoints/Anything-v4.5-pruned-mergedVae+lcm-lora-sdv1-5.safetensors
YAML_PATH=/Users/daiki/Documents/projects/sd-models/checkpoints/v1-inference.yaml
DF_OUTPUT_DIR=/Users/daiki/Documents/projects/sd-models/diffusers/Any45LCM

# --from_safetensors 
python ckpt2diff.py --from_safetensors --checkpoint_path $SD_CHECKPOINT \
  --original_config_file $YAML_PATH \
  --dump_path $DF_OUTPUT_DIR