# COMPUTE_UNIT=CPU_AND_GPU
# ATTN_IMPL=ORIGINAL
COMPUTE_UNIT=CPU_AND_NE
ATTN_IMPL=SPLIT_EINSUM_V2
python -m python_coreml_stable_diffusion.torch2coreml \
    --model-version /Users/daiki/Documents/projects/sd-models/diffusers/Any45LCM \
    --convert-unet \
    --convert-text-encoder \
    --convert-vae-decoder \
    --convert-vae-encoder \
    --convert-safety-checker \
    --quantize-nbits 6 \
    --attention-implementation $ATTN_IMPL \
    --compute-unit $COMPUTE_UNIT \
    --bundle-resources-for-swift-cli \
    -o /Users/daiki/Documents/projects/sd-models/coreml/any45lcm-6bit-$COMPUTE_UNIT-$ATTN_IMPL