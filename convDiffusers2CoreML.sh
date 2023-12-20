# COMPUTE_UNIT=CPU_AND_GPU, ALL, CPU_AND_NE
# ATTN_IMPL=ORIGINAL, SPLIT_EINSUM_V2, SPLIT_EINSUM
COMPUTE_UNIT=CPU_AND_NE
ATTN_IMPL=SPLIT_EINSUM
IS_QUANTIZE=TRUE
NBIT=6

if [ "$IS_QUANTIZE" = TRUE ] ; then
    QFLAG="--quantize-nbits $NBIT"
else
    NBIT=16
    QFLAG=""
fi

OUTPUT_PATH=/Users/daiki/Documents/projects/sd-models/coreml/any4lcm-$NBIT-bit-$COMPUTE_UNIT-$ATTN_IMPL

rm -rf $OUTPUT_PATH

python -m python_coreml_stable_diffusion.torch2coreml \
    --model-version /Users/daiki/Documents/projects/sd-models/diffusers/Any45LCM \
    $QFLAG \
    --convert-unet \
    --convert-text-encoder \
    --convert-vae-decoder \
    --convert-vae-encoder \
    --convert-safety-checker \
    --attention-implementation $ATTN_IMPL \
    --compute-unit $COMPUTE_UNIT \
    --bundle-resources-for-swift-cli \
    -o $OUTPUT_PATH