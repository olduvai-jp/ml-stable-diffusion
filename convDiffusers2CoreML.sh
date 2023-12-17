python -m python_coreml_stable_diffusion.torch2coreml \
  --convert-unet --convert-text-encoder --convert-vae-decoder --convert-safety-checker \
  --model-version /Users/daiki/Documents/projects/sd-models/diffusers/Any45LCM \
  -o /Users/daiki/Documents/projects/sd-models/coreml/any45lcm \
  --bundle-resources-for-swift-cli
