PROMPT="masterpiece,best quality, 1girl, solo, black skirt, blue eyes, electric guitar, guitar, headphones, holding, holding plectrum, instrument, long hair, music, one side up, pink hair, playing guiter, pleated skirt, black shirt"
python -m python_coreml_stable_diffusion.pipeline \
  --prompt "$PROMPT" --model-version /Users/daiki/Documents/projects/sd-models/diffusers/Any45LCM \
  -i /Users/daiki/Documents/projects/sd-models/coreml/any45lcm-6bit-CPU_AND_NE-SPLIT_EINSUM_V2 \
  -o ./ --compute-unit CPU_AND_NE --seed 42 \
  --guidance-scale 1.5 --num-inference-steps 6 --scheduler LCM