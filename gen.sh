PROMPT="masterpiece,best quality, 1girl, solo, black skirt, blue eyes, electric guitar, guitar, headphones, holding, holding plectrum, instrument, long hair, music, one side up, pink hair, playing guiter, pleated skirt, black shirt"
swift run StableDiffusionSample "$PROMPT" \
 --resource-path /Users/daiki/Documents/projects/sd-models/coreml/any45lcm-6bit-CPU_AND_NE-SPLIT_EINSUM_V2/Resources/ \
 --seed 42 \
 --disable-safety --reduce-memory --guidance-scale 1.5 --step-count 6 --scheduler lcm --output-path ./
