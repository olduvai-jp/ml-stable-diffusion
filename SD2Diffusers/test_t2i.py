from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt

DIFFUSERS_MODEL_PATH = "/Users/daiki/Documents/projects/sd-models/diffusers/Any45LCM"

pipe = DiffusionPipeline.from_pretrained(DIFFUSERS_MODEL_PATH)
pipe.enable_attention_slicing()

# for apple silicon
pipe = pipe.to("mps")

prompt = "1girl"
output = pipe(
  prompt,
  num_inference_steps=6,
  guidance_scale=1.0,
  output_type="pil"
  )

image = output.images[0]
image.save("output.png")