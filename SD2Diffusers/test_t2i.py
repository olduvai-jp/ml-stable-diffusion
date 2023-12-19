from diffusers import StableDiffusionPipeline, LCMScheduler
import torch

DIFFUSERS_MODEL_PATH = "/Users/daiki/Documents/projects/sd-models/diffusers/Any45LCM"

device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
if torch.backends.mps.is_available():
  device = "mps"

pipe = StableDiffusionPipeline.from_pretrained(DIFFUSERS_MODEL_PATH,safety_checker=None)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
if device == "mps":
  pipe.enable_attention_slicing()

pipe = pipe.to(device)

prompt = "1girl"
output = pipe(
  prompt,
  num_inference_steps=6,
  guidance_scale=1.5,
  output_type="pil",
  generator= torch.Generator(device).manual_seed(42),
  )

image = output.images[0]
image.save("output.png")