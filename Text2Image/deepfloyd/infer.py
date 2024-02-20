from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch 
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator

import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

def main(configs):

    # 모형
    pipeline = pipeline = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", torch_dtype=torch.float16)

    scheduler_args = {}
    variance_type = pipeline.scheduler.config.variance_type

    if variance_type in ["learned", "learned_range"]:
        variance_type = "fixed_small"

    scheduler_args["variance_type"] = variance_type

    project_dir = configs['project_dir']
    accelerator_project_config = ProjectConfiguration(project_dir=project_dir, logging_dir='logs')

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16',
        project_config=accelerator_project_config,
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device)

    pipeline.load_lora_weights(project_dir, weight_name="pytorch_lora_weights.safetensors")
    
    generator = torch.Generator(device=accelerator.device)

    prompt = configs['prompt']
    num_inference_steps = configs['num_inference_steps']
    n_sample = configs['n_sample']
    save_dir = configs['save_dir']

    
    safety_modules = {"feature_extractor": pipeline.feature_extractor, "safety_checker": pipeline.safety_checker, "watermarker": pipeline.watermarker}
    sr = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16)
    sr.enable_model_cpu_offload()
    prompt_embeds, negative_embeds = pipeline.encode_prompt(prompt)

    for i in range(n_sample):
        image = pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
        # SR
        image = sr(prompt=prompt, image=image, generator=generator, noise_level=100).images
        image[0].save(save_dir+f'{prompt}_{i+1}.jpg','JPEG')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/infer.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)
    main(configs)