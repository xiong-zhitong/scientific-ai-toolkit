import sys
import os

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from toolkit.config_modules import ModelConfig, GenerateImageConfig, NetworkConfig
from extensions_built_in.diffusion_models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusModel
from toolkit.lora_special import LoRASpecialNetwork
from safetensors.torch import load_file
from PIL import Image
from torchvision import transforms

def generate_image(
    prompt,
    image_path,
    output_path="output.png",
    lora_path=None,
    model_path="Qwen/Qwen-Image-Edit-2511",
    qtype="uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2511_torchao_uint3.safetensors",
    sampler="flowmatch",
    device="cuda:0",
):
    print(f"Generating image with prompt: {prompt}")
    print(f"Control image: {image_path}")
    print(f"Model path: {model_path}")
    print(f"QType: {qtype}")
    print(f"Sampler: {sampler}")

    if sampler != "flowmatch":
        print(f"Warning: Sampler '{sampler}' is not fully supported for this model. Defaulting to 'flowmatch' behavior provided by QwenImageEditPlusModel.")


    # 1. Setup Model Configuration
    # We use ModelConfig to handle the qtype parsing (splitting quantization type and adapter path)
    model_config = ModelConfig(
        name_or_path=model_path,
        lora_path=lora_path,
        qtype=qtype,
        quantize=True,
        quantize_te=True,
        qtype_te="qfloat8", # Default for text encoder
        low_vram=True, # Assuming low vram is desired/safer
        dtype="bf16",
    )

    # 2. Initialize Model
    # QwenImageEditPlusModel handles the custom pipeline and scheduler
    model = QwenImageEditPlusModel(
        device=device,
        model_config=model_config,
        dtype="bf16",
    )

    # 3. Load Model
    # This triggers model.load_model(), which calls quantize_model().
    # quantize_model() will parse the split qtype and load the accuracy recovery adapter.
    model.load_model()

    # 4. Prepare Generation Config
    # Aligning with training parameters mentioned in user memories/context
    gen_config = GenerateImageConfig(
        prompt=prompt,
        width=256, # Standard size, adjust if needed
        height=256,
        num_inference_steps=25, # Typical for this model
        guidance_scale=4.0, # CFG scale
        output_path=output_path,
        ctrl_img=image_path, # Path to control image
        seed=42 # Fixed seed for reproducibility
    )

    # 5. Get Custom Pipeline
    # This ensures we use QwenImageEditPlusCustomPipeline and CustomFlowMatchEulerDiscreteScheduler
    pipeline = model.get_generation_pipeline()

    if lora_path:
        print(f"Loading LoRA from {lora_path}")
        raw_lora_state_dict = load_file(lora_path)
        inspected_lora_state_dict = (
            model.convert_lora_weights_before_load(raw_lora_state_dict)
            if hasattr(model, "convert_lora_weights_before_load")
            else raw_lora_state_dict
        )

        network_type = "lora"
        first_key = list(inspected_lora_state_dict.keys())[0]
        if first_key.startswith("lycoris") and any(
            "lokr" in k for k in inspected_lora_state_dict.keys()
        ):
            network_type = "lokr"

        linear_dim = 4
        linear_alpha = linear_dim
        if network_type == "lora":
            for key, value in inspected_lora_state_dict.items():
                if ".lora_A." in key:
                    linear_dim = int(value.shape[0])
                    linear_alpha = linear_dim
                    break

        only_if_contains = []
        if network_type == "lora":
            for key in inspected_lora_state_dict.keys():
                if ".lora_" not in key:
                    continue
                contains_key = key.split(".lora_")[0]
                if contains_key not in only_if_contains:
                    only_if_contains.append(contains_key)

        network_kwargs = {"only_if_contains": only_if_contains}
        if hasattr(model, "target_lora_modules"):
            network_kwargs["target_lin_modules"] = model.target_lora_modules

        network_config_obj = NetworkConfig(
            type=network_type,
            linear=linear_dim,
            linear_alpha=linear_alpha,
            transformer_only=False,
            network_kwargs=network_kwargs,
        )

        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=model.model,
            lora_dim=network_config_obj.linear,
            multiplier=1.0,
            alpha=network_config_obj.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config_obj,
            network_type=network_config_obj.type,
            transformer_only=network_config_obj.transformer_only,
            is_transformer=getattr(model, "is_transformer", False),
            base_model=model,
            **network_config_obj.network_kwargs,
        )

        network.force_to(model.model.device, dtype=torch.float32)
        network._update_torch_multiplier()
        network.apply_to(None, model.model, apply_text_encoder=False, apply_unet=True)
        network.load_weights(inspected_lora_state_dict)
        network.eval()
        network.is_active = True
        print("LoRA loaded successfully via training-equivalent loader")

    # 6. Prepare Embeddings
    # We need to manually generate embeddings because we are using the low-level generate_single_image method
    # or to match the training flow where embeddings are pre-computed.
    
    # Load and process control image for embedding generation
    control_img = Image.open(image_path).convert("RGB")
    t = transforms.Compose([transforms.ToTensor()])
    # Model expects list of tensors for get_prompt_embeds
    control_tensor = t(control_img).unsqueeze(0).to(device, dtype=torch.bfloat16)
    
    print("Generating embeddings...")
    # get_prompt_embeds handles encoding prompt + control image into text embeddings
    conditional_embeds = model.get_prompt_embeds(prompt, control_images=control_tensor)
    unconditional_embeds = model.get_prompt_embeds("", control_images=control_tensor)

    # 7. Generate Image
    print("Running generation...")
    generator = torch.Generator(device=device).manual_seed(gen_config.seed)
    if lora_path:
        network.force_to(model.device_torch, dtype=torch.float32)
        network._update_torch_multiplier()
    
    generated_image = model.generate_single_image(
        pipeline=pipeline,
        gen_config=gen_config,
        conditional_embeds=conditional_embeds,
        unconditional_embeds=unconditional_embeds,
        generator=generator,
        extra={}
    )

    # 8. Save
    print(f"Saving to {output_path}")
    generated_image.save(output_path)
    print("Done.")

if __name__ == "__main__":
    # Example usage
    # You can modify these paths as needed
    prompt = "Keep all other regions of the image completely unchanged. Only edit the red line area. The red line represents the boundary between sea ice and ocean. Slightly shift this boundary irregularly toward the upper right, simulating gentle ice melting. The movement should be subtle and natural, without altering the surrounding texture, color, or structure of the image."
    image_path = "/home/zhitong/ice/ai-toolkit/datasets/calving_control/image2.png" 
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found. Please specify a valid control image path.")
    # Only run if we have an image
    if os.path.exists(image_path):
        generate_image(
            prompt=prompt,
            image_path=image_path,
            output_path="/home/zhitong/ice/qwen_edit_inference_result.png",
            lora_path="/home/zhitong/ice/ai-toolkit/output/qwen_edit_lora_plus/qwen_edit_lora_plus.safetensors",
            sampler="flowmatch"
        )
    else:
        print("Cannot run generation without a control image.")
