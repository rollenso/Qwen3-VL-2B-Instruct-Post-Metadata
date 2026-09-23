# Qwen3-VL-2B-Instruct-Post-Metadata

🌿 Model Details

Base Model: Qwen/Qwen3-VL-2B-Instruct (fine-tuned via Unsloth) Architecture: Autoregressive Vision-Language Model (Vision Transformer + Multimodal Projector + Causal LLM) Parameter Count: 2.2B (1.5 GB in 4-bit quantized / 4.5 GB in bfloat16) Fine-Tuning Method: Supervised Fine-Tuning (SFT / LoRA) via knowledge distillation from a larger Qwen-VL teacher Target Domain: In-the-wild Multimodal UGC (variable resolution, uncurated mobile photography, low-light/noisy captures, digital art) Primary Use Case: Automated post metadata generation (structured titles, descriptive body captions, and search tags) for social networks, content publishing workflows, and digital media feeds. Hardware Footprint: Ultra-low VRAM consumption (~2–3 GB in 4-bit) on budget or shared GPU instances.

🍃 Source files
* adapter: https://huggingface.co/rollenso/Qwen3-VL-2B-Instruct-Post-Metadata
* dataset (.npy included): https://huggingface.co/datasets/rollenso/Qwen3-VL-2B-Instruct-Post-Metadata-dataset-npy

🌱 Usage (Inference)

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from peft import PeftModel

base_model_id = "Qwen/Qwen3-VL-2B-Instruct"
adapter_path = "Qwen3-VL-2B-Instruct-Post-Metadata"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(base_model_id)

model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

image_path = "test1.png"
image = Image.open(image_path).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "Analyze the image and respond strictly according to the template: TITLE: [short post title for this image] DESCRIPTION: [describe the appearance in detail]",
            },
        ],
    }
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print(output_text[0])