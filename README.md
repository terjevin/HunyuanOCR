<p align="center">
 <img src="./assets/hunyuan_logo.png" width="400"/> <br>
</p>

<div align="center">

# HunyuanOCR: An Advanced multilingual VLM with 1B parameter for end-to-end OCR tasks

📑 Paper & Model Weights are coming soon!

[Demo](#demo) | [Install](#installation) | [Quick Start](#quick-start) | [Documentation](#documentation)

</div>

## 🔥 News
- **[2025/11/20]** 📝 Inference code and model weights publicly available.

## 📖 Introduction
HunyuanOCR is an open-source, commercial-grade, and highly efficient multilingual VLM designed for diverse OCR tasks. It integrates a wide range of capabilities, including text spotting, complex document parsing, information extraction, text-centric VQA and multilingual text image translation into a single end-to-end architecture. With only 1B parameters, the model achieves strong efficiency and competitive performance, outperforming established open-source cascade systems and rivalling large-scale VLMs as well as certain commercial APIs on specific tasks. Its purely end-to-end design enables single-pass inference, substantially simplifying deployment and reducing operational complexity.

<div align="center">
  <img src="./assets/hyocr-pipeline.png" alt="HunyuanOCR framework" width="90%">
</div>

## ✨ Key Features

- 💪 **Compact Yet Powerful**: The first open-source, compact (1B parameters), and efficient multilingual VLM designed specifically for OCR, delivering commercial-grade performance across multiple applications while maintaining deployment simplicity.
- 📑 **Unified Multi-task Capabilities**: Seamlessly integrates multiple OCR-related tasks including text detection, recognition, document parsing, information extraction, visual question answering, and text image translation in a single end-to-end architecture.
- 🌏 **Extensive Language Support**: Provides robust support for over 100 languages, excelling in both single-language and mixed-language scenarios across various document types and formats.
- 🚀 **Easy Deployment**: Simple integration with existing systems


## 🛠️ Dependencies and Installation

### System Requirements
- 🖥️ Operating System: Linux
- 🐍 Python: 3.12+ (recommended and tested)
- ⚡ CUDA: 12.8
- 🔥 PyTorch: 2.7.1
- 🎮 GPU: NVIDIA GPU with CUDA support
- 🧠 GPU Memory: ≥3×80GB (4×80GB recommended for better performance)
- 💾 Disk Space: 170GB for model weights

### Installation
```bash
pip install https://mirrors.tencent.com/repository/generic/transformers/transformers-4.56.0.post2-py3-none-any.whl
pip install opencv-python-headless
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

## 🚀 Quick Start with Transformers

### Model Inference

HunyuanOCR provides intuitive inference interfaces. Here's how to get started:

#### 1. Basic Setup
```python
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, HunYuanVLV1ForConditionalGeneration
from qwen_vl_utils import process_vision_info
```

#### 2. Model Loading
```python
def load_model():
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(
        "PATH_TO_MODEL",
        use_fast=False,
        trust_remote_code=True
    )
    
    model = HunYuanVLV1ForConditionalGeneration.from_pretrained(
        "PATH_TO_MODEL",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    return model, processor
```

## 3. Inference Function
```python
def inference(text: str, img_path: str, model, processor) -> list:
    # Construct input format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": text},
        ],
    }]
    
    # Process inputs
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
             for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Model inference
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate results
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024*8,
            repetition_penalty=1.03,
            do_sample=False
        )
    
    return processor.batch_decode(
        output[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
```

## 4. Usage Examples
```python
# Load model
model, processor = load_model()

# Single image inference
img_path = "path/to/your/image.jpg"
query = "Please extract the text content from this image"
result = inference(text=query, img_path=img_path, model=model, processor=processor)
print("Inference result:", result)

# Batch processing
import json
from tqdm import tqdm

with open("input.jsonl", "r") as fin, open("output.jsonl", "w") as fout:
    for line in tqdm(fin):
        data = json.loads(line)
        result = inference(
            text=data["question"],
            img_path=data["img_path"],
            model=model,
            processor=processor
        )
        data["response"] = result
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
```

## 📊 Evaluation


## 💡 Case Studies
<details>

<summary>Click to view example results</summary>

Document Processing


Table Recognition


Mixed Language Processing


</details>


## 📚 Citation
@misc{hunyuanocr2025,
    title={HunyuanOCR: Advanced OCR Engine for Document Understanding},
    author={Tencent Hunyuan Team},
    year={2025},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/Tencent/HunyuanOCR}}
}

## 🙏 Acknowledgements
Thanks to all contributors who helped build HunyuanOCR
Special thanks to the Tencent Hunyuan Team
We appreciate the support from the open-source community


## 📄 License
This project is licensed under the Apache 2.0 License.
