import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info  # 请确保该模块在你的环境可用
from transformers import HunYuanVLV1ForConditionalGeneration

import json
import tqdm

# ======================
# 全局变量：模型与处理器（只加载一次）
# ======================

# 模型和processor初始化为 None
model = None
processor = None
model_loaded = False

MODEL_PATH = "/data/manayang/500m_vl_ocr_1112_format/"  # 请根据实际情况修改

def load_model_once():
    """加载模型和processor，仅需执行一次"""
    global model, processor, model_loaded
    if model_loaded:
        return

    print("[INFO] 正在加载 Hunyuan-VL 模型和 Processor（仅需一次）...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 加载 processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)

    # 加载模型，使用 bfloat16 + cuda，注意设置好 attn_implementation
    model = HunYuanVLV1ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",  # 也可以是 flash_attention_2 或 sdpa，根据你的环境支持情况选择
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model_loaded = True
    print("[INFO] 模型加载完成 ✅，可以开始推理了！")

# ======================
# 推理函数：输入 text 和 img_path，返回 output_texts
# ======================
def inference_hunyuan_vl(text: str, img_path: str) -> list:
    """
    使用已经加载的 Hunyuan-VL 模型，对输入的图片和文本进行推理。

    Args:
        text (str): 用户提问，比如要提取图片中的哪些字段
        img_path (str): 图片的本地路径

    Returns:
        list: 模型生成的文本结果，通常是 [output_str]
    """
    global model, processor

    if not model_loaded:
        raise RuntimeError("模型未加载！请先调用 load_model_once() 或确保推理前已加载。")

    # 构造对话输入
    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": text},
            ],
        }
    ]
    messages = [messages1]

    # 使用 processor 构造输入格式
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 模型推理
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024*8,
            repetition_penalty=1.03,
            do_sample=False
        )

    # 解码输出
    if "input_ids" in inputs:
        input_ids = inputs.input_ids
    else:
        input_ids = inputs.inputs  # fallback

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]

    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_texts


# ======================
# 使用示例 👇
# ======================
if __name__ == '__main__':
    # 第一步：加载模型（只会执行一次！）
    load_model_once()

    with open("./ocr_test_data_response_bf16.jsonl",
              "r", encoding="utf-8") as f:
        with open("./ocr_test_data_response_bf16_1115.jsonl",
              "w", encoding="utf-8") as fout:
            for line in tqdm.tqdm(f):
                data = json.loads(line)
                question  = data["question"]
                img_path = data["img_path"]
                img_path = img_path.replace("./images/", "/apdcephfs_gy2/share_302508627/manayang/mm_sh/test_images/")
                result = inference_hunyuan_vl(text=question, img_path=img_path)
                data["response"] = result
                str_out = json.dumps(data, ensure_ascii=False) + "\n"
                print(f"{question} ||||| {result}")
                fout.write(str_out)

    # img_path = "/apdcephfs_gy2/share_302508627/manayang/mm_sh/test_images/0390fdf43974b617001ad98d6bea0ba7.jpg"
    # query = "提取图片中的：['*尿素的单位', '*尿素的参考范围', '*白蛋白的缩写', '*门冬氨酸氨基转移酶的参考范围', '白/球蛋白比值的结果', '血浆渗透压(计算法)的参考范围', '*无机磷的单位', '*氯的缩写', '*胆固醇的缩写']的字段内容，并且按照JSON格式返回。"

    # result = inference_hunyuan_vl(text=query, img_path=img_path)
    # print("推理结果:", result)
