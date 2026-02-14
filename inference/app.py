import argparse
import pdb
import time
from datetime import datetime
import os
import os.path as osp
import sys
sys.path.insert(0, "./utils")
import numpy as np
from PIL import Image
import torch
from copy import deepcopy
from utils_vllm.vllm_generation_utils import generate
from utils.seq_utils_fix import vis_interleve_sample, decode_generated_results
import gradio as gr
from infer import encode, split_seq_by_image, load_emu_model, log_with_time
import config as model_cfg

gpu_ids = ",".join(str(i) for i in range(model_cfg.tensor_parallel_size))
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL = {}

def infer(text_prompt, text_subtask, images):
    """
    text_prompt: str
    images: List[PIL.Image] or List[path]
    """
    sample = {
        "text_prompt": text_prompt,
        "user_input_subtask": text_subtask,
        "visual_placeholder": "<|VIS_PLH|>",
        "supervised_start": "<|extra_100|>",
        "supervised_end": "<|extra_101|>",
        "image_list": images,
    }

    result = run_single_sample(
        sample=sample,
        model=MODEL["emu3"],
        tokenizer=MODEL["tokenizer"],
        vq_model=MODEL["vq_model"],
        cfg=cfg,
    )

    return result["text"], result["images"]

def run_single_sample(
        sample,
        model,
        tokenizer,
        vq_model,
        cfg,
        area=512 * 512,
):
    # 1. encode
    # seq, subtask_count = encode(sample, tokenizer, vq_model, area=area)
    seq, uncond_text = encode(sample, tokenizer, vq_model, area=area)

    # 2. split by image
    seqs = split_seq_by_image(
        seq,
        cfg.special_token_ids.image_start_token,
        cfg.special_token_ids.image_end_token,
    )

    vq_type = "ibq"
    vised_complete_seq = vis_interleve_sample(
        vq_model,
        tokenizer,
        seqs,
        device=device,
        special_token_ids=cfg.special_token_ids,
        vq_type=vq_type
    )

    num_input_image_str = 0
    input_image_str = ""
    prompt = list()
    for seq in seqs:
        if seq[0] == 151852:
            input_image_str += tokenizer.decode(seq)
            num_input_image_str += 1
        prompt.append(seq)

    prompt_ori = deepcopy(prompt)
    log_with_time(f"-> init {prompt = }")
    log_with_time(f"{num_input_image_str = }")
    prompt = np.concatenate(prompt, 0)
    if prompt[0] != cfg.special_token_ids.bos_token:
        prompt = np.pad(prompt, (1, 0), constant_values=cfg.special_token_ids.bos_token)

    prompt = torch.tensor(prompt).long().to(device).view(1, -1)
    log_with_time(f"-> final {prompt = }, {prompt.shape = }")

    unconditional_ids = tokenizer.encode(
        # "<|extra_203|><|extra_100|>",
        f"<|extra_203|>{uncond_text}{input_image_str}<|extra_100|><|image start|>",  # x2i2
        return_tensors="pt",
        add_special_tokens=False
    )

    results = generate(
        cfg,
        model=model,
        tokenizer=tokenizer,
        input_ids=prompt,
        unconditional_ids=unconditional_ids
    )

    result = []
    for result_tokens in results:
        result.extend(result_tokens)
    result = np.array(result)

    result = np.concatenate([np.array([151852]), result])

    result_seqs = split_seq_by_image(
        result,
        cfg.special_token_ids.image_start_token,
        cfg.special_token_ids.image_end_token
    )

    vised_gen_seq = vis_interleve_sample(
        vq_model,
        tokenizer,
        result_seqs,
        device=device,
        special_token_ids=cfg.special_token_ids,
        vq_type=cfg.vq_type,
    )
    prompt_result_seqs = prompt_ori + result_seqs

    target_size = (512, 512)
    resized_images = []
    for img in vised_gen_seq:
        img = Image.fromarray(img)
        resized_img = img.resize(target_size)
        resized_img = np.array(resized_img)
        resized_images.append(resized_img)
    vised_gen_seq = resized_images

    gen_seq = decode_generated_results(
        vq_model,
        tokenizer,
        result_seqs,
        device=device,
        special_token_ids=cfg.special_token_ids,
        vq_type=cfg.vq_type
    )

    task_seq = decode_generated_results(
        vq_model,
        tokenizer,
        prompt_ori,
        device=device,
        special_token_ids=cfg.special_token_ids,
        vq_type=vq_type
    )

    gen_decode_results = {
        "images": [],
        "text": []
    }
    time_tag = datetime.now().strftime("%Y%m%d%H%M%S")
    save_sample_path = os.path.join(cfg.save_path, time_tag)
    os.makedirs(save_sample_path, exist_ok=True)
    jth = 0
    steps = []
    step_imgs = []
    for i, item in enumerate(gen_seq):
        if 'text' in item:
            txt_path = osp.join(save_sample_path, str(jth).zfill(4) + ".txt")
            if len(steps) > 0:
                step_imgs.append(imgs)
                imgs = []
            step_text = item['text']
            gen_decode_results["text"].append(step_text)
            with open(txt_path, 'w', encoding='utf-8') as file:
                file.write(step_text)

        if 'image' in item:
            img_path = osp.join(save_sample_path, str(jth).zfill(4) + ".png")
            jth += 1
            if type(item['image']) is tuple:
                image = item['image'][0]
            else:
                image = item['image']
            image = image.cpu().numpy()
            image = Image.fromarray(image)
            gen_decode_results['images'].append(image)
            image.save(img_path)

    task = task_seq[0]['text']

    task = task.split("USER: ")[-1].split("USER:")[-1].split("user: ")[-1].split("user:")[-1].split("User: ")[-1].split("User:")[-1]
    task = task.split("ASSISTANT: ")[0].split("ASSISTANT:")[0].split("assistant: ")[0].split("assistant:")[0].split("Assistant: ")[0].split("Assistant:")[0]

    task_text_path = osp.join(save_sample_path, "task.txt")
    with open(task_text_path, "w", encoding="utf-8") as f:
        f.write(task)

    task_imgs = []
    if len(task_seq) > 1:
        jth = 0
        for item in task_seq[1:]:
            if 'image' in item:
                img_path = osp.join(save_sample_path, "task_img_" + str(jth).zfill(4) + ".png")
                jth += 1
                if type(item['image']) is tuple:
                    image = item['image'][0]
                else:
                    image = item['image']
                image = image.cpu().numpy()
                Image.fromarray(image).save(img_path)
                task_imgs.append(img_path)

    del vised_gen_seq, result_seqs, result, prompt, seqs, seq
    torch.cuda.empty_cache()
    return gen_decode_results

def preview_uploaded_images(image_files):
    """
    预览上传的图片
    Args:
        image_files: 上传的图片文件列表
    Returns:
        list: PIL Image 对象列表
    """
    images = []
    if image_files:
        for file in image_files:
            img = Image.open(file)
            images.append(img)
    return images


if __name__ == "__main__":
    emu3, emu3_tokenizer, vq_model, cfg = load_emu_model()
    MODEL.update({
        "emu3": emu3,
        "tokenizer": emu3_tokenizer,
        "vq_model": vq_model,
        "cfg": cfg
    })

    with gr.Blocks() as demo:
        gr.Markdown("## VISTA World Model Inference Server")

        text_prompt = gr.Textbox(label="Text Prompt")
        text_subtask = gr.Textbox(label="Manual subtask input")

        with gr.Row():
            with gr.Column(scale=1):
                image_files = gr.Files(
                    label="Input Images",
                    file_types=["image"],
                    file_count="multiple"
                )
            with gr.Column(scale=8):
                uploaded_preview = gr.Gallery(
                    label="Uploaded Images Preview",
                    columns=4,
                    rows=1,
                    height=480,
                    object_fit="contain"
                )

        image_files.change(
            fn=preview_uploaded_images,
            inputs=[image_files],
            outputs=[uploaded_preview]
        )

        output_text = gr.Textbox(label="Generated Text", lines=3, max_lines=10)
        output_image = gr.Gallery(
            label="Generated Images",
            columns=3,
            height=480,
            object_fit="contain"
        )

        btn = gr.Button("Run")

        btn.click(
            fn=infer,
            inputs=[text_prompt, text_subtask, image_files],
            outputs=[output_text, output_image],
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )