import argparse
import pdb
import time
from datetime import datetime
import importlib as imp
import os
import os.path as osp
import random
import sys
import re
from pathlib import Path
import numpy as np
from PIL import Image
from copy import deepcopy
import torch
from easydict import EasyDict
import json
from moviepy import VideoFileClip
from utils_vllm.model_utils import build_emu3p5_vllm

def log_with_time(msg):
    """
    带时间戳的日志函数
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def smart_resize(image: Image.Image, area: int = 512 * 512, ds_factor: int = 16):
    width, height = image.size
    aspect_ratio = width / height  # 原图 和 目标 使用相同的 aspect ratio
    new_height = int((area / aspect_ratio) ** 0.5)  # 固定 area 和 aspect ratio 的前提下, 算出的目标 h
    new_width = int(new_height * aspect_ratio)  # 固定 area 和 aspect ratio 的前提下, 算出的目标 w
    # Round to nearest multiple of divisible_by
    new_height = ((new_height + ds_factor // 2) // ds_factor) * ds_factor  # 使目标 h 为 ds_factor 的整数倍
    new_width = ((new_width + ds_factor // 2) // ds_factor) * ds_factor  # 使目标 w 为 ds_factor 的整数倍

    if area == 512 * 512:
        new_width, new_height = 512, 512
    if area == 256 * 256:
        new_width, new_height = 256, 256
    return image.resize((new_width, new_height), Image.BICUBIC)


def normalize(image: Image.Image):
    image = np.array(image)
    image = image.astype(np.float32) / 255.0  # [0, 255] -> [0, 1]
    image = (image - 0.5) / 0.5  # [0, 1]   -> [-1, 1]
    image = image.transpose(2, 0, 1)
    return image.astype(np.float32)


def encode(sample, tokenizer, vq_model, area=512 * 512, ds_factor=16):
    # --------------------------------------------------------------------------------------------------------------
    def prepare_data(sample):
        """整理 sample dict 并把 image_list 转换为 resized torch.tensor"""
        text_prompt = sample["text_prompt"]
        text_subtask = sample["user_input_subtask"]

        image_list = sample["image_list"]
        if not isinstance(image_list, list):
            image_list = [image_list]
        if not isinstance(image_list, list):
            image_list = [image_list]

        img_num = len(image_list)

        visual_placeholder = sample.get("visual_placeholder", "<|VIS_PLH|>")
        supervised_start = sample.get("supervised_start", "<|extra_100|>")
        supervised_end = sample.get("supervised_end", "<|extra_101|>")

        # sub_text_prompt = "You are a helpful assistant for vla task. USER: " + f"{text_prompt_list[0]}" + ' ASSISTANT: '
        sub_text_prompt = "You are a helpful assistant. USER: " + f"{text_prompt}" + visual_placeholder * img_num + ' ASSISTANT: '

        sub_text_sent_list = text_prompt.split('.')
        uncond_text = "You are a helpful assistant. USER: " + f"{sub_text_sent_list[0]}." + f"{sub_text_sent_list[-1]}." + ' ASSISTANT: '

        text_prompt = sub_text_prompt + supervised_start
        count = 0
        if len(text_subtask) > 0:
            count = len(re.findall(r"Step\s+\d+:", text_subtask))
            text_subtask = text_subtask.replace("\n", "")
            curr_subtask = text_subtask.split(".")[0]
            text_prompt = text_prompt + "To finish the task, I need to do the following steps: " + text_subtask + f" I have already done 0 steps, and there are {count} steps left. Current Step: {curr_subtask}.<|image start|>"

        norm_image_list = []
        all_same_size = True
        for idx, im_path in enumerate(image_list):
            try:
                if isinstance(im_path, str):
                    image = Image.open(im_path)
                else:
                    video_path, frame_idx = im_path
                    video = VideoFileClip(video_path)
                    frame = video.get_frame(frame_idx / video.fps)
                    image = Image.fromarray(frame)
                    video.close()
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                # 将图片resize成512x512
                # image = image.resize((512, 512), Image.BICUBIC) # TODO: delete if mix-resolution is fixed
                image = smart_resize(image, area=area, ds_factor=ds_factor)  # 512*512 改 area
                image = normalize(image)
            except Exception as e:
                print(f"Error processing {im_path}: {e}")
                raise e
            norm_image_list.append(image)
            if idx > 0 and image.shape != norm_image_list[0].shape:
                all_same_size = False

        if not image_list:
            images = torch.Tensor([])  # 返回空张量，或你可以返回 None / raise 异常
        else:
            if all_same_size:
                images = torch.Tensor(np.stack(norm_image_list))
            else:
                images = [torch.Tensor(im).unsqueeze(0) for im in norm_image_list]

        for i, image in enumerate(images):
            print(f"-> prepare_data {i = }, {image.shape = }", flush=True)
            # 1, 3, h, w

        return {
            "text_prompt": text_prompt,
            "visual_placeholder": visual_placeholder,
            "supervised_start": supervised_start,
            "supervised_end": supervised_end,
            "images": images,
            "subtask_count": count,
            "uncond_text": uncond_text
        }

    @torch.inference_mode()
    def tokenize_image(
            image: torch.Tensor,
            model: torch.nn.Module,
            mini_batch_size: int = 16,
    ):
        """IBQ tokenize images"""
        batch_size = image.shape[0]
        token_ids = None
        for st_idx in range(0, batch_size, mini_batch_size):
            image_batch = image[st_idx:st_idx + mini_batch_size]
            with torch.no_grad():  # wyz
                quant, _, (_, _, _token_ids) = model.encode(image_batch)
            b, _, h, w = quant.shape
            _token_ids = _token_ids.view(b, h, w)
            token_ids = torch.cat([token_ids, _token_ids], dim=0) if token_ids is not None else _token_ids

        if token_ids is not None:
            return token_ids.detach().cpu().numpy()
        else:
            return np.array([])

    def pack(data_pack, tokenizer_max_seq_len=10240):
        """packing samples"""
        from utils.pack_sft import parse_string, format_image_string, SUPERVISION_START, SUPERVISION_END

        tokenizer_path = "./emu3_tokenizer/tokenizer_config"
        template = EasyDict(
            image_token="<|visual token {token_id:0>6d}|>",
            image_string="{image_start}{token_height}*{token_width}{image_token}{token_str}{image_end}",
        )

        from emu3_tokenizer.mllm2 import Emu3Tokenizer
        tokenizer = Emu3Tokenizer.from_pretrained(tokenizer_path)

        max_seq_len = tokenizer_max_seq_len
        sup_start_token_id, sup_end_token_id = tokenizer(SUPERVISION_START + SUPERVISION_END)["input_ids"]

        text_prompt = data_pack["text_prompt"]
        visual_placeholder = data_pack["visual_placeholder"]
        supervised_start = data_pack["supervised_start"]
        supervised_end = data_pack["supervised_end"]
        image_tokens = data_pack["image_tokens"]

        prompt_with_supervision_flag = parse_string(
            text_prompt,
            visual_placeholder,
            supervised_start,
            supervised_end,
        )

        is_image, token_list, image_index = False, [], 0
        is_break, prev_supervision_flag = False, 0
        for prompt_part, supervision_flag in prompt_with_supervision_flag:
            # tokenize parts and add supervision token according to supervision_flag
            if prompt_part == visual_placeholder:
                is_image = True
                image_string = format_image_string(template, tokenizer, image_tokens[image_index])
                token_ids = tokenizer(image_string)["input_ids"]
                image_index += 1
            else:
                is_image = False
                token_ids = tokenizer(prompt_part)["input_ids"]

            virtual_token_length = len(token_ids) + (2 if supervision_flag == 1 else 0)
            if is_image and len(token_list) + virtual_token_length > max_seq_len:
                break
            elif (_tmp_token_num := len(token_list) + virtual_token_length) > max_seq_len:
                token_ids = token_ids[:max_seq_len - _tmp_token_num]
                is_break = True

            if supervision_flag == 1:
                if prev_supervision_flag == 1:
                    token_list = token_list[:-1] + token_ids  # + [sup_end_token_id]
                else:
                    token_list += [sup_start_token_id] + token_ids  # + [sup_end_token_id]
            else:
                token_list += token_ids

            prev_supervision_flag = supervision_flag

            if is_break:
                break

        token_list = [tokenizer.bos_token_id] + token_list  # + [sup_start_token_id]  # + [tokenizer.eos_token_id]
        token_list = np.array(token_list)
        return token_list

    # --------------------------------------------------------------------------------------------------------------

    sample = prepare_data(sample)
    count = sample["subtask_count"]
    uncond_text = sample["uncond_text"]
    token_ids = list()
    for im in sample["images"]:
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        token_ids.append(tokenize_image(im.to(vq_model.quant_conv.bias.device), vq_model))
    # token_ids = [tokenize_image(im.to(vq_model.quant_conv.bias.device).unsqueeze(0), vq_model) for im in sample["images"]]
    token_ids = [_ti[0] for _ti in token_ids]

    data_pack = {
        "text_prompt": sample["text_prompt"],
        "visual_placeholder": sample["visual_placeholder"],
        "supervised_start": sample["supervised_start"],
        "supervised_end": sample["supervised_end"],
        "image_tokens": token_ids,
    }
    # print('prompt:', data_pack['text_prompt'])

    seq = pack(data_pack)
    return seq, uncond_text

def split_seq_by_image(seq, image_start_id, image_end_id):
    split_indices = []
    inside_image = False

    for i, token in enumerate(seq):
        if token == image_start_id:
            split_indices.append(i)
            inside_image = True
        elif token == image_end_id:
            split_indices.append(i + 1)
            inside_image = False

    # Return original sequence if no image tokens found
    if not split_indices:
        return [seq]

    split_indices = [0] + split_indices + [len(seq)]
    seqs = [seq[split_indices[i]:split_indices[i + 1]] for i in range(len(split_indices) - 1) if
            split_indices[i] != split_indices[i + 1]]
    return seqs

def package(task, task_imgs, steps, step_imgs, datasets, tag):
    text_prompt = ""
    image_list = []
    vis_plh = ''

    for simg in task_imgs:
        vis_plh += '<|VIS_PLH|>'
        image_list.append(simg)

    sub_text_prompt = "You are a helpful assistant. USER: " + f"{task}{vis_plh}" + ' ASSISTANT: '
    text_prompt = sub_text_prompt + text_prompt

    for i, (step, step_img) in enumerate(zip(steps, step_imgs)):
        vis_plh = ''
        for simg in step_img:
            vis_plh += '<|VIS_PLH|>'
            image_list.append(simg)

        sub_text_prompt = f"<|extra_100|>{step}{vis_plh}<|extra_101|>"
        text_prompt += sub_text_prompt

    assert text_prompt.count('<|VIS_PLH|>') == len(image_list)
    data = {
        "text_prompt": text_prompt,
        "visual_placeholder": "<|VIS_PLH|>",
        "supervised_start": "<|extra_100|>",
        "supervised_end": "<|extra_101|>",
        "image_list": image_list,
        "datasets": datasets,
        "tag": tag
    }

    return data

def load_emu_model():
    import config as cfg

    os.makedirs(cfg.save_root, exist_ok=True)
    print(f"-> {cfg.save_root = }", flush=True)

    log_with_time(f"Start loading model...")
    print(cfg.emu_tokenizer_path)
    model_load_start = time.time()
    emu3, emu3_tokenizer, vq_model = build_emu3p5_vllm(
        cfg.emu_model_path,
        cfg.emu_tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        vq_device=cfg.vq_device,
        seed=cfg.seed,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=0.8,
    )
    model_load_time = time.time() - model_load_start
    log_with_time(f"Model loading complete. Time consumed: {model_load_time} second")

    if not hasattr(cfg, "special_token_ids"):
        cfg.special_token_ids = EasyDict()
        for k, v in cfg.special_tokens.items():
            cfg.special_token_ids[k] = emu3_tokenizer.encode(v)[0]

    random.seed(cfg.seed)
    return emu3, emu3_tokenizer, vq_model, cfg