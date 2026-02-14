#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: seq_utils.py
# Created Date: Wednesday December 18th 2024
# Author: Zhengxiong Luo
# Contact: <zxluo@baai.ac.cn>
# 
# Last Modified: Monday January 27th 2025 1:10:00 pm
# 
# Copyright (c) 2024 Beijing Academy of Artificial Intelligence (BAAI)
# All rights reserved.
# -----
# HISTORY:
# Date      	 By	Comments
# ----------	---	----------------------------------------------------------
###

import os
import os.path as osp
import re

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .model_utils_32b import decode_visual_seq


def find_list_indexes(lst, element):
    return np.where(np.array(lst) == element)[0].tolist()


def find_string_indices(string, substring):
    return [match.start() for match in re.finditer(re.escape(substring), string)]


def split_seq_by_timestamp(seq, ts_token_id):
    ts_idxes = find_list_indexes(seq, ts_token_id)
    sub_seq_start_idxes = ts_idxes[::2]  # each sub sequence has two timestamp tokens

    seqs = []
    for idx, st in enumerate(sub_seq_start_idxes[:-1]):
        ed = sub_seq_start_idxes[idx + 1]
        seqs.append(seq[st:ed])
    seqs.append(seq[sub_seq_start_idxes[-1] :])
    return seqs


def split_seq_by_image(seq, image_start_id, image_end_id):
    """
    根据 image_start_id 和 image_end_id 对 numpy 数组 seq 进行分割。
    """
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
    
    seqs = [seq[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices) - 1) if split_indices[i] != split_indices[i+1]]
    
    return seqs

def add_fake_timestamp(seq, tokenizer, ts_token, image_start_id, image_end_id):
    """
    Add fake timestamps before each image pair in the sequence.
    Args:
        seq: Input sequence [L]
        ts_token: Timestamp token to be added
        image_start_id: Token ID indicating start of image
        image_end_id: Token ID indicating end of image
    Returns:
        Modified sequence with fake timestamps [L']
    """
    # First split the sequence by image tokens
    seqs = split_seq_by_image(seq, image_start_id, image_end_id)
    
    # Initialize result sequence
    result = []
    current_time = 0.0
    
    for i, sub_seq in enumerate(seqs):
        sub_seq = [ts_token] + tokenizer.encode(f"{current_time:.2f}") + [ts_token] + sub_seq.tolist()
        current_time += 1.0
        result.extend(sub_seq)
    
    return np.array(result)


def vis_interleve_sample(vq_model, tokenizer, seq_list, device, special_token_ids, vq_type):
    images = []
    for seq in seq_list:
        try:
            ts_idxes = find_list_indexes(seq, special_token_ids.ts_token)
            ts = float(tokenizer.decode(seq[ts_idxes[0] + 1 : ts_idxes[-1]]))
            seq = seq[ts_idxes[-1] + 1 :].tolist()
        except Exception as e:
            print("Without Timestamp:", e)
            seq = seq.tolist()
        
        seq = list(filter(lambda x: x != special_token_ids.pad_token, seq))

        # image
        if (
            special_token_ids.image_token in seq and 
            special_token_ids.image_start_token in seq and 
            special_token_ids.image_end_token in seq
        ):
            start_idx = seq.index(special_token_ids.image_start_token)  # boi_token_id = "<|image start|>" = special_token_ids.image_start_token = 151852
            ed_idx = seq.index(special_token_ids.image_end_token)       # eoi_token_id = "<|image end|>"   = special_token_ids.image_end_token = 151853
            img_token_idx = seq.index(special_token_ids.image_token)    #                "<|image token|>" = special_token_ids.image_token = 151851

            vis_tokens = tokenizer.decode(seq[img_token_idx+1: ed_idx])   # decode <|image token|> 
            res_info = tokenizer.decode(seq[start_idx+1: img_token_idx])  # decode <|image start|> 到第 1 个 <|image token|> 之前的参考信息 tokens

            if "FPS" in res_info:
                shape = tuple(map(int, res_info.split("FPS")[-1].split("*")))
            else:
                shape = (1, *tuple(map(int, res_info.split("*"))))
            try:
                image = decode_visual_seq(vq_model, vis_tokens, shape, device, vq_type=vq_type)

            except Exception as e:
                print(f"ERROR DECODE, {e}, Skip the current decoding")
                continue
                
            if len(image.shape) == 3:
                images.append(image)
            elif len(image.shape) == 4:
                images.extend(image.unbind(0))
        
        # text
        else:
            seqnp = np.array(seq)
            if len(seqnp[seqnp>special_token_ids.image_end_token]) > 0:
                try:
                    text = tokenizer.decode(seqnp[seqnp<=special_token_ids.image_end_token])
                    text = f"{text} !!!!! BAD VISUAL !!!!!!: {len(seqnp[seqnp>special_token_ids.image_end_token])}"
                    images.append(text)
                except Exception as e:
                    print("ERROR DECODE", e)
                    images.append(tokenizer.decode(seq))
            else:
                images.append(tokenizer.decode(seq))


    shape = (512, 512, 3)
    for item in images:
        if isinstance(item, torch.Tensor):
            shape = item.shape

    for idx, item in enumerate(images):
        if isinstance(item, torch.Tensor):
            item = item.cpu().numpy()
            images[idx] = item
        elif isinstance(item, str):
            images[idx] = plot_string(item, image_size=(shape[1], shape[0]))

    return images



def decode_generated_results(vq_model, tokenizer, seq_list, device, special_token_ids, vq_type):
    results = []
    for seq in seq_list:
        try:
            ts_idxes = find_list_indexes(seq, special_token_ids.ts_token)
            ts = float(tokenizer.decode(seq[ts_idxes[0] + 1 : ts_idxes[-1]]))
            seq = seq[ts_idxes[-1] + 1 :].tolist()
        except Exception as e:
            print("Without Timestamp:", e)
            seq = seq.tolist()
        
        seq = list(filter(lambda x: x != special_token_ids.pad_token, seq))
        if (
            special_token_ids.image_token in seq and 
            special_token_ids.image_start_token in seq and 
            special_token_ids.image_end_token in seq
        ):
            start_idx = seq.index(special_token_ids.image_start_token)
            ed_idx = seq.index(special_token_ids.image_end_token)
            img_token_idx = seq.index(special_token_ids.image_token)

            vis_tokens = tokenizer.decode(seq[img_token_idx + 1 : ed_idx])
            res_info = tokenizer.decode(
                seq[start_idx + 1  : img_token_idx]
            )

            if "FPS" in res_info:
                shape = tuple(map(int, res_info.split("FPS")[-1].split("*")))
            else:
                shape = (1, *tuple(map(int, res_info.split("*"))))
            try:
                image = decode_visual_seq(vq_model, vis_tokens, shape, device, vq_type=vq_type)
            except Exception as e:
                print(f"ERROR DECODE, {e}, Skip the current decoding")
                continue
            if len(image.shape) == 3:
                results.append(
                    {"image": image}
                )
            elif len(image.shape) == 4:
                results.append(
                    {"image": image.unbind(0)}
                )
        else:
            seqnp = np.array(seq)
            if len(seqnp[seqnp>special_token_ids.image_end_token]) > 0:
                try:
                    text = tokenizer.decode(seqnp[seqnp<=special_token_ids.image_end_token])
                    text = f"{text} !!!!! BAD VISUAL !!!!!!: {len(seqnp[seqnp>special_token_ids.image_end_token])}"
                    results.append(
                        {"bad_image": image}
                    )
                except Exception as e:
                    print("ERROR DECODE", e)
                    results.append(
                        {"bad_image": tokenizer.decode(seq)}
                    )
            else:
                # 可能是坏掉的text，也可能图片只生成到<image start>xx*xx<image token>,这时候还没有生成具体的token，但一般都是最后结尾部分
                results.append(
                    {"text": tokenizer.decode(seq)}
                )


    return results


def save_image_list_to_video(images, path, fps=1):

    os.makedirs(osp.dirname(path), exist_ok=True)
    
    if '.mp4' not in path and len(images) == 1:
        Image.fromarray(images[0]).save(path)
        return

    
    func = lambda x: (
        x.cpu().numpy().astype(np.uint8)
        if isinstance(x, torch.Tensor)
        else x.astype(np.uint8)
    )
    images = list(map(func, images))

    with imageio.get_writer(path, fps=fps, mode="I") as writer:
        for image in images:
            # print(f"-> decode image shape = {image.shape}", flush=True)
            writer.append_data(image)

            


def save_image_list_to_image(images, path, fps=1, only_save_first_image=False):  # wyz

    os.makedirs(osp.dirname(path), exist_ok=True)
    
    if '.mp4' not in path and len(images) == 1:
        Image.fromarray(images[0]).save(path)
        return

    func = lambda x: (
        x.cpu().numpy().astype(np.uint8)
        if isinstance(x, torch.Tensor)
        else x.astype(np.uint8)
    )
    images = list(map(func, images))

    for idx, image in enumerate(images):
        cur_path = path.replace(".mp4", f'_{idx}.png')
        Image.fromarray(image).save(cur_path)
        if only_save_first_image is True:
            print(f"-> only_save_first_image {idx=} in {cur_path}", flush=True)
            break
        print(f"-> save img {idx = } in {cur_path}", flush=True)





def wrap_text(draw, text, font, max_width):
    """
    Wrap the input text to fit within the given width, breaking it into multiple lines.
    """
    lines = []
    words = text.split()
    current_line = ""

    for word in words:
        # Check if the word fits in the current line
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            # If it fits, add it to the current line
            current_line = test_line
        else:
            # If it doesn't fit, start a new line
            if current_line:
                lines.append(current_line)
            current_line = word

    # Add the last line
    if current_line:
        lines.append(current_line)

    return lines


def plot_string(
    string,
    font_path=None,
    font_size=40,
    image_size=(500, 100),
    bg_color="white",
    text_color="black",
):
    """
    Render a string onto an image using Pillow and return it as a NumPy array.
    The text will wrap to the next line when it exceeds the image width.
    """
    img = Image.new("RGB", image_size, color=bg_color)
    draw = ImageDraw.Draw(img)

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    max_width = image_size[0] - 20  # 20px padding on the left and right
    lines = wrap_text(draw, string, font, max_width)

    line_height = draw.textbbox((0, 0), "A", font=font)[
        3
    ]  # Height of one line (using 'A' to get the height)
    total_text_height = line_height * len(lines)

    y_offset = (image_size[1] - total_text_height) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x_offset = (image_size[0] - text_width) // 2

        draw.text((x_offset, y_offset), line, fill=text_color, font=font)

        y_offset += line_height

    return np.array(img)