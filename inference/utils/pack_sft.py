import argparse
import os
import os.path as osp
import sys
import re
from pathlib import Path
import json

import numpy as np
np.set_printoptions(threshold=np.inf)
sys.path.insert(0, os.getcwd() + "/datasets")
sys.path.insert(0, "/share/project/lxh/project/QL/emu_robotics/copy/MMDataFactory_wyz/MMDataFactory/datasets")
import megatron_data as indexed_dataset
from omegaconf import OmegaConf
import torch

# sys.path.append("/share/project/zhangfan/codes/Emu3.5")
sys.path.insert(0, "/share/project/lxh/project/QL/emu_robotics/copy/zhangfan_emu3")
# from emu3.mllm import Emu3Tokenizer
# from emu3.mllm2 import Emu3Tokenizer
from mllm2 import Emu3Tokenizer

SUPERVISION_START = "<|extra_100|>"
SUPERVISION_END = "<|extra_101|>"


import re
def clean_think(input_str):
    # 使用正则表达式匹配并移除<|extra_50|>和<|extra_51|>之间的所有内容
    pattern = r'<\|extra_50\|>.*?<\|extra_51\|>'
    cleaned_str = re.sub(pattern, '', input_str, flags=re.DOTALL)
    return cleaned_str


def parse_string(s, vis_token="<|VIS_PLH|>", sup_start=SUPERVISION_START, sup_end=SUPERVISION_END):
    output = []
    supervision = 0

    vis_plh_re = re.escape(vis_token)
    sup_start_re = re.escape(sup_start)
    sup_end_re = re.escape(sup_end)

    # Split on all special markers

    parts = re.split(f"({vis_plh_re}|{sup_start_re}|{sup_end_re})", s)

    for part in parts:
        if part.strip() == "":
            continue

        if part == sup_start:
            supervision = 1
        elif part == sup_end:
            supervision = 0
        else:
            output.append([part, supervision])

    return output


def format_image_string(cfg, tokenizer, image_tokens):
    image_string = ""
    h, w = image_tokens.shape

    # 逐行 format
    for _h in range(h):
        row_string = ""
        for _w in range(w):
            # 把每个 token id 套上 str format (里面数字填充到 6 位), 例如 123 -> "<|visual token 000123|>"
            row_string += cfg.image_token.format(token_id=image_tokens[_h, _w])  # "<|visual token {token_id:0>6d}|>"
        # 在每行末尾加一个 end of line token, 即 "eol_token_id": 151846
        if _h < h - 1:
            row_string += tokenizer.eol_token
        image_string += row_string

    # 最终 format
    # special_token_ids.image_token = 151851, 
    # special_token_ids.image_start_token = 151852, "boi_token_id": 151852
    # special_token_ids.image_end_token = 151853, "eoi_token_id": 151853
    return cfg.image_string.format(  # "{image_start}{token_height}*{token_width}{image_token}{token_str}{image_end}"
        image_start=tokenizer.boi_token,
        token_height=h,
        token_width=w,
        image_token=tokenizer.img_token,
        token_str=image_string,
        image_end=tokenizer.eoi_token,
    )


def main(args: argparse.Namespace):
    config = OmegaConf.load(args.data_cfg).packing
    pts = os.listdir(config.input_dir)
    pts = [os.path.join(config.input_dir, pt) for pt in pts]
    pts = sorted(pts)
    pts = pts[args.worker_id::args.num_workers]

    tokenizer = Emu3Tokenizer.from_pretrained(config.tokenizer.path)
    os.makedirs(config.output_dir, exist_ok=True)

    # v2 - zh
    # "/share/project/wyz/datasets/ibq_token_think_v2_zh/ibq_token_area_768x768_packed_10752/deepfashion_ipimg_27340"
    
    # name = config.output_dir.split("/")[-1]
    # if name in {"deepfashion_ipimg_27340", "shop_product_ip_23792", "viton_hd_ip_23294"}:
    #     json_path = "/share/project/wyz/sr/jsons_unisft_en_to_zh/virtual_try_off_df_sp_vh.json"
    # else:
    #     json_path = f"/share/project/wyz/sr/jsons_unisft_en_to_zh/{name}.json"
    # with open(json_path, 'r') as ff:
    #     zh_dict = json.load(ff)
    #     print(f"-> {len(zh_dict) = }, {json_path = } -------------------------------------------------------------", flush=True)

    # "Inside Google's AI research center, 'SVM' written in icing on a cake.": "You are a helpful assistant. USER: 在谷歌人工智能研究中心内，蛋糕上的糖霜写着'SVM'。 ASSISTANT: <|extra_100|><|VIS_PLH|><|extra_101|>",

    max_seq_len = config.tokenizer.max_seq_len - 2  # -2 for bos and eos
    sup_start_token_id, sup_end_token_id = tokenizer(SUPERVISION_START + SUPERVISION_END)["input_ids"]

    print(f"-> {args.worker_id = }, {len(pts) = }\n{max_seq_len = }, {SUPERVISION_START = }, {sup_start_token_id = }, {SUPERVISION_END = }, {sup_end_token_id = }", flush=True)
    # , {pts[0] = }
    for pt in pts:
        data = torch.load(pt)
        filename = osp.splitext(Path(pt).stem)[0]

        cnt = 0

        cnt_match_zh = 0

        if os.path.exists(osp.join(config.output_dir, f"{filename}.idx")):
            print(f"-> {args.worker_id = }, exist {config.output_dir}/{filename}.idx with {cnt = }", flush=True)
            continue
        print(f"-> {args.worker_id = }, packing {config.output_dir}/{filename} ... ", flush=True)

        builder = indexed_dataset.MMapIndexedDatasetBuilder(
            osp.join(config.output_dir, f"{filename}.bin"),
        )

        for idx, d in enumerate(data):
            text_prompt = d["text_prompt"]



            # dfm 10752
            # text_prompt = text_prompt.replace("You are a helpful assistant for x2i task. ", "")


            # v2
            # text_prompt = clean_think(d["text_prompt"])  # wyz no think 20250901
            # text_prompt = "You are a helpful assistant. " + text_prompt.replace("\nASSISTANT:", " ASSISTANT:")

            # v2 - t2i
            # text_prompt = text_prompt.replace("<|extra_100|> <|VIS_PLH|> <|extra_101|>", "").strip()
            # text_prompt = f"You are a helpful assistant. USER: {text_prompt}. ASSISTANT: <|extra_100|><|VIS_PLH|><|extra_101|>"

            # if idx == 0:
            #     print(f"-> new text prompt: {text_prompt}", flush=True)

            # v2 - zh
            # t_prompt = clean_think(d["text_prompt"]).replace("\nASSISTANT: <|extra_100|><|VIS_PLH|><|extra_101|>", "").replace("USER: ", "").replace("<|extra_100|> <|VIS_PLH|> <|extra_101|>", "").replace(" <|VIS_PLH|> ", "<|VIS_PLH|>").replace(" <|VIS_PLH|>", "<|VIS_PLH|>").replace("<|VIS_PLH|> ", "<|VIS_PLH|>").strip()
            # if t_prompt in zh_dict:
            #     text_prompt = zh_dict[t_prompt]
            #     if idx == 0:
            #         print(f"-> {t_prompt = }, new text prompt: {text_prompt}", flush=True)
            #         # exit()
            # else:
            #     text_prompt = f"You are a helpful assistant. USER: {t_prompt}. ASSISTANT: <|extra_100|><|VIS_PLH|><|extra_101|>"

            # {
            #     "<|VIS_PLH|>Put a small golden crown on the cat and a little red royal robe": 
            #     "You are a helpful assistant. USER: <|VIS_PLH|>在猫头上放一个小金冠，并给它穿上一件红色的皇家斗篷。 ASSISTANT: <|extra_100|><|VIS_PLH|><|extra_101|>",
            # - zh

            # unisft - 后半部分 ---------
            # text_prompt = text_prompt.replace("You are a helpful assistant", "You are a helpful assistant for x2i task")


            # unisft zh
            # t_prompt = d["text_prompt"].replace(" ASSISTANT: <|extra_100|><|VIS_PLH|><|extra_101|>", "").replace("You are a helpful assistant. USER: ", "").replace("You are a helpful assistant for x2i task. USER: ", "").strip()
            # if idx == 0:
            #     print(f"-> {t_prompt = }", flush=True)
            # if t_prompt in zh_dict:
            #     # text_prompt = zh_dict[t_prompt]
            #     text_prompt = zh_dict[t_prompt].replace("You are a helpful assistant", "You are a helpful assistant for x2i task")
            #     cnt_match_zh += 1
            #     if idx == 0:
            #         print(f"-> {t_prompt = }, new text prompt: {text_prompt}", flush=True)
            #         # exit()
            # else:
            #     text_prompt = f"You are a helpful assistant for x2i task. USER: {t_prompt}. ASSISTANT: <|extra_100|><|VIS_PLH|><|extra_101|>"
                # text_prompt = f"You are a helpful assistant. USER: {t_prompt}. ASSISTANT: <|extra_100|><|VIS_PLH|><|extra_101|>"

            # text_prompt = "You are a helpful assistant. " + text_prompt.replace("\nASSISTANT:", " ASSISTANT:")

            # if idx == 0:
            #     print(f"-> new text prompt: {text_prompt}", flush=True)

            # text_prompt = text_prompt.replace("You are a helpful assistant for x2i task", "You are a helpful assistant")

            text_prompt = text_prompt.replace("You are a helpful assistant", "You are a helpful assistant for x2i task")


            visual_placeholder = d["visual_placeholder"]
            supervised_start = d["supervised_start"]
            supervised_end = d["supervised_end"]
            image_tokens = d["image_tokens"]
            
            if type(image_tokens) is not list:
                image_tokens = [image_tokens]
            # print(supervised_start, supervised_end)

            # wyz
            try:
            # if True:
                # print(text_prompt)
                prompt_with_supervision_flag = parse_string(
                    text_prompt,  # text_prompt == None 的 sample 因有效数据不完整被丢弃
                    visual_placeholder,
                    supervised_start,
                    supervised_end,
                )

                is_image, token_list, image_index = False, list(), 0
                is_break, prev_supervision_flag = False, 0
                for prompt_part, supervision_flag in prompt_with_supervision_flag:
                    # tokenize parts and add supervision token according to supervision_flag
                    if prompt_part == visual_placeholder:
                        is_image = True
                        image_string = format_image_string(config.template, tokenizer, image_tokens[image_index])
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
                            token_list = token_list[:-1] + token_ids + [sup_end_token_id]
                        else:
                            token_list += [sup_start_token_id] + token_ids + [sup_end_token_id]
                    else:
                        token_list += token_ids
                    
                    prev_supervision_flag = supervision_flag
                    # print(prompt_part, supervision_flag)
                    # print(token_list)
                    # input()
                    
                    if is_break:
                        break
                
            except Exception as e:
                print(f"-> error {e = } | {text_prompt = }", flush=True)
                continue


            token_list = [tokenizer.bos_token_id] + token_list + [tokenizer.eos_token_id]
            if len(token_list) < config.tokenizer.max_seq_len:
                token_list += [tokenizer.pad_token_id] * (config.tokenizer.max_seq_len - len(token_list))
            
            if len(token_list) != config.tokenizer.max_seq_len:
                print(f"len(token_list) != config.tokenizer.max_seq_len at {filename}")
                continue

            # print(f"process {idx} of {filename} done")

            token_list = np.array(token_list)
            builder.add_item(token_list)
            builder.end_document()

            cnt += 1
            
        builder.finalize(osp.join(config.output_dir, f"{filename}.idx"))
        print(f"-> {args.worker_id = }, finish {config.output_dir}/{filename}.idx with {cnt = }", flush=True)
    print(f"-> {args.worker_id = }, all pack finish !!!", flush=True)

    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str, required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    main(args)


