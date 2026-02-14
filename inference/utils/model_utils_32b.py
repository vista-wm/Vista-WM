#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: model_utils.py
# Created Date: Wednesday December 18th 2024
# Author: Zhengxiong Luo
# Contact: <zxluo@baai.ac.cn>
#
# Last Modified: Monday January 27th 2025 12:53:16 pm
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
from omegaconf import OmegaConf
import re
import torch
from einops import rearrange
from transformers import AutoConfig, AutoTokenizer
from transformers.generation import (LogitsProcessorList,
                                     PrefixConstrainedLogitsProcessor, TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper)
from tqdm import tqdm
import torch.nn.functional as F
from vllm import LLM, SamplingParams
import numpy as np
import vllm
print("vllm version:", vllm.__path__)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from emu3p5 import Emu3ForCausalLM, Emu3Config


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k: keep only top k tokens with highest probability (top-k filtering).
            Must be >0. Default to 0 (no filtering).
        top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Must be in [0, 1]. Default to 1.
        min_tokens_to_keep: Minimum number of tokens that cannot be filtered. Default to 1.
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def build_tokenizer(tokenizer_path, vq_type=None):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=os.path.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )

    if vq_type is None:
        return tokenizer

    if vq_type == "movqgan":
        from external.movqgan_video import get_movqgan_model
        vq_model = get_movqgan_model(
            "v0",
            device="cuda",
            cache_dir="/share/project/cyf/scripts/MMInference/external/emu3/movqgan_video/cache_dir",
            sane_index_shape=True,
        )
    elif vq_type == "ibq":
        from external.IBQTokenizer.src.ibq import IBQ
        cfg = OmegaConf.load(osp.join("/share/project/zhangfan/weights/Emu3.5-Tokenizer/IBQ-XL-f16c131k-FI/", "config.yaml"))
        vq_model = IBQ(**cfg.model.init_args).to("cuda")
        ckpt = torch.load(osp.join("/share/project/zhangfan/weights/Emu3.5-Tokenizer/IBQ-XL-f16c131k-FI/", "model.ckpt"), map_location="cpu")["state_dict"]
        vq_model.load_state_dict(ckpt)

    return tokenizer, vq_model


def build_emu3(model_path, tokenizer_path, local_rank=0, bad_model_path=None, vq_type="ibq", vllm_engine=False):
    if isinstance(local_rank, int):
        device_map = f"cuda:{local_rank}"
    else:
        device_map = local_rank
    print(f"-> build_emu3 {device_map = }", flush=True)
   
    # if 'emu3p5' in model_path:
    if 'emu3p5' in model_path and '14b' not in model_path:  # emu3.5 IBQ 进 
        print("load emu3p5 4b or 32b model!")
        # /share/project/wyz/projects/emu3p5/MMInference/external/emu3p5/modeling_emu3.py -> Emu3ForCausalLM
        # /share/project/wyz/projects/emu3p5/MMInference/external/emu3p5/configuration_emu3.py -> Emu3Config
        # from external.emu3p5 import Emu3ForCausalLM, Emu3Config

        model_config = Emu3Config.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if vllm_engine:
            model = None
        else:
            model = Emu3ForCausalLM.from_pretrained(  # -> 
                model_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                attn_implementation="flash_attention_2"
            )

    else: # emu3 MoVQ 进
        model_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        from external.emu3 import MMLlamaForCausalLM

        if vllm_engine:
            model = None
        else:   
            model = MMLlamaForCausalLM.from_pretrained(
                model_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )

    # print(f"-> build emu3 ... {tokenizer_path = }", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,  # vision tokenizer path: emu_tokenizer_path = "/share/project/cyf/scripts/MMInference/external/tokenizer_emu3_ibq"
        special_tokens_file=os.path.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )

    # vq_type = getattr(model.config, "vq_type", "ibq")
    if vq_type == "movqgan":
        print("[tokenizer] load movqgan model!")
        from external.movqgan_video import get_movqgan_model
        vq_model = get_movqgan_model(
            model.config.movqgan_name,
            device=model.device,
            cache_dir=model.config.movqgan_cache_dir,
            sane_index_shape=True,
        )
    elif vq_type == "ibq":
        print("[tokenizer] load ibq model!")
        from external.IBQTokenizer.src.ibq import IBQ

        if model is not None and hasattr(model.config, "ibq_cache_dir"):
            ibq_cache_dir = model.config.ibq_cache_dir
        else:
            ibq_cache_dir = "/share/project/cyf/scripts/MMInference/external/IBQTokenizer"

        cfg = OmegaConf.load(osp.join(ibq_cache_dir, "config.yaml"))

        vq_model = IBQ(**cfg.model.init_args).to(torch.device("cuda:0"))
        ckpt = torch.load(osp.join(ibq_cache_dir, "model.ckpt"), map_location="cpu")["state_dict"]
        vq_model.load_state_dict(ckpt)

    if model is not None:   
        model.init_vision(tokenizer, vq_model)


    if vllm_engine:
        if model is not None:
            model.model.to('cpu')
            model.lm_head.to('cpu')
        
        # resolution tokens
        resolution_map = {}
        resolution_str = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*"]
        for digit_str in resolution_str:
            resolution_map[tokenizer.encode(digit_str)[0]] = digit_str

        vllm_model = LLM(
            f"{model_path}",
            # f"{tokenizer_path}",
            trust_remote_code=True,
            dtype='bfloat16',
            disable_log_stats=False,
            tensor_parallel_size=2,
            max_num_batched_tokens=26000,
            max_model_len=16384,
            # max_seq_len_to_capture=2048,
            # swap_space=80,
            gpu_memory_utilization=0.8,
            enable_chunked_prefill=False,
            max_num_seqs=2,
            # seed=19971104,
            # quantization="fp8",'
            generation_config='vllm',
            scheduler_cls="vllm.v1.core.sched.batch_scheduler.Scheduler",
            compilation_config={
                "full_cuda_graph": True,
                "backend": "cudagraph",
                "cudagraph_capture_sizes": [1, 2],
            },
            additional_config={
                "boi_token_id": tokenizer.encode("<|image start|>")[0],
                "soi_token_id": tokenizer.encode("<|image token|>")[0],
                "eol_token_id": tokenizer.encode("<|extra_200|>")[0],
                "eoi_token_id": tokenizer.encode("<|image end|>")[0],
                "resolution_map": resolution_map,
            },
        )
        vllm_model.set_tokenizer(tokenizer)
        
        return vllm_model, tokenizer, vq_model


    if bad_model_path is None:
        return model, tokenizer, None

    bad_model = MMLlamaForCausalLM.from_pretrained(
        bad_model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    return model, tokenizer, bad_model



def build_movqgan(device, dtype, MODEL_DIR):
    import sys

    sys.path.insert(0, MODEL_DIR)
    import get_movqgan_model

    model = (
        get_movqgan_model(
            "v0",
            device=device,
            cache_dir=osp.join(MODEL_DIR, "cache_dir"),
            sane_index_shape=True,
        )
        .to(device)
        .to(dtype)
    )
    sys.path.pop(0)
    return model


@torch.inference_mode()
def decode_visual_token_movqgan(model, tokens):
    if len(tokens.shape) == 2:
        H, W = tokens.shape
        B = T = 1
    elif len(tokens.shape) == 3:
        B, H, W = tokens.shape
        T = 1
    elif len(tokens.shape) == 4:
        B, T, H, W = tokens.shape
    else:
        raise ValueError(f"expected tokens with 2 or 3 or 4 dimensions but got {len(tokens.shape)}")

    quant = model.quantize.embedding(
        tokens.flatten()
        ).view((B * T, 1, H, W, 4))
    quant = rearrange(quant, 'b t h w c -> b c t h w').contiguous()
    quant2 = model.post_quant_3dconv(quant)

    quant = rearrange(quant, "b c t h w -> b t c h w")
    quant2 = rearrange(quant, "b c t h w -> b t c h w")
    dec = model.decoder(quant2, quant) # B*T*4 C H W

    images = rearrange(dec, "b*t c h w -> b t h w c", b = B)
    images = images.float().add(1).mul(127.5).clamp(0, 255).to(torch.uint8)

    images = images.squeeze()
    return images


def build_ibq_model(device, dtype, MODEL_DIR="minbpe"):
    import sys

    sys.path.insert(0, MODEL_DIR)
    from src.ibq import IBQ

    cfg = OmegaConf.load(osp.join(MODEL_DIR, "config.yaml"))
    model = IBQ(**cfg.model.init_args).to(device).to(dtype)

    ckpt = torch.load(osp.join(MODEL_DIR, "model.ckpt"), map_location="cpu")["state_dict"]

    model.load_state_dict(ckpt)
    sys.path.pop(0)
    return model


@torch.inference_mode()
def decode_visual_token_ibq(model, tokens):
    if len(tokens.shape) == 2:
        H, W = tokens.shape
        B = 1
    elif len(tokens.shape) == 3:
        B, H, W = tokens.shape
    else:
        raise ValueError(f"expected tokens with 2 or 3 dimensions but got {len(tokens.shape)}")

    images = model.decode_code(tokens.view(1, -1), shape=(B, H, W, 256))

    images = rearrange(images, "b c h w -> b h w c")
    images = images.float().add(1).mul(127.5).clamp(0, 255).to(torch.uint8)

    images = images.squeeze()
    return images


@torch.inference_mode()
def decode_visual_seq(vq_model, seq, shape, device, vq_type):
    pattern = r"<\|visual token (\d{6})\|>"
    token_ids = list(map(int, re.findall(pattern, seq)))

    token_ids = torch.tensor(token_ids).long().to(device).view(*shape)

    if vq_type == "ibq":
        decode_func = decode_visual_token_ibq
    elif vq_type == "movqgan":
        decode_func = decode_visual_token_movqgan
    else:
        raise TypeError(f"only support vq_type to be ibq or movgqn, but got {vq_type}")

    images = decode_func(vq_model, token_ids)
    return images



@torch.inference_mode()
def get_model_next_tokens(
    model,
    tokenizer,
    input_ids = [],
):
    input_ids = input_ids.to(model.device)

    next_token, next_token_logits = model.generate_next_token(input_ids, do_sample=True)


    return next_token, next_token_logits


@torch.inference_mode()
def get_model_loss(
    model,
    tokenizer,
    input_ids = [],
):
    if isinstance(model, vllm.LLM):
        print("vllm model")

        sampling_params = SamplingParams(
            best_of=1,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            stop_token_ids=[tokenizer.convert_tokens_to_ids('<|extra_204|>')],
            include_stop_str_in_output=True,
            max_tokens=1,
            detokenize=False,
            skip_special_tokens=False,
            guidance_scale=None,
            # prompt_logprobs=10,
        )

        inputs = [
            {
                "prompt_token_ids": input_ids[0].tolist(),
            }
        ]

        outputs = model.generate(inputs, sampling_params=sampling_params)
        import pdb
        pdb.set_trace()

        # Get logprobs from vLLM output
        logprobs = []
        for output in outputs:
            prompt_logprobs = output.prompt_logprobs
            for token_logprobs in prompt_logprobs:
                logprobs.append(-token_logprobs[output.prompt_token_ids[len(logprobs)]])

        return torch.tensor(logprobs, device=input_ids.device)

    else:
        input_ids = input_ids.to(model.device)

        outputs = model(input_ids, labels=input_ids, return_dict=True)

        loss = outputs.loss

        return loss


def get_model_response(
    model,
    tokenizer,
    generation_config,
    input_ids = [],
    negative_prompt=[],
    prefix_allowed_tokens_fn=None,
    logits_processors=[],
):
    input_ids_len = input_ids.shape[1]
    # input_ids = input_ids.to(model.device)

    if len(negative_prompt) > 0:
        negative_prompt_ids = model.mmencode(
            tokenizer,
            negative_prompt,
            [],
            return_tensors="pt",
            add_special_tokens=False,
        )
        # negative_prompt_ids = negative_prompt_ids.to(model.device)
        print("negative_prompt_ids:", negative_prompt_ids, flush=True)

    logits_processor = LogitsProcessorList()
    for processor in logits_processors:
        logits_processor.append(processor)
    if prefix_allowed_tokens_fn is not None:
        logits_processor.append(
            PrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn,
                num_beams=generation_config.num_beams
                // generation_config.num_beam_groups,
            )
        )

    token_ids = model.generate(
        input_ids,
        generation_config,
        logits_processor=logits_processor,
    )

    gen_token_ids = token_ids[:, input_ids_len:]
    print("gen_token_ids", f"{gen_token_ids.shape}", flush=True)
    return gen_token_ids




def get_model_response_vllm(
    model,
    tokenizer,
    generation_config,
    input_ids = [],
    negative_prompt=[],
    prefix_allowed_tokens_fn=None,
    logits_processors=[],
):

    sampling_params = SamplingParams(
        best_of=generation_config.num_beams // generation_config.num_beam_groups,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        top_k=generation_config.top_k,
        stop_token_ids=[tokenizer.convert_tokens_to_ids('<|extra_204|>')],
        include_stop_str_in_output=True,
        max_tokens=generation_config.max_new_tokens,
        detokenize=False,
        skip_special_tokens=False,
        guidance_scale=generation_config.guidance_scale,
    )
    print(f"{sampling_params=}")

    input_ids_len = input_ids.shape[1]
    input_ids = input_ids.to(model.device)

    if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1.0:

        if len(negative_prompt) == 0:
            negative_prompt = ["<|extra_203|>"]

        negative_prompt_ids = tokenizer.encode(negative_prompt, add_special_tokens=False)
        print("negative_prompt_ids:", negative_prompt_ids, flush=True)
        print("len(input_ids[0].tolist()):", len(input_ids[0].tolist()), flush=True)
        negative_prompt_ids = negative_prompt_ids.to(model.device)
        # print("negative_prompt_ids:", negative_prompt_ids, flush=True)

        inputs = [
            {
                "prompt_token_ids": input_ids[0].tolist(),
                "negative_prompt_token_ids": negative_prompt_ids,
            }
        ]
        # print("inputs:", inputs, flush=True)
    else:
        inputs = [
            {
                "prompt_token_ids": input_ids[0].tolist(),
            }
        ]

    gen_token_ids = model.generate(inputs, sampling_params=sampling_params)[0].outputs[0].token_ids
    gen_token_ids = np.array(gen_token_ids)

    print("gen_token_ids", f"{gen_token_ids.shape}", flush=True)

    return gen_token_ids

def get_model_response_with_type(
    model,
    tokenizer,
    generation_config1,
    generation_config2,
    input_ids = [],
    negative_prompt=[],
    prefix_allowed_tokens_fn=None,
    logits_processors=[],
):
    """
    Multi-stage generation with different parameters for text and image tokens.

    Args:
        model: The language model
        tokenizer: The tokenizer
        generation_config: Base generation config to modify
        input_ids: Input token ids
        negative_prompt: Negative prompt for guidance
        prefix_allowed_tokens_fn: Function to constrain allowed tokens
        logits_processors: List of logits processors
    """
    input_ids = input_ids.to(model.device)
    current_ids = input_ids.clone()

    # Get special token IDs
    image_start_id = tokenizer.encode("<|image start|>")[0]
    image_end_id = tokenizer.encode("<|image end|>")[0]
    eos_token_id = tokenizer.eos_token_id

    # Setup logits processors
    logits_processor = LogitsProcessorList()
    for processor in logits_processors:
        logits_processor.append(processor)
    if prefix_allowed_tokens_fn is not None:
        logits_processor.append(
            PrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn,
                num_beams=generation_config1.num_beams // generation_config1.num_beam_groups,
            )
        )

    generated_sequence = []
    current_config = generation_config1

    # Track sequences generated with each config
    config1_sequences = []
    config2_sequences = []
    current_segment = []

    # Create progress bar with unknown total
    pbar = tqdm(desc="Generating tokens", total=generation_config1.max_new_tokens)

    for _ in range(generation_config1.max_new_tokens):
        outputs = model.generate(
            current_ids,
            generation_config=current_config,
            logits_processor=logits_processor,
            max_new_tokens=1,  # Generate one token at a time
        )

        # Get the new token
        new_token = outputs[0][-1].item()
        generated_sequence.append(new_token)
        current_segment.append(new_token)
        current_ids = outputs

        # Update progress bar
        pbar.update(1)

        # Switch generation config based on token type and save segments
        if new_token == image_start_id:
            config1_sequences.append(current_segment)
            print("switch to generation_config2:", current_segment, tokenizer.decode(current_segment), flush=True)
            current_segment = [new_token]
            current_config = generation_config2
            # current_config = generation_config2.copy()


        elif new_token == image_end_id:
            config2_sequences.append(current_segment)
            print("switch to generation_config1:", current_segment, flush=True)
            current_segment = [new_token]
            # current_config = generation_config1.copy()
            current_config = generation_config1
        elif new_token == eos_token_id:
            # Save final segment
            if current_config == generation_config1:
                config1_sequences.append(current_segment)
            else:
                config2_sequences.append(current_segment)
            break

    pbar.close()

    # Convert to tensor and return
    generated_sequence = torch.tensor(generated_sequence, device=model.device).unsqueeze(0)
    return generated_sequence


def get_model_response_with_type_with_kvcache(
    model,
    tokenizer,
    generation_config1,
    generation_config2,
    input_ids = [],
    negative_prompt=[],
    prefix_allowed_tokens_fn=None,
    logits_processors=[],
):
    """
    Multi-stage generation with different parameters for text and image tokens.
    Uses KV cache for more efficient generation.

    Args:
        model: The language model
        tokenizer: The tokenizer
        generation_config: Base generation config to modify
        input_ids: Input token ids
        negative_prompt: Negative prompt for guidance
        prefix_allowed_tokens_fn: Function to constrain allowed tokens
        logits_processors: List of logits processors
    """
    input_ids = input_ids.to(model.device)
    current_ids = input_ids.clone()

    # Get special token IDs
    image_start_id = tokenizer.encode("<|image start|>")[0]
    image_end_id = tokenizer.encode("<|image end|>")[0]
    eos_token_id = tokenizer.eos_token_id

    # Setup logits processors
    logits_processor = LogitsProcessorList()
    for processor in logits_processors:
        logits_processor.append(processor)
    if prefix_allowed_tokens_fn is not None:
        logits_processor.append(
            PrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn,
                num_beams=generation_config1.num_beams // generation_config1.num_beam_groups,
            )
        )


    if True:
        logits_processor1 = LogitsProcessorList()
        logits_processor1.append(TemperatureLogitsWarper(generation_config1.temperature))
        logits_processor1.append(TopPLogitsWarper(generation_config1.top_p))
        logits_processor1.append(TopKLogitsWarper(generation_config1.top_k))


        logits_processor2 = LogitsProcessorList()
        logits_processor2.append(TemperatureLogitsWarper(generation_config2.temperature))
        logits_processor2.append(TopPLogitsWarper(generation_config2.top_p))
        logits_processor2.append(TopKLogitsWarper(generation_config2.top_k))


    generated_sequence = []
    current_config = generation_config1

    # Track sequences generated with each config
    config1_sequences = []
    config2_sequences = []
    current_segment = []

    # Create progress bar with unknown total
    pbar = tqdm(desc="Generating tokens", total=generation_config1.max_new_tokens)

    # Initialize KV cache
    past_key_values = None
    negative_past_key_values = None

    # Check if CFG is enabled
    if generation_config2.guidance_scale > 1:
        use_cfg = True
    else:
        use_cfg = False

    print(f"use_cfg: {use_cfg}", flush=True)

    now_config_type = "generation_config1"

    for _ in range(generation_config1.max_new_tokens):

        if now_config_type == "generation_config2" and use_cfg:
            # Process both conditional and unconditional paths for CFG
            model_inputs = {
                "input_ids": current_ids if past_key_values is None else current_ids[:, -1:],
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            negative_inputs = {
                "input_ids": negative_ids if negative_past_key_values is None else current_ids[:, -1:],
                "past_key_values": negative_past_key_values,
                "use_cache": True,
            }

            with torch.no_grad():
                outputs = model(**model_inputs)
                negative_outputs = model(**negative_inputs)

            logits = outputs.logits[:, -1, :]
            negative_logits = negative_outputs.logits[:, -1, :]

             # Apply CFG
            logits = negative_logits + (logits - negative_logits) * generation_config2.guidance_scale

            past_key_values = outputs.past_key_values
            negative_past_key_values = negative_outputs.past_key_values

        else:
            # Forward pass with KV cache
            model_inputs = {
                "input_ids": current_ids if past_key_values is None else current_ids[:, -1:],
                "past_key_values": past_key_values,
                "use_cache": True,
            }

            with torch.no_grad():
                outputs = model(**model_inputs)

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values


        # Apply logits processors
        processed_logits = logits_processor(current_ids, logits)

        # Sample next token based on current config
        if current_config.do_sample:
            # probs = torch.nn.functional.softmax(processed_logits / current_config.temp, dim=-1)
            # if current_config.top_k > 0:
            #     probs = top_k_top_p_filtering(probs, top_k=current_config.top_k, top_p=current_config.top_p)
            # # Ensure valid probabilities by clamping and renormalizing
            # probs = torch.clamp(probs, min=0.0)
            # probs = probs / probs.sum(dim=-1, keepdim=True)
            # next_token = torch.multinomial(probs, num_samples=1)

            if now_config_type == "generation_config1":
                current_logits_processor = logits_processor1
            else:
                current_logits_processor = logits_processor2


            next_token_scores = current_logits_processor(current_ids, processed_logits)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # print("next_token", next_token.shape, next_token, flush=True)

        else:
            next_token = torch.argmax(processed_logits, dim=-1).unsqueeze(-1)

        new_token = next_token.item()
        generated_sequence.append(new_token)
        current_segment.append(new_token)

        # Fix dimension mismatch by reshaping next_token to match current_ids
        next_token = next_token.reshape(current_ids.shape[0], 1)
        current_ids = torch.cat([current_ids, next_token], dim=1)

        # Update progress bar
        pbar.update(1)

        # Switch generation config based on token type and save segments
        if new_token == image_start_id:
            config1_sequences.append(current_segment)
            print("switch to generation_config2:", current_segment, tokenizer.decode(current_segment), flush=True)
            current_segment = [new_token]
            current_config = generation_config2
            now_config_type = "generation_config2"

            negative_past_key_values = None
            negative_ids = next_token

        elif new_token == image_end_id:
            config2_sequences.append(current_segment)
            print("switch to generation_config1:", current_segment, flush=True)
            current_segment = [new_token]
            current_config = generation_config1
            now_config_type = "generation_config1"

        elif new_token == eos_token_id or new_token == 151747:
            # Save final segment
            if current_config == generation_config1:
                config1_sequences.append(current_segment)
            else:
                config2_sequences.append(current_segment)
            break

    pbar.close()

    # Convert to tensor and return
    generated_sequence = torch.tensor(generated_sequence, device=model.device).unsqueeze(0)
    return generated_sequence
