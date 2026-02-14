import torch
from easydict import EasyDict
from pathlib import Path

cfg_name = Path(__file__).stem

emu_tokenizer_path = "./tokenizer_emu3_ibq"

save_root = "./gradio_output"

emu_model_path = "./ckpt"

image_size = 262144

num_test_samples = 16

tensor_parallel_size = 8

# max_new_tokens = 20480 #2048 #25600 #18432 #10240 #5120 #16384
max_new_tokens = 16384  # 2048 #25600 #18432 #10240 #5120 #16384

top_k = 131072

top_p = 1.0

temperature = 1.0

guidance_scale = 1.0

task_type = "vla"
classifier_free_guidance = 3.0
dtype = torch.bfloat16

vq_path = "./IBQTokenizer"
vq_type = "ibq"
vq_device = "cuda:0"

ts_token_id = 151848

seed = 0

hf_device = 0

sampling_params = EasyDict(
    use_cache=True,
    # 文字token采样配置
    text_top_k=200,  # 文字token使用较小的topk，提高生成质量
    text_top_p=0.5,  # 文字token使用nucleus sampling
    text_temperature=0.7,  # 文字token温度

    # 图片token采样配置
    image_top_k=5120,  # 图片token使用较大的topk，保持多样性
    image_top_p=1.0,  # 图片token不使用nucleus sampling
    image_temperature=1.0,  # 图片token温度

    # 通用配置
    top_k=131072,  # 默认topk（向后兼容）
    top_p=1.0,  # 默认top_p（向后兼容）
    temperature=1.0,  # 默认温度（向后兼容）
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=max_new_tokens,
    guidance_scale=1.0,

    # 启用差异化采样
    use_differential_sampling=True,
    # use_differential_sampling=False,
)

sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]

special_tokens = EasyDict(
    pad_token="<|endoftext|>",
    eos_token="<|extra_204|>",
    bos_token="<|extra_203|>",
    ts_token="<|extra_202|>",
    image_token="<|image token|>",
    image_start_token="<|image start|>",
    image_end_token="<|image end|>",
)
