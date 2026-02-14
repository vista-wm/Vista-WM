import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F
from transformers.generation import LogitsProcessor

BOS = 151849
EOS = 151850

IMG = 151851
BOI = 151852
EOI = 151853

BOV = 151854
EOL = 151846
EOF = 151847


# 参考资料：https://blog.csdn.net/fydw_715/article/details/146601416
# 参考资料：https://blog.csdn.net/flyfish1986/article/details/145960485

#   logits_processor 是 _get_logits_processor 方法的一个参数，它是一个可选的 LogitsProcessorList 对象。
#   这个方法会根据 GenerationConfig 中的各种配置参数，创建一系列不同的 LogitsProcessor 实例，并将它们添加到 processors 列表中。
#   最后，如果传入了 logits_processor，还会将其与新创建的处理器列表进行合并。
#
# def _get_logits_processor(
#         self,
#         generation_config: GenerationConfig,
#         input_ids_seq_length: int,
#         encoder_input_ids: torch.LongTensor,
#         prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
#         logits_processor: Optional[LogitsProcessorList],
#         device: str = None,
#         model_kwargs: Optional[Dict[str, Any]] = None,
#         negative_prompt_ids: Optional[torch.Tensor] = None,
#         negative_prompt_attention_mask: Optional[torch.Tensor] = None,
#     ) -> LogitsProcessorList:
#         """
#         此函数返回一个 `LogitsProcessorList` 对象，该对象包含所有用于修改语言模型头部得分的相关 `LogitsProcessor` 实例。
#         这些处理器会对模型预测的 logits 进行调整，以控制文本生成的行为，例如避免重复、控制生成长度等。
#         参数:
#             generation_config (GenerationConfig): 生成配置对象，包含了文本生成过程中的各种配置参数。
#             input_ids_seq_length (int): 输入 ID 序列的长度。
#             encoder_input_ids (torch.LongTensor): 编码器的输入 ID。
#             prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]): 一个可调用对象，用于指定允许的前缀标记。
#             logits_processor (Optional[LogitsProcessorList]): 可选的 logits 处理器列表。
#             device (str, optional): 设备名称，如 'cuda' 或 'cpu'。默认为 None。
#             model_kwargs (Optional[Dict[str, Any]], optional): 模型的其他关键字参数。默认为 None。
#             negative_prompt_ids (Optional[torch.Tensor], optional): 负提示的 ID。默认为 None。
#             negative_prompt_attention_mask (Optional[torch.Tensor], optional): 负提示的注意力掩码。默认为 None。
#         返回:
#             LogitsProcessorList: 包含所有 logits 处理器的列表。
#         """
#
## 由函数可知，processors 先加入各种禁用和停止的 LogitsProcessor, 然后加入自定义的 logits_processor list, 最后是 do_sample 相关的 LogitsProcessor
## 比如 TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, MinPLogitsWarper, TypicalLogitsWarper, EpsilonLogitsWarper, EtaLogitsWarper ...

# hf 中, UnbatchedClassifierFreeGuidanceLogitsProcessor 表示 Unbatched CFG 处理器，适用于【一次只处理一个样本的场景】

# 用于处理 visual tokens 的 Classifier-Free Guidance, CFG 的自定义 Logits Processor，主要用于图像生成任务中结合 textual & visual tokens 的生成过程
# 这个类实现了以下核心功能：
# ​- ​分类器自由引导 (CFG)​​：通过有条件和无条件生成的 logits 差异来引导生成过程
# ​​- 视觉标记处理​​：专门处理图像生成相关的标记（BOI, IMG, EOI 等）
# ​​- 文本-视觉模式切换​​：根据当前生成的标记类型切换处理逻辑
# ​​- 图像尺寸控制​​：解析并强制保持图像的高度和宽度

# transformers 每次都会调用 forward, 同时返回算上了当前预测结果构建的 past_key_values, 可以用作下一步预测的 kvcache

class UnbatchedClassifierFreeGuidanceLogitsForVisualTokenProcessor(LogitsProcessor):
    '''标准 Logit Processor'''
    def __init__(
        self,
        guidance_scale: float,
        model,
        tokenizer,
        unconditional_ids: Optional[torch.LongTensor],
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        allowed_tokens_control: bool = True,
        force_same_image_size: bool = True,
        unconditional_type: str = "no_text",  # no_text, no_prev_text, no_prev_modal, etc.  # 默认只有 "no_text", 表示预测的和生成的 text token 都不加入 uncond seq
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.tokenizer = tokenizer

        self.unconditional_context = {
            "input_ids": unconditional_ids,  # uncond token ids
            "attention_mask": (
                unconditional_attention_mask 
                if unconditional_attention_mask is not None 
                else torch.ones_like(unconditional_ids, dtype=torch.long)
            ),
            "default_input_ids": unconditional_ids,
            "default_attention_mask": (
                unconditional_attention_mask 
                if unconditional_attention_mask is not None 
                else torch.ones_like(unconditional_ids, dtype=torch.long)
            ),
            "first_pass": True,
            "past_key_values": None,
        }
        self.use_cache = use_cache

        self.first_in_image = True                          # 是否是第一次生成图像
        self.in_image = False                               # 是否在生成图像 pattern & tokens
        self.in_visual = False                              # 是否在生成视觉 tokens

        self.allowed_tokens_control = allowed_tokens_control

        self.height = None
        self.width = None                                   # 图像宽高
        self.hw_tokens = None                               # 存储生成的第一张图的图像宽高 tokens
        self.force_same_image_size = force_same_image_size  # 强制生成相同的 image size
        self.unconditional_type = unconditional_type        # 各种设置 uncond 的策略, 默认 no_text

        self.text_segment = 0

    ## 只有预测 visual tokens 才会进 get_unconditional_logits() 函数, 并且每预测一个 visual token 就通过 self.update_unconditional_context 把 visual token 塞入 uncond seq
    ## only used for visual tokens with cfg
    def get_unconditional_logits(self, input_ids):
        # update cache
        # 把 input_ids 的最后一个 token ids 加到 uncond token ids 并扩展一个 uncond attn mask 长度
        input_ids, attention_mask = self.update_unconditional_context(input_ids)  
        if self.use_cache and not self.unconditional_context["first_pass"]:
            input_ids = input_ids[:, -1:]
        else:
            self.unconditional_context["first_pass"] = False

        # print(f"unc inputs: {self.tokenizer.decode(input_ids[0])}", flush=True)

        ## uncond 序列单独预测 uncond token
        out = self.model(
            input_ids,  # uncond token ids
            attention_mask=attention_mask,  # uncond token attn mask
            use_cache=self.use_cache,  # 是否使用 uncond 的 kvcache
            past_key_values=self.unconditional_context["past_key_values"],  # uncond 的 kvcache
        )

        # 记录当前所有已知 uncond tokens 的 kv cache, 可用于下一次 uncond token 生成
        if self.use_cache:
            self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits[:, -1]  # 返回最后一个, 即最新预测的 last uncond token logit

    ### 主入口方法，根据当前 token 决定使用 text 还是 image 处理逻辑
    # input_ids 表示当前已生成的 seq 的 token index
    # scores 即 last_token_logits, shape = (bs, vocab_size) = (1, 282926) 表示模型在当前时间步对所有 vocab 中每个 token 的预测得分（通常是 logits）
    # (hidden states @ nn.Embedding -> logits -> softmax -> prob)
    def __call__(self, input_ids, scores):
        # IMAGE MODE
        if input_ids[0][-1] == BOI:                               # 当前已预测的最新 token 是 boi - 那么要开始生成 image 相关 tokens - image mode
            self.set_unconditional_context(input_ids)

        # TEXT MODE
        if input_ids[0][-1] == EOI:                               # 当前已预测的最新 token 是 eoi - 那么要开始生成 text 相关 tokens - text mode
            self.exit_image(input_ids)
        
        if input_ids[0][-1] != BOI and input_ids[0][-2] == EOI:   # 当前已预测的 次新 token 不是 eoi, 最新 token 不是 boi, 意味着当前预测的 token 不属于上一张图, 是 text token
            self.text_segment += 1
        
        N = input_ids.shape[1]
        if self.in_image:
            # all image tokens between boi and eoi will be processed in in_image_logits_processor
            scores = self.in_image_logits_processor(input_ids, scores)
        else:
            # all text tokens will be processed in in_text_logits_processor
            scores = self.in_text_logits_processor(input_ids, scores)
        return scores                                             # 返回调整后的 last_token_logits
    
    # all image tokens between boi and eoi will be processed in in_image_logits_processor
    def in_image_logits_processor(self, input_ids, scores):
        # all visual related logits should call get_unconditional_logits to update unconditional cache
        unc_scores = self.get_unconditional_logits(input_ids)     # 返回最后一个 即最新预测的 last uncond token logit

        # 正在生成视觉 tokens
        if self.in_visual:  
            # generating visual tokens
            img_idx = self.find_last_token_index(input_ids[0], IMG)  # 找到最后一个 IMG token 的 idx【IMG 后才是各种 vis vis vis eol vis vis vis ... tokens】
            vis_idx = input_ids.shape[1] - img_idx                   # 表示预测到第几个视觉 (vis/eol/eoi) token
            # 当前最新预测的是 eoi token
            if vis_idx == self.height * (self.width + 1):
                print(f"in visual and generate eoi", flush=True)
                # 预测每一个 end of image token 时, 同下, 非 eoi 的概率置为 -inf (当然，直接把 eoi 塞进 input_ids 也行)
                mask = torch.full_like(scores, -math.inf)
                mask[:, EOI] = 0
                scores = scores + mask
                return scores
            # 当前最新预测的是 eol token
            elif vis_idx % (self.width + 1) == 0:
                # 预测每一个 end of image token 时, 同下, 非 eol 的概率置为 -inf (当然，直接把 eol 塞进 input_ids 也行)
                mask = torch.full_like(scores, -math.inf)
                mask[:, EOL] = 0
                scores = scores + mask
                return scores
            # 当前最新预测的是 visual token
            else:
                # visual tokens from second one to the last one - 这时候生成 visual tokens 部分就要开 cfg 了
                scores = self.apply_cfg(scores, unc_scores, self.guidance_scale)  # 返回 log_prob / scores_log_softmax

                # 预测每一个 visual token 时
                # 把 bos 之前的所有 text vocab 的位置设为 -inf 表示采样概率无限小, 把 bos 及其后面的所有 image vocab 的位置设为 0 表示采样概率不变
                # 确保预测 image token 时 image token id 的高采样概率 —— 不会有任何 text token id 的采样概率太高以至于被采样到
                mask = torch.full_like(scores, -math.inf)
                mask[:, BOV:] = 0
                scores = scores + mask
                return scores
            
        # self.in_visual = False, 并未正在生成视觉 tokens, 而是刚刚开始预测到 IMG token, 后面才是各种 vis/eol/eoi 
        else:
            # which means the h and w is generated, need to confirm h and w
            if input_ids[0][-1] == IMG:  
                self.in_visual = True  # 表明接下来要开始预测正式的 h * w 个 image token 了 (各种 vis/eol/eoi)
                if self.first_in_image or not self.force_same_image_size:  # 如果是第一次生成图像 且不要求 force_same_image_size
                    self.parse_hw(input_ids)  # 解析出最后一张图像中的 h 和 w 并记录到 self.height 和 self.width
                    self.first_in_image = False  # 以后再进来 就不再是第一次生成图像了
                print(f"generate image with h: {self.height} and w: {self.width}", flush=True)

                # the first visual token - 这时候生成 visual tokens 部分就要开 cfg 了
                scores = self.apply_cfg(scores, unc_scores, self.guidance_scale)  # 返回 log_prob / scores_log_softmax

                # 把 bos 之前的所有 text vocab 的位置设为 -inf 表示采样概率无限小, 把 bos 及其后面的所有 image vocab 的位置设为 0 表示采样概率不变
                # 确保预测 image token 时 image token id 的高采样概率 —— 不会有任何 text token id 的采样概率太高以至于被采样到
                # mask non visual tokens
                mask = torch.full_like(scores, -math.inf)
                mask[:, BOV:] = 0
                scores = scores + mask  # 加性 mask
                return scores
            
            # do not reach IMG token, which means the model is generating h x w parts, so directly return
            else:
                if self.first_in_image or not self.force_same_image_size:
                    print(f"generate h and w", flush=True)
                    return scores
                # not self.first_in_image or self.force_same_image_size
                else:
                    # force same image size, which means must generate previous h and w
                    boi_idx = self.find_last_token_index(input_ids[0], BOI)  # 找到最后一个 boi
                    hw_idx = input_ids.shape[1] - boi_idx  # 最后一个 boi 后面的 h, w 的 idx - 确保后面生成的图像和前一张的 h 与 w 一样 ?
                    mask = torch.full_like(scores, -math.inf)
                    if hw_idx > len(self.hw_tokens):  # self.hw_tokens 即 {token_height}*{token_width} ids
                        mask[:, IMG] = 0
                        print(f"generate h and w, mask {hw_idx-1} IMG", flush=True)
                    else:
                        mask[:, self.hw_tokens[hw_idx - 1]] = 0
                        print(f"generate h and w, mask {hw_idx-1} {self.hw_tokens[hw_idx-1]}", flush=True)
                    scores = scores + mask
                    return scores


    # all text tokens will be processed in in_text_logits_processor
    def in_text_logits_processor(self, input_ids, scores):
        # for text tokens, do not apply cfg
        # control dont pick visual tokens
        # 没进, 默认 no_text
        if self.unconditional_type == "uncondition_tokens":  
            self.get_unconditional_logits(torch.ones_like(input_ids) * (151756 + self.text_segment))  # 返回最后一个 即最新预测的 last uncond token logit

        # print(f"in_text_logits_processor", flush=True)

        # 把 bos 之前的所有 text vocab 的位置设为 0 表示采样概率不变, 把 bos 及其后面的所有 image vocab 的位置设为 -inf 表示采样概率无限小 
        # 确保预测 text token 时 text token id 的高采样概率 —— 不会有任何 image token id 的采样概率太高以至于被采样到
        mask = torch.full_like(scores, -math.inf)
        mask[:, :BOV] = 0  
        scores = scores + mask  # 加性 sampling rate mask
        return scores


    # 解析出最后一张图像中的 h 和 w 并记录到 self.height 和 self.width
    def parse_hw(self, input_ids):
        # image_token: "<|visual token {token_id:0>6d}|>"
        # image_string: "{image_start}{token_height}*{token_width}{image_token}{token_str}{image_end}"
        if self.height is not None and self.width is not None and self.force_same_image_size:
            return

        # Find indices of last BOI and IMG tokens in the sequence
        seq = input_ids[0]
        img_indices = (seq == IMG).nonzero().flatten()  # 所有 {image_token}
        boi_indices = (seq == BOI).nonzero().flatten()  # {image_start}

        # Get the last occurrence of each token
        last_img_pos = img_indices[-1].item()  # 最后一个 {image_token}
        last_boi_pos = boi_indices[-1].item()  # 最后一个 {image_start}

        # Get tokens between last BOI and IMG
        self.hw_tokens = seq[last_boi_pos+1:last_img_pos]  # 取出最后一个 {token_height}*{token_width} ids

        # Decode to string and parse dimensions
        hw_str = self.tokenizer.decode(self.hw_tokens)  # 解码成 {token_height}*{token_width}
        print(f"hw_str: {hw_str}, hw_tokens: {self.hw_tokens}", flush=True)
        h, w = hw_str.split('*')  # 拆分成 token_height, token_width 后记录到 self 属性
        self.height = int(h)
        self.width = int(w)


    def find_last_token_index(self, seq, token_id):
        # seq: N
        token_indices = (seq == token_id).nonzero().flatten()
        if len(token_indices) == 0:
            return -1
        return token_indices[-1].item()

    def find_first_token_index(self, seq, token_id):
        token_indices = (seq == token_id).nonzero().flatten()
        if len(token_indices) == 0:
            return -1
        return token_indices[0].item()
    
    
    # 为什么用 F.log_softmax 而不是 F.softmax
    # LlaMAGen 用的 F.softmax, transformers 库用的 F.log_softmax
    # ​​推荐使用 F.log_softmax​​，因为：log(softmax(x)) = log_softmax(x)，可以直接用 log_softmax 计算，避免数值不稳定
    # - 数值稳定性更好
    # - 计算效率更高
    # - 数学上等价，但避免额外计算
    # ​​某些代码用 F.softmax​​ 可能是：
    # - 历史代码遗留
    # - 为了更直观地表示概率调整
    # - 后续需要概率值而非对数概率

    # 生成 visual tokens 开 cfg
    def apply_cfg(self, scores, unc_scores, cfg_scale):
        # logits -> softmax -> probs
        scores = F.log_softmax(scores, dim=-1)  # log_prob / scores_log_softmax of cond
        unc_scores = F.log_softmax(unc_scores, dim=-1)  # log_prob / scores_log_softmax of uncond
        scores = cfg_scale * (scores - unc_scores) + unc_scores  # output = uncond + cfg * (cond - uncond)
        return scores  # log_prob / scores_log_softmax
    
    # 从 if input_ids[0][-1] == BOI: 进来的, 用于准备好当前 token 的 uncond logit
    def set_unconditional_context(self, input_ids):
        self.in_image = True  # 正在生成图像 pattern & tokens

        # exclude frontmost "BOS <|extra_100|>" and the newest "BOI"
        input_ids = input_ids[:, 2:-1]

        # 进, 默认 no_text
        if self.unconditional_type == "no_text" or self.unconditional_type == "uncondition_tokens":
            # do nothing special
            return
        
        # 没进, 默认 no_text
        elif self.unconditional_type == "no_prev_text":
            # if the previous token is eoi, which means no new generated text, so do nothing
            if input_ids[0][-1] == EOI:
                return
            
            last_eoi_idx = self.find_last_token_index(input_ids[0], EOI)
            # if there is no eoi, it means this is the first time to generate image
            # so we can use the default unconditional context
            if last_eoi_idx == -1:
                return
            
            # otherwise, select from start to the last image, exclude newly generated text
            self.unconditional_context["input_ids"] = torch.cat(
                [
                    self.unconditional_context["default_input_ids"],
                    input_ids[:, :last_eoi_idx+1],
                ],
                dim=1,
            )
            self.unconditional_context["attention_mask"] = torch.cat(
                [
                    self.unconditional_context["default_attention_mask"],
                    torch.ones_like(input_ids[:, :last_eoi_idx+1], dtype=torch.long),
                ],
                dim=1,
            )
            self.unconditional_context["past_key_values"] = None
            self.unconditional_context["first_pass"] = True
            print(f"reset unconditional context to {self.unconditional_context['input_ids']}", flush=True)

        # 没进, 默认 no_text
        elif self.unconditional_type == "no_prev_modal":
            if input_ids[0][-1] == EOI:
                # exclude last image
                last_boi_idx = self.find_last_token_index(input_ids[0], BOI)
                self.unconditional_context["input_ids"] = torch.cat(
                    [
                        self.unconditional_context["default_input_ids"],
                        input_ids[:, :last_boi_idx],
                    ],
                    dim=1,
                )
                self.unconditional_context["attention_mask"] = torch.cat(
                    [
                        self.unconditional_context["default_attention_mask"],
                        torch.ones_like(input_ids[:, :last_boi_idx], dtype=torch.long),
                    ],
                    dim=1,
                )
                self.unconditional_context["past_key_values"] = None
                self.unconditional_context["first_pass"] = True
                print(f"reset unconditional context to {self.unconditional_context['input_ids']}", flush=True)
            else:
                # exclude last text segment
                last_eoi_idx = self.find_last_token_index(input_ids[0], EOI)
                # no image generated, just discard all previous tokens and use the default unconditional context
                if last_eoi_idx == -1:
                    return

                self.unconditional_context["input_ids"] = torch.cat(
                    [
                        self.unconditional_context["default_input_ids"],
                        input_ids[:, :last_eoi_idx + 1],
                    ],
                    dim=1,
                )
                self.unconditional_context["attention_mask"] = torch.cat(
                    [
                        self.unconditional_context["default_attention_mask"],
                        torch.ones_like(input_ids[:, :last_eoi_idx + 1], dtype=torch.long),
                    ],
                    dim=1,
                )
                self.unconditional_context["past_key_values"] = None
                self.unconditional_context["first_pass"] = True
                print(f"reset unconditional context to {self.unconditional_context['input_ids']}", flush=True)

        # 没进, 默认 no_text
        elif self.unconditional_type == "no_text_random_drop_vis":
            # not generating a new image, use previous image
            if input_ids[0][-1] != EOI:
                return
            
            boi_idx = (seq == BOI).nonzero().flatten()
            eoi_idx = (seq == EOI).nonzero().flatten()
            if len(boi_idx) == 0 or len(eoi_idx) == 0:
                return
            
            boi_idx = boi_idx[-1]
            eoi_idx = eoi_idx[0]

        # 没进, 默认 no_text
        elif self.unconditional_type == "unc_single_token":
            if input_ids[0][-1] == EOI:
                return
            
            self.get_unconditional_logits(torch.ones_like(input_ids) * 151756)  # 返回最后一个 即最新预测的 last uncond token logit

        else:
            raise ValueError(f"Unconditional type {self.unconditional_type} not supported")
    
    # 把 input_ids 的最后一个 token ids 加到 uncond token ids 并扩展一个 uncond attn mask 长度
    def update_unconditional_context(self, input_ids):
        self.unconditional_context["input_ids"] = torch.cat(
            [
                self.unconditional_context["input_ids"],  # uncond token ids
                input_ids[:, -1:],  # last input ids
            ],
            dim=1,
        )
        self.unconditional_context["attention_mask"] = torch.cat(
            [
                self.unconditional_context["attention_mask"],
                torch.ones_like(input_ids[:, -1:], dtype=torch.long),
            ],
            dim=1,
        )
        return self.unconditional_context["input_ids"], self.unconditional_context["attention_mask"]

    def exit_image(self, input_ids):
        self.in_image = False   # 没在生成图像 pattern & tokens
        self.in_visual = False  # 没在生成视觉 tokens
        # update eoi to unconditional cache
        self.get_unconditional_logits(input_ids)  # 返回最后一个 即最新预测的 last uncond token logit - 但这里本质是更新 last token
        print(f"exit image", flush=True)


class UnbatchedClassifierFreeGuidanceLogitsForVisualTokenWithDifferentialTopKProcessor(UnbatchedClassifierFreeGuidanceLogitsForVisualTokenProcessor):
    """
    扩展原有的 CFG 标准 Logit Processor, 添加差异化 TopK 功能
    在生成文字token和图片token时使用不同的topk、top_p、temperature参数
    """
    def __init__(
        self,
        guidance_scale: float,
        model,
        tokenizer,
        unconditional_ids: Optional[torch.LongTensor],
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        allowed_tokens_control: bool = True,
        force_same_image_size: bool = True,
        unconditional_type: str = "no_text",  # 默认 no_text
        # 新增的差异化 topk 参数
        use_differential_sampling: bool = False,
        text_top_k: int = 10240,
        image_top_k: int = 65536,
        text_top_p: float = 0.9,
        image_top_p: float = 1.0,
        text_temperature: float = 1.0,
        image_temperature: float = 1.0,
    ):
        # 调用父类初始化
        super().__init__(
            guidance_scale=guidance_scale,
            model=model,
            tokenizer=tokenizer,
            unconditional_ids=unconditional_ids,
            unconditional_attention_mask=unconditional_attention_mask,
            use_cache=use_cache,
            allowed_tokens_control=allowed_tokens_control,
            force_same_image_size=force_same_image_size,
            unconditional_type=unconditional_type,  # 默认 no_text
        )
        
        # 差异化采样参数
        self.use_differential_sampling = use_differential_sampling
        self.text_top_k = text_top_k
        self.image_top_k = image_top_k
        self.text_top_p = text_top_p
        self.image_top_p = image_top_p
        self.text_temperature = text_temperature
        self.image_temperature = image_temperature
        
        if use_differential_sampling:
            print(f"DifferentialTopK enabled:")
            print(f"  Text: top_k={text_top_k}, top_p={text_top_p}, temp={text_temperature}")
            print(f"  Image: top_k={image_top_k}, top_p={image_top_p}, temp={image_temperature}")
    
    def apply_differential_topk(self, scores, is_image_generation=False):
        """
        根据当前生成模式应用差异化的 topk、top_p、temperature 参数
        
        Args:
            scores: 原始 logits of the last token
            is_image_generation: 是否在生成图片 token
            
        Returns:
            处理后的scores
        """
        if not self.use_differential_sampling:
            return scores
            
        # 根据当前状态选择参数
        if is_image_generation:
            current_top_k = self.image_top_k
            current_top_p = self.image_top_p
            current_temperature = self.image_temperature
            mode = "IMAGE"
        else:
            current_top_k = self.text_top_k
            current_top_p = self.text_top_p
            current_temperature = self.text_temperature
            mode = "TEXT"

        # 应用温度缩放
        if current_temperature != 1.0:
            scores = scores / current_temperature
        
        # 应用TopK过滤
        if current_top_k > 0 and current_top_k < scores.size(-1):
            top_k = min(current_top_k, scores.size(-1))
            # 获取topk的索引
            indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
            scores[indices_to_remove] = float('-inf')
        
        # 应用TopP过滤
        if current_top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 找到累积概率超过top_p的位置, 其余位置则为要被 remove 的内容
            sorted_indices_to_remove = cumulative_probs > current_top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  # 整体非循环右移一位
            sorted_indices_to_remove[..., 0] = 0  # 把 idx = 0 的位置赛一个 0 进去
            ## 为什么要往开头塞进去一个 0? 为了确保我们保留的是累积概率刚好超过 top-p 阈值之前的那些 tokens
            #   原始数组可能是这样的：[F, F, T, T, T]（F=False, T=True）
            #   执行这行后变成：[F, F, F, T, T]
            #   这样做的目的是确保我们保留的是第一个使累积概率超过 top-p 的 token 及其之前的所有 token
            #   同时确保第一个 token 永远不会被移除
            
            # 将过滤后的结果映射回原始索引
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores[indices_to_remove] = float('-inf')
        
        return scores
    
        ## 以上过程推演

        # >>> import torch
        # >>> scores = torch.tensor([[0.2, 0.5, 0.4, -0.3, -0.6, 0.1, -0.1, 0.3]])  # shape = (bs, vocab_size)
        # >>> scores.shape
        # torch.Size([1, 8])

        # # temp
        # >>> current_temperature = 0.25
        # >>> scores = scores / current_temperature
        # >>> scores
        # tensor([[ 0.8000,  2.0000,  1.6000, -1.2000, -2.4000,  0.4000, -0.4000,  1.2000]])

        # # topk
        # >>> current_top_k = 4
        # >>> top_k = min(current_top_k, scores.size(-1))
        # >>> top_k
        # 4 
        # >>> torch.topk(scores, top_k)
        # torch.return_types.topk(
        #     values=tensor([[2.0000, 1.6000, 1.2000, 0.8000]]),
        #     indices=tensor([[1, 2, 7, 0]]))
        # )
        # >>> torch.topk(scores, top_k)[0]
        # tensor([[2.0000, 1.6000, 1.2000, 0.8000]])
        # >>> torch.topk(scores, top_k)[0][..., -1, None]
        # tensor([[0.8000]])
        # >>> indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        # >>> indices_to_remove
        # tensor([[False, False, False,  True,  True,  True,  True, False]])
        # >>> scores[indices_to_remove] = float('-inf')  # 把 topk 之外的 vocab id logit score 都置为 -inf
        # >>> scores
        # tensor([[0.8000, 2.0000, 1.6000,   -inf,   -inf,   -inf,   -inf, 1.2000]])

        # # topp
        # >>> torch.sort(scores, descending=True)
        # torch.return_types.sort(
        #     values=tensor([[2.0000, 1.6000, 1.2000, 0.8000,   -inf,   -inf,   -inf,   -inf]]),
        #     indices=tensor([[1, 2, 7, 0, 3, 4, 5, 6]]))
        # )
        # >>> sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        # >>> torch.softmax(sorted_logits, dim=-1)  # logits -> probs
        # tensor([[0.4131, 0.2769, 0.1856, 0.1244, 0.0000, 0.0000, 0.0000, 0.0000]])
        # >>> cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)  # 累加 probs 和
        # >>> cumulative_probs
        # tensor([[0.4131, 0.6900, 0.8756, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])
        # >>> current_top_p = 0.7
        # >>> sorted_indices_to_remove = cumulative_probs > current_top_p
        # >>> sorted_indices_to_remove
        # tensor([[False, False,  True,  True,  True,  True,  True,  True]])
        # >>> sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        # >>> sorted_indices_to_remove
        # tensor([[False, False, False,  True,  True,  True,  True,  True]])
        # >>> sorted_indices_to_remove[..., 0] = 0
        # >>> sorted_indices_to_remove
        # tensor([[False, False, False,  True,  True,  True,  True,  True]])
        # >>> indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        # >>> indices_to_remove
        # tensor([[ True, False, False,  True,  True,  True,  True, False]])
        # >>> scores[indices_to_remove] = float('-inf')
        # >>> scores
        # tensor([[  -inf, 2.0000, 1.6000,   -inf,   -inf,   -inf,   -inf, 1.2000]])

    def in_image_logits_processor(self, input_ids, scores):
        """
        重写父类的图片logits处理器，添加差异化topk支持
        """
        # 调用父类方法获取基础处理结果
        scores = super().in_image_logits_processor(input_ids, scores)
        
        # 应用差异化 topk（图片token）
        scores = self.apply_differential_topk(scores, is_image_generation=True)
        
        return scores
    
    def in_text_logits_processor(self, input_ids, scores):
        """
        重写父类的文字logits处理器，添加差异化topk支持
        """
        # 调用父类方法获取基础处理结果
        scores = super().in_text_logits_processor(input_ids, scores)
        
        # 应用差异化 topk（文字token）
        scores = self.apply_differential_topk(scores, is_image_generation=False)
        
        return scores
