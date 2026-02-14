import torch
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from transformers.generation import LogitsProcessor
import math

class SplitUnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        guidance_scale: float,
        model,
        tokenizer,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        allowed_tokens_control = True
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.tokenizer = tokenizer
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }
        self.in_image_start = False
        self.in_image_token = False


        self.first_in_image = True
        self.image_token_count = [0,0]

        self.allowed_tokens_control = allowed_tokens_control

    def get_unconditional_logits(self, input_ids, add_eoi=False):
        if input_ids[0][-1] > 151853:
            self.image_token_count[0] += 1
            # print(self.tokenizer.decode(input_ids[:, -1:][0]), flsuh=True, end=' ')
        elif input_ids[0][-1] == 151846:
            self.image_token_count[1] += 1
        else:
            print(input_ids[:, -1:][0], self.image_token_count, self.tokenizer.decode(input_ids[:, -1:][0]), flsuh=True)
           

        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            if not add_eoi:
                attention_mask = torch.cat(
                    [
                        self.unconditional_context["attention_mask"],
                        torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                    ],
                    dim=1,
                )
                if not self.unconditional_context["use_cache"]:
                    input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
                else:
                    input_ids = input_ids[:, -1:]
                self.unconditional_context["input_ids"] = input_ids
                self.unconditional_context["attention_mask"] = attention_mask

            else:
                attention_mask = torch.cat(
                    [
                        self.unconditional_context["attention_mask"],
                        torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                        torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                    ],
                    dim=1,
                )

                eoi_token = torch.tensor([[151853]]).to(input_ids.dtype).to(input_ids.device)

                if not self.unconditional_context["use_cache"]:
                    input_ids = torch.cat([self.unconditional_context["input_ids"], eoi_token, input_ids[:, -1:]], dim=1)
                else:
                    input_ids = torch.cat([eoi_token, input_ids[:, -1:]], dim=1)

                self.unconditional_context["input_ids"] = input_ids
                self.unconditional_context["attention_mask"] = attention_mask


        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        # print("input_ids", input_ids.shape, flush=True)
        # if input_ids[0][-1] == 151851:

        if input_ids[0][-1] == 151852:
            print("in_image")
            self.in_image_start = True

        if input_ids[0][-1] == 151851:
            print("in_image_token")
            self.in_image_token = True
        
        if input_ids[0][-1] == 151853:
            print("out_image")
            self.in_image_token = False
            self.in_image_start = False

        # 判断是否在<|image start|> 到 <|image end|>
        if self.in_image_start:
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
            if self.guidance_scale == 1:
                return scores

            # 让neg 有完整 
            if self.first_in_image or input_ids[0][-1] != 151852:
                logits = self.get_unconditional_logits(input_ids)
                self.first_in_image = False
            else:
                print("add_eoi")
                logits = self.get_unconditional_logits(input_ids, add_eoi=True)

            # visual token  
            if self.in_image_token:
                unconditional_logits = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
                out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
                # todo 控制 只出visual token
                if self.allowed_tokens_control:
                    mask = torch.full_like(scores, -math.inf)
                    mask[:, 151643:] = 0 # mask掉有意义的字符
                    out = out + mask
                return out
            else:
                return scores
        else:
            # 控制不出visual token
            if self.allowed_tokens_control:
                mask = torch.full_like(scores, -math.inf)
                mask[:, :151854] = 0
                scores = scores + mask
                
            return scores


# class SplitUnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):

#     def __init__(
#         self,
#         guidance_scale: float,
#         model,
#         tokenizer,
#         unconditional_ids: Optional[torch.LongTensor] = None,
#         unconditional_attention_mask: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = True,
#         allowed_tokens_control = True
#     ):
#         self.guidance_scale = guidance_scale
#         self.model = model
#         self.tokenizer = tokenizer
#         self.unconditional_context = {
#             "input_ids": unconditional_ids,
#             "attention_mask": unconditional_attention_mask,
#             "use_cache": use_cache,
#             "past_key_values": None,
#             "first_pass": True,
#         }
#         self.in_image_start = False
#         self.in_image_token = False


#         self.first_in_image = True
#         self.image_token_count = [0,0]

#         self.image_token_count_gen = ""
#         self.image_token_count_gen1 = [0,0]
#         # self.image_token_count_gen2 = [0,0]

#         self.allowed_tokens_control = allowed_tokens_control


#     def get_unconditional_logits(self, input_ids, add_eoi=False):
#         if input_ids[0][-1] > 151853:
#             self.image_token_count[0] += 1
#             # print(self.tokenizer.decode(input_ids[:, -1:][0]), flsuh=True, end=' ')
#         elif input_ids[0][-1] == 151846:
#             self.image_token_count[1] += 1
#         elif input_ids[0][-1] == 151851:
#             assert len(self.image_token_count_gen) == 5
#             row_col = self.image_token_count_gen.split('*')
#             self.image_token_count_gen1[0] = int(row_col[0])
#             self.image_token_count_gen1[1] = int(row_col[1])
#         elif input_ids[0][-1] == 151853:
#             self.image_token_count_gen = ""
#             self.image_token_count_gen1 = [0,0]
#         else:
#             if input_ids[0][-1] != 151852:
#                 self.image_token_count_gen += self.tokenizer.decode(input_ids[:, -1:][0])
#             print(input_ids[:, -1:][0], self.image_token_count, self.tokenizer.decode(input_ids[:, -1:][0]), flsuh=True)
           

#         if self.unconditional_context["first_pass"]:
#             if self.unconditional_context["input_ids"] is None:
#                 self.unconditional_context["input_ids"] = input_ids[:, -1:]
#             if self.unconditional_context["attention_mask"] is None:
#                 self.unconditional_context["attention_mask"] = torch.ones_like(
#                     self.unconditional_context["input_ids"], dtype=torch.long
#                 )
#             input_ids = self.unconditional_context["input_ids"]
#             attention_mask = self.unconditional_context["attention_mask"]
#             self.unconditional_context["first_pass"] = False
#         else:
#             if not add_eoi:
#                 attention_mask = torch.cat(
#                     [
#                         self.unconditional_context["attention_mask"],
#                         torch.ones_like(input_ids[:, -1:], dtype=torch.long),
#                     ],
#                     dim=1,
#                 )
#                 if not self.unconditional_context["use_cache"]:
#                     input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
#                 else:
#                     input_ids = input_ids[:, -1:]
#                 self.unconditional_context["input_ids"] = input_ids
#                 self.unconditional_context["attention_mask"] = attention_mask

#             else:
#                 attention_mask = torch.cat(
#                     [
#                         self.unconditional_context["attention_mask"],
#                         torch.ones_like(input_ids[:, -1:], dtype=torch.long),
#                         torch.ones_like(input_ids[:, -1:], dtype=torch.long),
#                     ],
#                     dim=1,
#                 )

#                 eoi_token = torch.tensor([[151853]]).to(input_ids.dtype).to(input_ids.device)

#                 if not self.unconditional_context["use_cache"]:
#                     input_ids = torch.cat([self.unconditional_context["input_ids"], eoi_token, input_ids[:, -1:]], dim=1)
#                 else:
#                     input_ids = torch.cat([eoi_token, input_ids[:, -1:]], dim=1)

#                 self.unconditional_context["input_ids"] = input_ids
#                 self.unconditional_context["attention_mask"] = attention_mask


#         out = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             use_cache=self.unconditional_context["use_cache"],
#             past_key_values=self.unconditional_context["past_key_values"],
#         )
#         self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

#         return out.logits

#     def __call__(self, input_ids, scores):
#         # print("input_ids", input_ids.shape, flush=True)
#         # if input_ids[0][-1] == 151851:

#         if input_ids[0][-1] == 151852:
#             print("in_image")
#             self.in_image_start = True

#         if input_ids[0][-1] == 151851:
#             print("in_image_token")
#             self.in_image_token = True
        
#         if input_ids[0][-1] == 151853:
#             print("out_image")
#             self.in_image_token = False
#             self.in_image_start = False

#         # 判断是否在<|image start|> 到 <|image end|>
#         if self.in_image_start:
#             scores = torch.nn.functional.log_softmax(scores, dim=-1)
#             if self.guidance_scale == 1:
#                 return scores

#             # 让neg 有完整 
#             if self.first_in_image or input_ids[0][-1] != 151852:
#                 logits = self.get_unconditional_logits(input_ids)
#                 self.first_in_image = False
#             else:
#                 print("add_eoi")
#                 logits = self.get_unconditional_logits(input_ids, add_eoi=True)

#             # visual token  
#             if self.in_image_token:
#                 unconditional_logits = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
#                 out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
#                 # todo 控制 只出visual token
#                 return out
#             else:
#                 return scores
#         else:
#             # 控制不出visual token
#             if self.allowed_tokens_control:
#                 mask = torch.full_like(scores, -math.inf)
#                 mask[:, :151854] = 0
#                 scores = scores + mask
                
#             return scores
