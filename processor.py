"""
Processor class for DiffVLM
"""

import numpy as np
import torch

from PIL import Image
from typing import List, Optional, Union

IMAGE_TOKEN = "<image>"


def build_string_from_input(prompt, bos_token, image_seq_len, image_token, num_images):
    """
    Builds a string from the input prompt and image tokens.
    For example, for the call:
    build_string_from_input(
        prompt="Prefix str"
        bos_token="<s>",
        image_seq_len=3,
        image_token="<im>",
    )
    The output will be:
    "<im><im><im><s>Initial str"
    Args:
        prompt (`list[Union[str, ImageInput]]`): The input prompt.
        bos_token (`str`): The beginning of sentence token.
        image_seq_len (`int`): The length of the image sequence.
        image_token (`str`): The image token.
        num_images (`int`): Number of images in the prompt.
    """
    return f"{image_token * image_seq_len * num_images}{bos_token}{prompt}\n"


class Processor:
    def __init__(self,
                 image_processor=None,
                 tokenizer=None,
                 image_seq_length: int = 256,
    ):
        super().__init__()
        self.image_seq_length = image_seq_length
        tokens_to_add = {"additional_special_tokens": [IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.image_token = IMAGE_TOKEN
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self,
                 main_images: Union[Image.Image, List[Image.Image]],
                 ref_images: Union[Image.Image, List[Image.Image]],
                 text: Union[str, List[str]],
                 suffix: Optional[Union[str, List[str]]] = None,
        ):
        """
        Method to prepare for the model one or more sequences and images.

        Example
        ```python
        main_images = Image
        ref_images = Image
        prompt = "answer en what has changed compared to the reference image?"
        suffix = "left lung field"
        input = processor(text=prompt, main_image=main_image, ref_images=ref_images, suffix=suffix)
        ```
        Here `inputs` will contain the `input_ids` and `token_type_ids` that follow
        ```python
        inputs["input_ids"][:, 197:]
        # tensor([[2, 6006, 603, 578, 13910]])
        inputs["token_type_ids"][:, 197:]
        tensor([[0, 0, 0, 0, 0, 0, 1, 1]])
        ```
        Meaning the last 3 tokens are label

        Args:
            main_images (`PIL.Image.Image`, `np.ndarray`, `list[Image.Image]`):
                The main image or batch of images to be prepared. Each image can be a PIL image or a numpy array
            ref_images (`PIL.Image.Image`, `np.ndarray`, `list[Image.Image]`):
                The reference image or batch of images to be prepared.Each image can be a PIL image, numpy array
            text (`str`, List[str]):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or list of strings
            return_tensors (`str`):
                If set, will return tensors of a particular framework 
                    - `'pt'`: Return Pytorch `torch.Tensor` objects
                    - `'np'`: Return Numpy `np.ndarray` objects
            suffix (`str`, `list[str]`):
                The suffixes or batch of suffixxes to be encoded.

        Returns:
            - **input_ids** -- List of token ids to be fed to a model. Returned when text is not None. If suffix is provided,
            the `input_ids` will also contain the suffix
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
            `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_name` and if `text` is not None).
            - **pixel_values_main** -- Pixel values of the main image to be fed to a model.
            - **pixel_values_ref** -- Pixel values of the ref image to be fed to the model.
            - **labels** -- Labels compatible with training if `suffix` is not None
        """
        if isinstance(main_images, Image.Image):
            main_images = [main_images]
        if isinstance(ref_images, Image.Image):
            ref_images = [ref_images]
        if isinstance(text, str):
            text = [text]
        return_token_type_ids = True
        return_mm_token_type_ids = True
        if not any(IMAGE_TOKEN in sample for sample in text):

            if isinstance(text, list) and isinstance(ref_images, list) and isinstance(main_images, list):
                if len(main_images) != len(text) and len(ref_images) != len(main_images):
                    raise ValueError(
                        f"Received {len(main_images)} main images for {len(ref_images)} ref images and {len(text)} prompts"
                    )
            
            input_strings = [
                build_string_from_input(
                    prompt=prompt,
                    bos_token=self.tokenizer.bos_token,
                    image_seq_len=self.image_seq_length,
                    image_token=IMAGE_TOKEN,
                    num_images=(len(main_image_list) if isinstance(main_image_list, list) else 1) + (len(ref_image_list) if isinstance(ref_image_list, list) else 1)     
                )
                for prompt, main_image_list, ref_image_list in zip(text, main_images, ref_images)
            ]
        else:
            expanded_samples = []
            for sample in text:
                expanded_sample = sample.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length)
                bos_rfind_index = expanded_sample.rfind(IMAGE_TOKEN)
                bos_index = bos_rfind_index + len(IMAGE_TOKEN) if bos_rfind_index != -1 else 0
                expanded_sample = (
                    expanded_sample[:bos_index] + self.tokenizer.bos_token + expanded_sample[bos_index:]
                )
                expanded_samples.append(expanded_sample)
            input_strings = [f"{sample}\n" for sample in expanded_samples]

        if suffix is not None and not isinstance(suffix, list):
            suffix = [suffix]
        if suffix is not None:
            suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]

        pixel_values_main = self.image_processor(main_images, return_tensors="pt")["pixel_values"]
        pixel_values_ref = self.image_processor(ref_images, return_tensors="pt")["pixel_values"]

        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            return_tensors="pt",
            padding=True,
        )
        # return_data = {**inputs, "pixel_values_main": pixel_values_main, "pixel_values_ref": pixel_values_ref}
        input_ids = inputs["input_ids"][:, :-1]
        attention_mask = inputs["attention_mask"][:, :-1]
        return_data = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_main=pixel_values_main,
            pixel_values_ref=pixel_values_ref,
        )
        if return_token_type_ids:
            labels = inputs["input_ids"].clone()
            labels[inputs["token_type_ids"] == 0] = -100
            labels = labels[:, 1:]
            return_data.update({"labels": labels})

        if return_mm_token_type_ids:
            array_ids = return_data["input_ids"]
            mm_token_type_ids = torch.zeros_like(return_data["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            return_data["mm_token_type_ids"] = mm_token_type_ids

        return return_data