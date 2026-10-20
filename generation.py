"""Generation branch: Perceiver's 32 tokens -> trainable fc (768->5120) -> frozen
RadFM LLaMA-13B decoder -> next-token cross-entropy against the organ's GT report.

Builds inputs_embeds directly (32 image-token embeddings concatenated with the target
report's own token embeddings) rather than going through RadFM's full MyEmbedding/
MultiLLaMAForCausalLM - that class's embedding_layer does its own image+text token
fusion for RadFM's native multi-image prompt format, which this project doesn't use
(we supply our own Perceiver's tokens directly, not via RadFM's <image>...</image>
placeholder scheme). Only lang_model's weights are taken from the full checkpoint.
"""
import os

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

LANGUAGE_FILES_DIR = "/home/wisc/dxiang23/Projects/chestCT/code/RadFM/Quick_demo/Language_files"
FULL_CHECKPOINT_PATH = "/mnt/local/data/dxiang23/radfm_weights/pytorch_model.bin"

LLM_HIDDEN_DIM = 5120
IMAGE_TOKEN_COUNT = 32  # must match model.PERCEIVER_NUM_LATENTS


def build_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained(LANGUAGE_FILES_DIR)
    if tokenizer.pad_token_id is None:
        # Language_files/tokenizer_config.json sets bos/eos/unk_token to literal
        # empty strings (not null) - a quirk of this particular release, not a
        # general LLaMA thing. `tokenizer.pad_token = tokenizer.eos_token` inherits
        # that broken empty string and still fails the library's "has a pad token"
        # check, so set pad_token_id directly instead: id 0 is vocab's <unk> (already
        # a real row in lang_model's frozen embedding table, so this reuses an
        # existing id rather than requiring a resize).
        tokenizer.pad_token_id = 0
    return tokenizer


def build_lang_model(device):
    """Frozen LLaMA-13B, architecture from Language_files/config.json, weights from
    the full RadFM checkpoint's lang_model.* keys (RadFM's own further-pretrained
    13B, not off-the-shelf llama-13b-hf - see Language_files/config.json's
    _name_or_path + the Book_mix_2048_13B_full training path referenced in
    finetune.py's ModelArguments default).
    """
    # Language_files/ only has config.json + tokenizer files, no weight file - it's
    # RadFM's Quick_demo way of building the 13B architecture shape before loading the
    # real weights from the separate full checkpoint via load_state_dict (see
    # Quick_demo/test.py). from_pretrained(LANGUAGE_FILES_DIR) looks for an actual
    # weight file there and fails with "no file named pytorch_model.bin..." - build
    # from config only (random init) instead, then overwrite every tensor below.
    #
    # Language_files/config.json declares torch_dtype=float32 - at 13B params that's
    # ~52GB just for weights, more than this environment's single-GPU memory (~46GB)
    # even before the rest of the model/activations. lang_model is frozen and only
    # used for its loss signal, so there's no accuracy reason to keep it in fp32;
    # build and load in bf16 instead (matches how Reg2RG itself trains its own LLM in
    # bf16).
    config = LlamaConfig.from_pretrained(LANGUAGE_FILES_DIR)
    config.torch_dtype = torch.bfloat16
    lang_model = LlamaForCausalLM(config).to(torch.bfloat16)

    full_state_dict = torch.load(FULL_CHECKPOINT_PATH, map_location="cpu")
    lang_model_sd = {
        k[len("lang_model."):]: (v.to(torch.bfloat16) if v.is_floating_point() else v)
        for k, v in full_state_dict.items()
        if k.startswith("lang_model.")
    }
    del full_state_dict  # ~50GB combined dict - drop the rest (embedding_layer.* etc)
                          # once we've pulled out just the lang_model.* tensors we need
    assert lang_model_sd, (
        f"No 'lang_model.*' keys found in {FULL_CHECKPOINT_PATH} - check the "
        "checkpoint's top-level key prefixes."
    )
    missing, unexpected = lang_model.load_state_dict(lang_model_sd, strict=False)
    assert not unexpected, f"unexpected lang_model checkpoint keys: {unexpected[:10]}"
    assert not missing, f"lang_model missing expected keys: {missing[:10]}"

    for param in lang_model.parameters():
        param.requires_grad = False
    lang_model = lang_model.to(device).eval()
    return lang_model


def build_fc():
    """Trainable, bridges the Perceiver's 768-d tokens into the LLM's 5120-d
    embedding space. Discarded after training - Reg2RG never receives this module,
    only perceiver.state_dict()."""
    return nn.Linear(768, LLM_HIDDEN_DIM)


def generation_loss(image_tokens, texts, fc, lang_model, tokenizer, device, max_length=256):
    """image_tokens: (B, 32, 768) - Perceiver's raw (un-pooled) output.
    texts: list[str], the organ's ground-truth report sentences.
    Returns the scalar next-token cross-entropy loss (LlamaForCausalLM's own loss,
    computed over the report-text positions only - image-token positions are masked
    out of the loss via label = -100, matching how Reg2RG/RadFM mask prompt tokens).
    """
    batch_size = image_tokens.size(0)
    image_embeds = fc(image_tokens)  # (B, 32, 5120)

    tokenized = tokenizer(
        texts, padding="max_length", truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(device)
    text_ids = tokenized["input_ids"]  # (B, L)
    text_attention_mask = tokenized["attention_mask"]  # (B, L)

    text_embeds = lang_model.get_input_embeddings()(text_ids)  # (B, L, 5120)

    inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # (B, 32+L, 5120)

    image_attention = torch.ones(
        batch_size, IMAGE_TOKEN_COUNT, dtype=text_attention_mask.dtype, device=device
    )
    attention_mask = torch.cat([image_attention, text_attention_mask], dim=1)

    # Image-token positions never contribute to the loss (-100 = ignored by
    # CrossEntropyLoss); pad positions are already -100 via the tokenizer's pad
    # token id substitution below, matching Reg2RG's own label-masking convention.
    image_labels = torch.full(
        (batch_size, IMAGE_TOKEN_COUNT), -100, dtype=text_ids.dtype, device=device
    )
    text_labels = text_ids.clone()
    text_labels[text_attention_mask == 0] = -100
    labels = torch.cat([image_labels, text_labels], dim=1)

    output = lang_model(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels,
    )
    return output.loss
