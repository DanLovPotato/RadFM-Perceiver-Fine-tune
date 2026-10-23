"""Generation branch: Perceiver's 32 tokens -> fc (768->4096, lives in model.py,
generation-only) -> frozen LLaMA-2-7B-chat decoder -> next-token cross-entropy
against the organ's GT report.

Uses LLaMA-2-7B-chat specifically (not RadFM's own further-pretrained native 13B,
which an earlier revision of this module used) because that's what Reg2RG's own
archive actually trains against (see Reg2RG-debug-current-full_correct_F_20260920_
archive/configs/train_radgenome/jhcpu7.sh's lang_encoder_path and Reg2RG.py's
self.hidden_dim = 4096) - training fc/perceiver against RadFM's 13B would shape them
for a decoder Reg2RG never actually uses, which doesn't transfer to the LLM Reg2RG's
own forward pass reads fc's output into. Matching the real decoder means fc's weights
become directly loadable into Reg2RG's own self.fc, not just perceiver's into
self.perceiver.

Builds inputs_embeds directly (32 image-token embeddings concatenated with the target
report's own token embeddings) rather than going through Reg2RG's own MyEmbedding's
full image+text token fusion (its one-hot/extended-vocabulary scheme, built for
interleaving multiple images anywhere in a prompt) - this project only ever has one
image prefixed to one report, so a plain concatenation is equivalent and simpler.
"""
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

LLAMA2_CHAT_PATH = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/weights/llama_weights"

LLM_HIDDEN_DIM = 4096
IMAGE_TOKEN_COUNT = 32  # must match model.PERCEIVER_NUM_LATENTS


def build_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA2_CHAT_PATH)
    if tokenizer.pad_token_id is None:
        # This checkpoint's special_tokens_map.json already declares a real pad_token
        # ("<unk>"), unlike RadFM's Quick_demo/Language_files (which had the broken
        # literal-empty-string issue this fallback was originally written for) - kept
        # as a defensive fallback in case that's ever not true, not because it's
        # expected to trigger here.
        tokenizer.pad_token_id = 0
    return tokenizer


def build_lang_model(device):
    """Frozen LLaMA-2-7B-chat, standard HuggingFace safetensors checkpoint - unlike
    the old RadFM-13B path, this needs no manual config-then-filtered-state-dict
    dance, from_pretrained() handles it directly."""
    lang_model = LlamaForCausalLM.from_pretrained(LLAMA2_CHAT_PATH, torch_dtype=torch.bfloat16)
    for param in lang_model.parameters():
        param.requires_grad = False
    lang_model = lang_model.to(device).eval()
    return lang_model


def generation_loss(image_embeds, texts, lang_model, tokenizer, device, max_length=256):
    """image_embeds: (B, 32, 4096) - PerceiverAlignmentModel.encode_image's fc-projected,
    un-pooled output (fc lives in model.py, generation-only - see model.py's module
    docstring).
    texts: list[str], the organ's ground-truth report sentences.
    Returns the scalar next-token cross-entropy loss (LlamaForCausalLM's own loss,
    computed over the report-text positions only - image-token positions are masked
    out of the loss via label = -100, matching how Reg2RG/RadFM mask prompt tokens).
    """
    batch_size = image_embeds.size(0)

    tokenized = tokenizer(
        texts, padding="max_length", truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(device)
    text_ids = tokenized["input_ids"]  # (B, L)
    text_attention_mask = tokenized["attention_mask"]  # (B, L)

    text_embeds = lang_model.get_input_embeddings()(text_ids)  # (B, L, 4096)

    inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # (B, 32+L, 4096)

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
