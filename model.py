"""Model construction for the Perceiver/BERT contrastive (+ later, generation)
pretraining.

Components, all built to be load-compatible with their original source checkpoints:
    vision_encoder  - RadFM's 3D ViT (frozen), loaded from RadFM_vit3d.pth. Identical
                       architecture/weights to Reg2RG's own vision_encoder (verified:
                       byte-identical vit_3d.py, identical construction args).
    perceiver       - RadFM's PerceiverResampler (trainable, the target), warm-started
                       from RadFM_perceiver_fc.pth's 'perceiver' key. Built with
                       dim=768, num_latents=32 to match Reg2RG's self.perceiver exactly,
                       so perceiver.state_dict() can be loaded into Reg2RG unmodified.
    bert_text       - fvlm's text_encoder (frozen), warm-started from fvlm's
                       checkpoint_040.pth. No projection head (text_proj) is used - the
                       raw 768-d [CLS] hidden state is compared directly against the
                       perceiver's pooled 768-d output, so no new image-side projection
                       head is needed either (see conversation: dropping text_proj was
                       the chosen way to avoid adding a throwaway image projection head).

generation branch (fc + frozen LLaMA-13B decoder) is added once RadFM's full released
checkpoint finishes downloading - not wired up here yet.
"""
import importlib
import importlib.util
import os
import sys

import torch
import torch.nn as nn

_FVLM_ROOT = "/home/wisc/dxiang23/Projects/chestCT/code/fvlm"
_ARCHIVE_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Reg2RG-debug-current-full_correct_F_20260920_archive",
    "src",
)
for path in (_FVLM_ROOT, _ARCHIVE_SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

# archive Reg2RG's src/Model and RadFM's src/Model are two different directory trees
# that both happen to be named "Model" - archive's has an __init__.py (a real
# package), RadFM's doesn't (a namespace package), and Python's import resolution
# picks the real package globally once *either* is found anywhere on sys.path, which
# would silently make `Model.RadFM.*` resolve into archive's tree instead (and fail,
# since archive's Model has no RadFM/ subdirectory). `Model.helpers.PerceiverResampler`
# below deliberately relies on that resolution landing on archive's package (that's
# the exact same PerceiverResampler Reg2RG's own self.perceiver uses) - RadFM's ViT is
# loaded separately, under its own private module name, to avoid needing RadFM's
# Model/ on sys.path under the same "Model" name at all.
from Model.helpers import PerceiverResampler  # noqa: E402


def _load_radfm_vit_class():
    radfm_model_dir = "/home/wisc/dxiang23/Projects/chestCT/code/RadFM/src/Model/RadFM"
    pkg_name = "_radfm_model_pkg"
    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name, os.path.join(radfm_model_dir, "__init__.py"),
            submodule_search_locations=[radfm_model_dir],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = module
        spec.loader.exec_module(module)
    vit3d_module = importlib.import_module(f"{pkg_name}.vit_3d")
    return vit3d_module.ViT


RadFMViT = _load_radfm_vit_class()

RADFM_VIT3D_PATH = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/weights/Reg2RG_weights/RadFM_vit3d.pth"
RADFM_PERCEIVER_FC_PATH = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/weights/Reg2RG_weights/RadFM_perceiver_fc.pth"
FVLM_CHECKPOINT = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/weights/fvlm_weights/finetuned/checkpoint_040.pth"

VIS_DIM = 768
PERCEIVER_NUM_LATENTS = 32  # must match Reg2RG's self.perceiver exactly


def build_vision_encoder():
    """RadFM's 3D ViT, frozen, loaded strict from RadFM_vit3d.pth - identical
    construction args to Reg2RG's own vision_encoder (image_size=512, frames=512,
    patch_size=32, frame_patch_size=4, depth=12, heads=8, mlp_dim=2048)."""
    vit = RadFMViT(
        image_size=512, frames=512, image_patch_size=32, frame_patch_size=4,
        dim=VIS_DIM, depth=12, heads=8, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
    )
    state_dict = torch.load(RADFM_VIT3D_PATH, map_location="cpu")
    vit.load_state_dict(state_dict, strict=True)
    for param in vit.parameters():
        param.requires_grad = False
    vit.eval()
    return vit


def build_perceiver():
    """Trainable - warm-started from RadFM's own pretrained perceiver weights rather
    than random init, matching Reg2RG's own usage of pretrained_adapter."""
    perceiver = PerceiverResampler(dim=VIS_DIM, num_latents=PERCEIVER_NUM_LATENTS)
    ckpt = torch.load(RADFM_PERCEIVER_FC_PATH, map_location="cpu")
    perceiver.load_state_dict(ckpt["perceiver"], strict=True)
    for param in perceiver.parameters():
        param.requires_grad = True
    return perceiver


def build_bert_text_encoder(device):
    """fvlm's frozen text_encoder (BiomedVLP-CXR-BERT-based XBertEncoder). No
    text_proj - callers use the raw [CLS] hidden state (768-d) directly. Mirrors
    fvlm/KE_prepare_unique/2_compute_text_embeddings.py's build_text_side(), minus
    text_proj.
    """
    from transformers import BertConfig, BertTokenizer
    from lavis.models.med import XBertEncoder

    import finetune as fvlm_finetune  # fvlm/finetune.py
    fvlm_finetune._apply_environment_patches()

    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    bert_config = BertConfig(
        vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=512, type_vocab_size=2,
        hidden_dropout_prob=0.25, attention_probs_dropout_prob=0.25,
        layer_norm_eps=1e-12, pad_token_id=0,
        add_type_embeddings=False, tie_word_embeddings=False,
    )
    text_encoder = XBertEncoder(config=bert_config, add_pooling_layer=False)
    text_encoder.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(FVLM_CHECKPOINT, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    text_encoder_sd = {
        k[len("text_encoder."):]: v for k, v in state_dict.items()
        if k.startswith("text_encoder.")
    }
    expected_missing = {
        f"cls.predictions.{k}" for k in (
            "bias", "transform.dense.weight", "transform.dense.bias",
            "transform.LayerNorm.weight", "transform.LayerNorm.bias",
        )
    }
    missing, unexpected = text_encoder.load_state_dict(text_encoder_sd, strict=False)
    missing = [k for k in missing if k not in expected_missing]
    assert not unexpected, f"unexpected text_encoder checkpoint keys: {unexpected}"
    assert not missing, f"text_encoder missing expected keys: {missing}"

    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder = text_encoder.to(device).eval()

    return tokenizer, text_encoder


class PerceiverAlignmentModel(nn.Module):
    """Wraps vision_encoder (frozen) + perceiver (trainable) + bert text_encoder
    (frozen) for the contrastive branch. encode_image/encode_text return raw 768-d
    vectors - L2-normalization happens in the loss function, not here, so the model's
    outputs stay inspectable/reusable (e.g. for the later generation branch, which
    needs the un-normalized 32-token sequence, not the pooled+normalized vector).
    """

    def __init__(self, device, init_temperature=0.07):
        super().__init__()
        self.device = device
        self.vision_encoder = build_vision_encoder().to(device)
        self.perceiver = build_perceiver().to(device)
        self.bert_tokenizer, self.bert_text_encoder = build_bert_text_encoder(device)
        # Learnable InfoNCE temperature, matching fvlm's own
        # self.temp = nn.Parameter(0.07 * torch.ones([])) - fvlm clamps it to
        # [0.001, 0.5] in-place before every use (blip_pretrain.py:196), not just
        # once; losses.info_nce_loss does the same clamp at point of use.
        self.temp = nn.Parameter(init_temperature * torch.ones([], device=device))

    def encode_image(self, volume):
        """volume: (B, 3, 256, 256, 64). Returns (tokens, pooled):
        tokens: (B, 32, 768) - the perceiver's raw latent tokens (for the generation
            branch, once wired up).
        pooled: (B, 768) - mean-pooled over the 32 tokens (for the contrastive loss).
        """
        with torch.no_grad():
            patch_tokens, _ = self.vision_encoder(volume)  # (B, N, 768)
        # Perceiver expects (B, S, F, N, D) - S=num images/sample, F=frames-per-image;
        # one cropped volume per sample here, so S=F=1 (matches how Reg2RG's own
        # region branch calls self.perceiver: vision_temp.unsqueeze(2) after an
        # equivalent (b s) v d -> b s v d reshape with s=1).
        patch_tokens = patch_tokens.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, 768)
        tokens = self.perceiver(patch_tokens)  # (B, 1, 32, 768)
        tokens = tokens[:, 0]  # (B, 32, 768)
        pooled = tokens.mean(dim=1)  # (B, 768)
        return tokens, pooled

    def encode_text(self, texts):
        """texts: list[str]. Returns (B, 768) - raw [CLS] hidden state, frozen BERT,
        no projection head."""
        tokens = self.bert_tokenizer(
            texts, padding="max_length", truncation=True, max_length=384,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.bert_text_encoder.forward_text(tokens)
        return output.last_hidden_state[:, 0, :]  # (B, 768)
