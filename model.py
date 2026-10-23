"""Model construction for the Perceiver/BERT contrastive + generation pretraining.

Components:
    vision_encoder  - RadFM's 3D ViT (frozen), loaded from RadFM_vit3d.pth. Identical
                       architecture/weights to Reg2RG's own vision_encoder (verified:
                       byte-identical vit_3d.py, identical construction args).
    perceiver       - RadFM's PerceiverResampler (trainable, the target), warm-started
                       from RadFM_perceiver_fc.pth's 'perceiver' key. Built with
                       dim=768, num_latents=32 to match Reg2RG's self.perceiver exactly,
                       so perceiver.state_dict() can be loaded into Reg2RG unmodified.
    fc              - Linear(768, 4096), trainable, random init (Reg2RG's own
                       my_embedding_layer.py builds this same shape and, per its own
                       comment, *intends* to warm-start it from a pretrained 'fc' key,
                       but that load line is dead code - see build_fc's docstring - so
                       "fc trains from scratch" is Reg2RG's real, working precedent).
                       SHARED between both branches: encode_image applies it to
                       perceiver's output unconditionally, so its 4096-d result is both
                       (a) what the generation branch concatenates into LLaMA-2-7B-
                       chat's input directly, and (b) what the contrastive branch
                       compares token-by-token against text (see losses.info_nce_loss -
                       no pooling happens in this model; it averages 32 per-token
                       similarity scores, not the vectors). 4096 matches Reg2RG's
                       actual self.hidden_dim (not RadFM's own native 13B's 5120 - see
                       generation.py's module docstring), so fc.state_dict() becomes
                       directly loadable into Reg2RG's self.fc too, not just
                       perceiver's into self.perceiver.
    bert_text       - fvlm's text_encoder (frozen), warm-started from fvlm's own
                       originally-released checkpoint (model.pth, not the further-
                       finetuned checkpoint_040.pth - see FVLM_CHECKPOINT's own
                       comment for why, verified identical where it matters).
    adapter         - Linear(768, 4096), trainable, random init. Projects frozen
                       BERT's 768-d [CLS] output into fc's same 4096-d space, so the
                       contrastive branch compares two *projected* 4096-d vectors
                       instead of comparing a raw frozen-BERT vector against
                       perceiver's raw 768-d output directly. No pretrained
                       equivalent exists - this is a new module, and deliberately
                       does NOT reuse fvlm's own 256-d text_proj, since the design
                       here is "everything aligns to fc's space", not "everything
                       aligns to fvlm's existing retrieval space".

perceiver and fc are the only modules touched by both losses - see grad_balance.py.
adapter only ever receives gradient from the contrastive loss.

PerceiverResampler, ViT, and the BERT-environment patch function are copied in
directly from vendored.py rather than imported live from the Reg2RG archive/RadFM/
fvlm repos (see that file's own docstring for why and exactly what was copied) - this
project only needs those three repos' WEIGHT FILES, not their source checked out as
sibling directories. dataset.py is the one exception that still imports the archive's
own RadGenomeDataset_Train live, deliberately - see its docstring.

`lavis` (Salesforce's open-source LAVIS library, used below for XBertEncoder) isn't
fvlm's own code - fvlm just bundles a copy of it at fvlm/lavis/ instead of installing
it from PyPI (it's also published there as `salesforce-lavis`, but fvlm's bundled copy
may carry local patches/a pinned version, unverified against that release - safer to
keep an exact copy than risk a silent behavioral mismatch from swapping to the PyPI
one). Copied wholesale into this project's own lavis/ directory (not vendored.py,
which is hand-picked individual pieces - this is one working third-party library
copied as-is, 35 files, too interlinked to cherry-pick safely) so fvlm's repo no
longer needs to exist on disk at all for this project to run.
"""
import torch
import torch.nn as nn

from vendored import PerceiverResampler, ViT as RadFMViT, apply_environment_patches

RADFM_VIT3D_PATH = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/weights/Reg2RG_weights/RadFM_vit3d.pth"
RADFM_PERCEIVER_FC_PATH = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/weights/Reg2RG_weights/RadFM_perceiver_fc.pth"
#  fvlm's own originally-released checkpoint (4-organ, pretrained), not the
# further-finetuned 10-organ checkpoint_040.pth - text_encoder is byte-identical
# between the two (fvlm's own organ-expansion finetuning only ever touches new
# organs' query_tokens/vision_proj, never text_encoder/text_proj - see fvlm/
# KE_prepare_unique/2_compute_text_embeddings.py's own comment on this), so this
# carries zero quality cost while also being fvlm's publicly-released checkpoint
# (https://drive.google.com/drive/folders/15BnMo1lIAlOH_8KLdB2NugiHnmj9AWSD), not
# a private file.
FVLM_CHECKPOINT = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/weights/fvlm_weights/model.pth"

VIS_DIM = 768
PERCEIVER_NUM_LATENTS = 32  # must match Reg2RG's self.perceiver exactly
# Duplicated from generation.py's LLM_HIDDEN_DIM (not imported from there - model.py
# must not import generation.py, to avoid a circular import; generation.py's own
# constant stays the source of truth for the decoder's actual hidden size). Keep the
# two literals in sync if the generation decoder ever changes.
LLM_HIDDEN_DIM = 4096


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


def build_fc():
    """Bridges the Perceiver's 768-d tokens into LLaMA-2-7B-chat's 4096-d embedding
    space. Trainable, random init - Reg2RG's own my_embedding_layer.py builds this
    same shape of fc and (per its own comment) *intends* to warm-start it from a
    pretrained 'fc' checkpoint key, but that load line is dead code (it's commented
    out, and even if it weren't, self.fc doesn't exist yet at that point in __init__ -
    see the archive's source). Reg2RG's real, working precedent is therefore "fc
    trains from scratch", which this matches.

    Shared between both branches (see PerceiverAlignmentModel.encode_image): its
    output feeds the generation branch's LLaMA input directly, and is also what the
    contrastive branch compares token-by-token against text.
    """
    return nn.Linear(VIS_DIM, LLM_HIDDEN_DIM)


def build_adapter():
    """Projects the frozen BERT text encoder's 768-d [CLS] output into the same
    4096-d space fc produces for images, so the contrastive branch compares two
    4096-d vectors (both having passed through a trainable projection) instead of
    comparing a raw frozen-BERT vector against perceiver's raw 768-d output directly.
    Trainable, random init - no pretrained equivalent exists."""
    return nn.Linear(VIS_DIM, LLM_HIDDEN_DIM)


def build_bert_text_encoder(device):
    """fvlm's frozen text_encoder (BiomedVLP-CXR-BERT-based XBertEncoder). No
    text_proj - callers use the raw [CLS] hidden state (768-d) directly, then
    build_adapter's 768->4096 projection (above) handles the rest. Mirrors fvlm/
    KE_prepare_unique/2_compute_text_embeddings.py's build_text_side(), minus
    text_proj (fvlm's own text_proj targets its own 256-d retrieval space, not this
    project's 4096-d fc-aligned space).
    """
    from transformers import BertConfig, BertTokenizer
    from lavis.models.med import XBertEncoder

    apply_environment_patches()

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
    """Wraps vision_encoder (frozen) + perceiver (trainable) + fc (trainable,
    shared) + bert text_encoder (frozen) + adapter (trainable, contrastive-only).

    fc is shared between the two branches: encode_image applies it to perceiver's
    output unconditionally, so its 4096-d result is both (a) what the generation
    branch concatenates into LLaMA's input directly, and (b) what the contrastive
    branch compares token-by-token against BERT (see losses.info_nce_loss - no
    pooling happens in this model at all; it compares each of the 32 tokens against
    text and averages the resulting similarity scores, not the vectors themselves).
    adapter projects frozen BERT's 768-d [CLS] output into that same 4096-d space.
    """

    def __init__(self, device, init_temperature=0.07):
        super().__init__()
        self.device = device
        self.vision_encoder = build_vision_encoder().to(device)
        self.perceiver = build_perceiver().to(device)
        self.fc = build_fc().to(device)
        self.adapter = build_adapter().to(device)
        self.bert_tokenizer, self.bert_text_encoder = build_bert_text_encoder(device)
        # Learnable InfoNCE temperature, matching fvlm's own
        # self.temp = nn.Parameter(0.07 * torch.ones([])) - fvlm clamps it to
        # [0.001, 0.5] in-place before every use (blip_pretrain.py:196), not just
        # once; losses.info_nce_loss does the same clamp at point of use.
        self.temp = nn.Parameter(init_temperature * torch.ones([], device=device))

    def encode_image(self, volume):
        """volume: (B, 3, 256, 256, 64). Returns image_embeds: (B, 32, 4096) -
        perceiver's tokens after fc, un-pooled. Used as-is by both branches: the
        generation branch concatenates it directly into LLaMA's input, and
        losses.info_nce_loss compares it token-by-token against text (averaging
        similarity scores, not vectors - no pooling happens here)."""
        with torch.no_grad():
            patch_tokens, _ = self.vision_encoder(volume)  # (B, N, 768)
        # Perceiver expects (B, S, F, N, D) - S=num images/sample, F=frames-per-image;
        # one cropped volume per sample here, so S=F=1 (matches how Reg2RG's own
        # region branch calls self.perceiver: vision_temp.unsqueeze(2) after an
        # equivalent (b s) v d -> b s v d reshape with s=1).
        patch_tokens = patch_tokens.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, 768)
        tokens = self.perceiver(patch_tokens)  # (B, 1, 32, 768)
        tokens = tokens[:, 0]  # (B, 32, 768)
        image_embeds = self.fc(tokens)  # (B, 32, 4096)
        return image_embeds

    def encode_text(self, texts):
        """texts: list[str]. Returns (B, 4096) - frozen BERT's [CLS] hidden state,
        projected through the trainable adapter into fc's output space."""
        tokens = self.bert_tokenizer(
            texts, padding="max_length", truncation=True, max_length=384,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.bert_text_encoder.forward_text(tokens)
        cls = output.last_hidden_state[:, 0, :]  # (B, 768)
        return self.adapter(cls)  # (B, 4096)
