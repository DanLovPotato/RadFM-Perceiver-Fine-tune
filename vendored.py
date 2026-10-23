"""Code copied in verbatim (not imported) from fvlm/RadFM/the Reg2RG archive, so this
project only depends on those repos for WEIGHT FILES, not their source code being
checked out as sibling directories. Chosen deliberately for these three pieces (not
for dataset.py's RadGenomeDataset_Train, which stays a live import - see dataset.py's
docstring) because:
    - they're small, stable, architecture-definition code unlikely to change upstream
    - copying PerceiverResampler and ViT here also eliminates the Model/-package-name-
      collision workaround model.py used to need (RadFM's and the archive's src/Model
      directories share that name on sys.path - see git history if that trick is ever
      needed again elsewhere)
If the originals ever change, this file needs a manual re-sync - there's no automated
link back to them.

Sources (copied as of 2026-10-23):
    PerceiverResampler, PerceiverAttention, FeedForward, exists
        <- Reg2RG-debug-current-full_correct_F_20260920_archive/src/Model/helpers.py
           ("Taken from https://github.com/lucidrains/flamingo-pytorch" per that
           file's own header) - only the Perceiver-related classes; that file also
           has MaskedCrossAttention/GatedCrossAttentionBlock, unused here, not copied.
    ViT, Transformer, Attention, PreNorm, pair
        <- RadFM/src/Model/RadFM/vit_3d.py
    PositionEmbeddingLearned3d
        <- RadFM/src/Model/RadFM/position_encoding.py (only the one class vit_3d.py
           actually uses; that file has two other position-encoding classes and a
           build_position_encoding() helper, unused here, not copied)
    apply_environment_patches
        <- fvlm/finetune.py's _apply_environment_patches() (renamed without the
           leading underscore since it's now a public cross-module API)
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many
from torch import einsum


# ---------------------------------------------------------------------------
# PerceiverResampler (archive's Model/helpers.py)
# ---------------------------------------------------------------------------

def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=64,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


# ---------------------------------------------------------------------------
# PositionEmbeddingLearned3d (RadFM's position_encoding.py)
# ---------------------------------------------------------------------------

class PositionEmbeddingLearned3d(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256, h_patch_num=16, w_patch_num=16, d_patch_num=64):
        super().__init__()
        self.h_patch_num = h_patch_num
        self.w_patch_num = w_patch_num
        self.d_patch_num = d_patch_num
        self.row_embed = nn.Embedding(h_patch_num, num_pos_feats)
        self.col_embed = nn.Embedding(w_patch_num, num_pos_feats)
        self.dep_embed = nn.Embedding(d_patch_num, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.dep_embed.weight)

    def forward(self, B, h, w, d, x):
        i = (torch.arange(h, device=x.device) + 1) * (self.h_patch_num // h) - 1
        j = (torch.arange(w, device=x.device) + 1) * (self.w_patch_num // w) - 1
        k = (torch.arange(d, device=x.device) + 1) * (self.d_patch_num // d) - 1
        x_emb = self.row_embed(i).unsqueeze(1).unsqueeze(2).repeat(1, w, d, 1)
        y_emb = self.col_embed(j).unsqueeze(0).unsqueeze(2).repeat(h, 1, d, 1)
        z_emb = self.dep_embed(k).unsqueeze(0).unsqueeze(1).repeat(h, w, 1, 1)
        pos = torch.cat([x_emb, y_emb, z_emb], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        pos = rearrange(pos, 'b h w d c -> b (h w d) c')
        return pos


# ---------------------------------------------------------------------------
# ViT (RadFM's vit_3d.py)
# ---------------------------------------------------------------------------

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class _ViTFeedForward(nn.Module):
    """Renamed from vit_3d.py's own FeedForward to avoid colliding with the
    Perceiver section's FeedForward above (same name, different implementation -
    this one is a plain MLP block, not the Perceiver's LayerNorm-first variant)."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, _ViTFeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.frame_patch_size = frame_patch_size

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (f pf) -> b (h w f) (p1 p2 pf c)', p1=patch_height, p2=patch_width, pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = PositionEmbeddingLearned3d(dim // 3, (image_height // patch_height), (image_width // patch_width), (frames // frame_patch_size))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, video):
        B, C, H, W, D = video.shape
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        pos = self.pos_embedding(B, H // self.patch_height, W // self.patch_width, D // self.frame_patch_size, x)
        x += pos
        x = self.dropout(x)

        x = self.transformer(x)
        return x, pos


# ---------------------------------------------------------------------------
# apply_environment_patches (fvlm's finetune.py's _apply_environment_patches())
# ---------------------------------------------------------------------------

def apply_environment_patches():
    """Patches for gaps in *this* environment (missing local tokenizer/config assets,
    a newer transformers than blip.py's hard pin, a real bug in XBertEncoder) - not
    workarounds for anything about model design. lavis (a pip package, not fvlm's own
    code) is what's being patched here; this has nothing to do with fvlm's own
    finetune.py logic beyond being copied from it.
    """
    from transformers import BertTokenizer

    from lavis.models.blip_models import blip as blip_module
    from lavis.models.med import XBertEncoder

    # blip.py asserts transformers<4.27, inherited from upstream BLIP's reliance on
    # HF's BertModel internals. med.py is a from-scratch reimplementation of BERT
    # (only using stable base classes), so that constraint doesn't apply here.
    blip_module.BlipBase.__init__ = nn.Module.__init__

    @classmethod
    def _init_tokenizer(cls):
        return BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    blip_module.BlipBase.init_tokenizer = _init_tokenizer

    # XBertEncoder overrides get_output_embeddings() but never defines the matching
    # set_output_embeddings(), which newer transformers' PreTrainedModel requires when
    # tie_word_embeddings=False (resize_token_embeddings() calls it directly, and
    # there's no base-class fallback). The MLM head is unused for organ-contrastive
    # training, so this only needs to keep resize functional.
    def _set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    XBertEncoder.set_output_embeddings = _set_output_embeddings
