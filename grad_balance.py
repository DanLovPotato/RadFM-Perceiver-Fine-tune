"""Gradient-norm balancing between the contrastive and generation losses, where they
share the Perceiver's parameters.

Why this exists: a plain weighted sum (contrastive_weight*contrastive_loss +
generation_weight*generation_loss).backward() lets whichever loss happens to produce
the larger gradient dominate Perceiver's updates, regardless of the loss weights -
measured directly (diag_gradient_scale.py) at 28x raw gradient-norm ratio
(generation's gradient on Perceiver vs contrastive's), 112x after applying the
0.2/0.8 weights. Training confirmed this: contrastive loss sat flat at ~ln(4) (the
untrained baseline) for 300+ steps while generation loss kept dropping - the
contrastive signal was being drowned out, not slow to start.

Fix: compute each loss's gradient on the *shared* Perceiver parameters separately,
normalize each to unit norm, then recombine with the stated weights - so the weights
actually control each loss's relative influence on Perceiver, independent of the
losses' raw numeric scale. fc and temp each only receive gradient from one loss
(generation and contrastive respectively), so they need no normalization - they just
take whichever backward pass actually touches them.

DDP note: this needs two backward passes per step (one per loss, sharing one forward
pass via retain_graph=True) instead of DDP's usual single backward-call-per-iteration
assumption, and the gradient values get manually rewritten *after* both backwards
complete - DDP's own hook-based gradient averaging (which fires during backward,
mid-flight) is not compatible with that rewrite. So training disables DDP's automatic
sync entirely here (via no_sync()) and this module does one explicit manual
all_reduce instead, over the final, already-combined gradient - the same approach
train.py already uses for model.temp (a lone Parameter DDP never wraps).
"""
import torch
import torch.distributed as dist


def _flat_norm(grads):
    sq_sum = sum(g.detach().float().pow(2).sum() for g in grads if g is not None)
    return sq_sum.sqrt() if torch.is_tensor(sq_sum) else torch.tensor(0.0)


def balanced_backward(
    contrastive_loss, generation_loss, perceiver_params, fc_params,
    contrastive_weight, generation_weight, distributed, eps=1e-8,
):
    """Replaces `(w1*contrastive_loss + w2*generation_loss).backward()`.

    perceiver_params: params shared by both losses - gradient-normalized before
        combining.
    fc_params: params touched only by generation_loss - just takes that gradient
        directly, scaled by generation_weight (no normalization needed - there's
        nothing competing with it).
    Leaves .grad populated on every param in both lists (and on temp, if it's part of
    the contrastive graph - the caller is responsible for temp exactly as before,
    this function only touches perceiver_params/fc_params).

    Returns (contrastive_grad_norm, generation_grad_norm) - the raw (pre-normalization)
    norms on perceiver_params, logged so the 28x-style imbalance can be watched over
    training rather than just assumed fixed once and forgotten.
    """
    perceiver_params = list(perceiver_params)
    fc_params = list(fc_params)
    all_params = perceiver_params + fc_params

    for p in all_params:
        p.grad = None

    # Pass 1: contrastive. retain_graph=True keeps the shared encode_image() forward
    # alive for pass 2 - both losses were computed from the same volume forward pass.
    contrastive_loss.backward(retain_graph=True)
    contrastive_grads = [
        p.grad.detach().clone() if p.grad is not None else None
        for p in perceiver_params
    ]
    contrastive_grad_norm = _flat_norm(contrastive_grads)

    for p in all_params:
        p.grad = None

    # Pass 2: generation. Touches both perceiver_params (via encode_image's shared
    # graph) and fc_params (generation_loss's own parameters).
    generation_loss.backward()
    generation_grads_perceiver = [
        p.grad.detach().clone() if p.grad is not None else None
        for p in perceiver_params
    ]
    generation_grad_norm = _flat_norm(generation_grads_perceiver)
    # fc's gradient from this pass is already sitting in p.grad for each fc param -
    # leave it there untouched, just apply generation_weight to it below.

    # Recombine perceiver's gradient: each loss's contribution normalized to unit
    # norm first, so contrastive_weight/generation_weight directly control relative
    # influence regardless of the two losses' raw gradient scale.
    c_scale = contrastive_weight / (contrastive_grad_norm + eps)
    g_scale = generation_weight / (generation_grad_norm + eps)
    for p, cg, gg in zip(perceiver_params, contrastive_grads, generation_grads_perceiver):
        combined = torch.zeros_like(p.data)
        if cg is not None:
            combined = combined + c_scale * cg
        if gg is not None:
            combined = combined + g_scale * gg
        p.grad = combined

    for p in fc_params:
        if p.grad is not None:
            p.grad = generation_weight * p.grad

    if distributed:
        for p in all_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    return contrastive_grad_norm.item(), generation_grad_norm.item()
