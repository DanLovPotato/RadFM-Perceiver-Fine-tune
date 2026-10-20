"""Training loop for the Perceiver/BERT alignment pretraining: contrastive loss
(Perceiver pooled output vs frozen BERT) + generation loss (Perceiver tokens -> fc ->
frozen LLaMA-13B decoder -> next-token cross-entropy against the organ's report),
combined as a weighted sum. --no_generation drops the LLaMA-13B branch entirely for a
lighter contrastive-only run.

Multi-GPU: data parallelism via torch.distributed/DistributedDataParallel, launched
with torchrun (matches Reg2RG's own launch convention - see scripts/train_radgenome.sh
in the Reg2RG archive). Each GPU holds its own full copy of every module (including
the frozen 13B decoder) and processes a different data shard; only perceiver, fc, and
temp need gradient sync across GPUs, since everything else is frozen. Launched without
torchrun (no RANK env var), this falls back to ordinary single-GPU/CPU execution
unchanged - that's what the smoke test and all single-GPU runs so far have used.

Batch composition note: InfoNCE's negatives are "every other item in the batch", so a
batch dominated by one organ (e.g. many 'lung' items, since lung has by far the most
samples) gives weak negatives - mostly easy same-organ-different-patient pairs rather
than a diverse mix. Not yet addressed here (plain shuffled/DistributedSampler
DataLoader); revisit if contrastive loss plateaus suspiciously fast.
"""
import argparse
import os

import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import PerOrganDataset, collate_skip_none
from generation import build_fc, build_lang_model, build_tokenizer, generation_loss
from grad_balance import balanced_backward
from losses import info_nce_loss
from model import PerceiverAlignmentModel

DATA_ROOT = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default=os.path.join(DATA_ROOT, "train_preprocessed"))
    parser.add_argument("--mask_folder", default=os.path.join(DATA_ROOT, "train_region_mask"))
    parser.add_argument("--csv_file", default=os.path.join(DATA_ROOT, "radgenome_files", "train_region_report.csv"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16,
                         help="per-GPU batch size - effective batch size is this "
                              "times the number of GPUs under torchrun")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07,
                         help="initial value only - this is now a learnable "
                              "parameter (matching fvlm's own self.temp), trained "
                              "alongside the Perceiver and clamped to [0.001, 0.5]")
    parser.add_argument("--contrastive_weight", type=float, default=0.2)
    parser.add_argument("--generation_weight", type=float, default=0.8)
    parser.add_argument("--no_generation", action="store_true",
                         help="contrastive loss only, skip the LLaMA-13B decoder "
                              "branch entirely (useful while its checkpoint isn't "
                              "ready yet, or for an ablation)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--smoke_test", action="store_true",
                         help="run a handful of steps on a tiny subset, then exit")
    parser.add_argument("--wandb_project", default="radfm-perceiver-align")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    # torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; a plain `python3 train.py` (the
    # smoke-test path so far) sets none of them, so this falls back to ordinary
    # single-process execution on whatever CUDA_VISIBLE_DEVICES names (still must be
    # set in the shell before launch - see the single-GPU note this repo already has
    # elsewhere: torch's CUDA subsystem queues lazy init at import time, so setting it
    # from inside this script would be too late).
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            raise RuntimeError(
                "Set CUDA_VISIBLE_DEVICES in the shell before launching this script."
            )
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    use_wandb = is_main and not args.no_wandb and not args.smoke_test
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or os.path.basename(args.output_dir.rstrip("/")),
            config=vars(args),
        )

    if is_main:
        print("Building dataset...")
    dataset = PerOrganDataset(args.data_folder, args.mask_folder, args.csv_file)
    if args.smoke_test:
        dataset.flat_index = dataset.flat_index[: args.batch_size * 3 * world_size]

    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, collate_fn=collate_skip_none, drop_last=True,
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_skip_none, drop_last=True,
        )

    if is_main:
        print("Building model...")
    model = PerceiverAlignmentModel(device, init_temperature=args.temperature)

    fc = lang_model = tokenizer = None
    if not args.no_generation:
        if is_main:
            print("Building generation branch (fc + frozen LLaMA-13B decoder)...")
        fc = build_fc().to(device)
        lang_model = build_lang_model(device)
        tokenizer = build_tokenizer()

    # No DDP wrapping: gradient-norm balancing (see grad_balance.py) needs two
    # backward passes per step whose results get manually combined and rewritten into
    # .grad *after* both complete, which doesn't fit DDP's hook-based "average
    # whatever backward() touches, as it happens" model. Every rank instead runs a
    # fully local forward/backward on its own data shard, and every trainable
    # parameter's gradient (perceiver, fc, temp) is explicitly all_reduce-averaged by
    # hand after combining - same approach already used for temp alone before this
    # change, just extended to cover perceiver/fc too.
    perceiver_module = model.perceiver
    fc_module = fc

    trainable_params = list(perceiver_module.parameters()) + [model.temp]
    if fc_module is not None:
        trainable_params += list(fc_module.parameters())

    if is_main:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in model.parameters())
        if lang_model is not None:
            n_total += sum(p.numel() for p in lang_model.parameters())
        if fc_module is not None:
            n_total += sum(p.numel() for p in fc_module.parameters())
        print(f"Trainable params: {n_trainable:,} / {n_total:,} total "
              f"({100 * n_trainable / n_total:.3f}%)")
        if distributed:
            print(f"Distributed: {world_size} GPUs, per-GPU batch_size="
                  f"{args.batch_size}, effective batch_size="
                  f"{args.batch_size * world_size}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        if distributed:
            sampler.set_epoch(epoch)  # reshuffles differently each epoch
        model.perceiver.train()
        if fc is not None:
            fc.train()
        for batch in loader:
            if batch is None:
                continue

            volume = batch["image"].to(device)
            texts = batch["text"]

            image_tokens, image_pooled = model.encode_image(volume)
            text_embeds = model.encode_text(texts)

            contrastive = info_nce_loss(image_pooled, text_embeds, model.temp)

            optimizer.zero_grad()

            if args.no_generation:
                # Nothing to balance - contrastive is the only loss touching
                # perceiver, so a plain backward is correct as-is.
                contrastive.backward()
                loss_value = contrastive.item()
                gen_loss_value = None
                c_grad_norm = g_grad_norm = None
                if distributed:
                    for p in trainable_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            else:
                generation = generation_loss(
                    image_tokens, texts, fc, lang_model, tokenizer, device,
                )
                # Computes both backward passes internally (contrastive first, with
                # retain_graph=True, then generation), normalizes each loss's
                # gradient on the shared perceiver params before combining with the
                # stated weights, and (if distributed) all_reduces the final combined
                # gradient for perceiver/fc - see grad_balance.py's module docstring
                # for why this replaces a plain weighted-sum .backward() call.
                c_grad_norm, g_grad_norm = balanced_backward(
                    contrastive, generation, perceiver_module.parameters(),
                    fc_module.parameters(), args.contrastive_weight,
                    args.generation_weight, distributed,
                )
                if distributed and model.temp.grad is not None:
                    dist.all_reduce(model.temp.grad, op=dist.ReduceOp.AVG)
                loss_value = (args.contrastive_weight * contrastive.item()
                              + args.generation_weight * generation.item())
                gen_loss_value = generation.item()

            optimizer.step()

            if is_main and global_step % args.log_every == 0:
                msg = f"epoch {epoch} step {global_step} loss {loss_value:.4f} contrastive {contrastive.item():.4f}"
                if gen_loss_value is not None:
                    msg += f" generation {gen_loss_value:.4f}"
                if c_grad_norm is not None:
                    msg += f" | grad_norm contrastive={c_grad_norm:.4f} generation={g_grad_norm:.4f}"
                print(msg)
                if use_wandb:
                    log_dict = {
                        "loss": loss_value,
                        "contrastive_loss": contrastive.item(),
                        "temperature": model.temp.item(),
                        "epoch": epoch,
                    }
                    if gen_loss_value is not None:
                        log_dict["generation_loss"] = gen_loss_value
                    if c_grad_norm is not None:
                        log_dict["contrastive_grad_norm"] = c_grad_norm
                        log_dict["generation_grad_norm"] = g_grad_norm
                    wandb.log(log_dict, step=global_step)

            if is_main and (not args.smoke_test) and global_step > 0 and global_step % args.save_every_steps == 0:
                ckpt_path = os.path.join(args.output_dir, f"perceiver-step{global_step}.pth")
                torch.save(perceiver_module.state_dict(), ckpt_path)
                print(f"Saved {ckpt_path}")

            global_step += 1

            if args.smoke_test and global_step >= 3:
                if is_main:
                    print("Smoke test passed: forward + backward + optimizer step "
                          "ran cleanly for 3 steps.")
                return

    if is_main:
        final_path = os.path.join(args.output_dir, "perceiver-final.pth")
        torch.save(perceiver_module.state_dict(), final_path)
        print(f"Saved {final_path}")
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
