import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import editdistance
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CONFIGS
from models.svtr import build_model
from engine.codec import Encoder
from engine.loss import FocalCTCLoss
from engine.dataset import build_loaders
from engine.augment import get_train_transforms, get_val_transforms, cutmix_batch


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="tiny", choices=["tiny", "small", "base"])
    p.add_argument("--data",    default=None,  help="Override data root dir")
    p.add_argument("--weights", default=None,  help="Resume from checkpoint .pth")
    p.add_argument("--epochs",  type=int, default=None)
    p.add_argument("--device",  default=None,  help="cuda / cpu / cuda:1 etc.")
    return p.parse_args()


def setup_device(override=None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, criterion, encoder, device, beam_width, lexicon_re, desc="Eval"):
    model.eval()
    tot_loss  = 0.0
    correct   = 0
    total     = 0
    char_dist = 0
    char_len  = 0
    hard_negs = []

    pbar = tqdm(loader, desc=desc, leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for imgs, targets, lengths, raw_labels in pbar:
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=device.type != "cpu"):
            log_probs = model(imgs)

        preds_ctc = log_probs.permute(1, 0, 2)
        in_lens   = torch.full((preds_ctc.size(1),), preds_ctc.size(0),
                               dtype=torch.long, device=device)
        loss = criterion(preds_ctc, targets, in_lens, lengths)
        tot_loss += loss.item()

        lp_np = log_probs.cpu().numpy()
        for i, gt in enumerate(raw_labels):
            pred, conf = encoder.decode_beam(lp_np[i],
                                             beam_width=beam_width,
                                             lexicon_pattern=lexicon_re)
            correct   += int(pred == gt)
            total     += 1
            char_dist += editdistance.eval(pred, gt)
            char_len  += len(gt)
            if pred != gt:
                hard_negs.append((gt, pred, float(conf)))

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / max(total, 1) * 100:.2f}%",
            cer=f"{char_dist / max(char_len, 1) * 100:.2f}%",
        )

    acc      = correct / max(total, 1)
    char_acc = 1.0 - char_dist / max(char_len, 1)
    return tot_loss / len(loader), acc, char_acc, hard_negs


def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    grad_clip, device, epoch, use_sgm, use_frm, cutmix_alpha):
    model.train()
    tot_loss  = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} train",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for step, (imgs, targets, lengths, _) in enumerate(pbar, 1):
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        imgs, targets, lengths = cutmix_batch(imgs, targets, lengths, alpha=cutmix_alpha)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=device.type != "cpu"):
            if use_sgm and use_frm:
                log_probs, sgm_out = model(imgs, return_sgm=True)
                preds_ctc = log_probs.permute(1, 0, 2)
                in_lens   = torch.full((preds_ctc.size(1),), preds_ctc.size(0),
                                       dtype=torch.long, device=device)
                loss_ctc  = criterion(preds_ctc, targets, in_lens, lengths)
                sgm_ctc   = sgm_out.permute(1, 0, 2)
                loss_sgm  = criterion(sgm_ctc, targets, in_lens, lengths)
                loss      = loss_ctc + 0.1 * loss_sgm
            else:
                log_probs = model(imgs)
                preds_ctc = log_probs.permute(1, 0, 2)
                in_lens   = torch.full((preds_ctc.size(1),), preds_ctc.size(0),
                                       dtype=torch.long, device=device)
                loss      = criterion(preds_ctc, targets, in_lens, lengths)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tot_loss += loss.item()
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{tot_loss / step:.4f}",
            step=f"{step}/{n_batches}",
        )

    return tot_loss / n_batches


def print_header():
    cols = (f"{'Ep':>4} | {'Train':>8} | {'Val':>8} | {'Exact%':>7} | "
            f"{'Char%':>7} | {'LR':>9} | {'Time':>6} | {'Status'}")
    print("-" * len(cols))
    print(cols)
    print("-" * len(cols))


def main():
    args = get_args()
    cfg  = dict(CONFIGS[args.variant])

    if args.data:
        cfg["train_dir"]    = os.path.join(args.data, "train")
        cfg["val_dir"]      = os.path.join(args.data, "val")
        cfg["test_dir"]     = os.path.join(args.data, "test")
        cfg["train_labels"] = os.path.join(args.data, "train_labels.txt")
        cfg["val_labels"]   = os.path.join(args.data, "val_labels.txt")
        cfg["test_labels"]  = os.path.join(args.data, "test_labels.txt")

    if args.epochs:
        cfg["epochs"] = args.epochs

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device  = setup_device(args.device)
    encoder = Encoder(cfg["chars"])

    print(f"\n  transformers-ocr  |  variant={cfg['variant']}  |  device={device}")
    print(f"  chars={cfg['chars']}  num_classes={encoder.num_classes}\n")

    train_tf = get_train_transforms(cfg["img_h"], cfg["img_w"])
    val_tf   = get_val_transforms(cfg["img_h"], cfg["img_w"])
    train_loader, val_loader, test_loader = build_loaders(cfg, encoder, train_tf, val_tf)

    model = build_model(cfg["variant"], cfg["img_h"], cfg["img_w"],
                        encoder.num_classes,
                        use_frm=cfg["use_frm"],
                        use_sgm=cfg["use_sgm"]).to(device)
    print(f"  params = {model.param_count():,}\n")

    criterion = FocalCTCLoss(blank=encoder.blank,
                             gamma=cfg["focal_gamma"],
                             label_smoothing=cfg["label_smooth"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])

    # SequentialLR: linear warmup → cosine annealing (replaces custom lr_lambda)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=cfg["warmup_epochs"],
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(cfg["epochs"] - cfg["warmup_epochs"], 1),
        eta_min=cfg["lr"] * 1e-2,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg["warmup_epochs"]],
    )

    scaler = GradScaler(device_type=device.type) if device.type != "cpu" else None

    start_epoch  = 1
    best_acc     = 0.0
    patience_cnt = 0

    if args.weights and os.path.isfile(args.weights):
        ckpt        = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("val_acc", 0.0)
        print(f"  resumed from {args.weights}  epoch={ckpt['epoch']}  "
              f"val_acc={best_acc*100:.2f}%\n")

    os.makedirs(os.path.dirname(cfg["save_path"]) or ".", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{cfg['variant']}_train.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_acc,char_acc,lr\n")

    print_header()

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0         = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, cfg["grad_clip"], device,
            epoch, cfg["use_sgm"], cfg["use_frm"],
            cfg["cutmix_alpha"],
        )
        scheduler.step()
        val_loss, val_acc, char_acc, hard_negs = evaluate(
            model, val_loader, criterion, encoder, device,
            cfg["beam_width"], cfg["lexicon_re"],
            desc=f"Epoch {epoch:03d}  val",
        )
        elapsed = time.time() - t0
        cur_lr  = scheduler.get_last_lr()[0]

        if val_acc > best_acc:
            best_acc     = val_acc
            patience_cnt = 0
            status       = "BEST"
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":         val_acc,
                "char_acc":        char_acc,
                "cfg":             cfg,
            }, cfg["save_path"])
            if hard_negs:
                hn_path = f"logs/{cfg['variant']}_hard_negs_ep{epoch}.txt"
                with open(hn_path, "w") as f:
                    for gt, pred, conf in hard_negs:
                        f.write(f"gt={gt}\tpred={pred}\tconf={conf:.4f}\n")
        else:
            patience_cnt += 1
            status        = f"patience {patience_cnt}/{cfg['patience']}"

        print(f"{epoch:>4} | {train_loss:>8.4f} | {val_loss:>8.4f} | "
              f"{val_acc*100:>6.2f}% | {char_acc*100:>6.2f}% | "
              f"{cur_lr:>9.2e} | {elapsed:>5.1f}s | {status}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                    f"{val_acc:.6f},{char_acc:.6f},{cur_lr:.8f}\n")

        if patience_cnt >= cfg["patience"]:
            print(f"\n  early stop at epoch {epoch}")
            break

    print(f"\n  best val acc = {best_acc*100:.2f}%")
    print(f"  running test set ...")
    ckpt = torch.load(cfg["save_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, test_char, _ = evaluate(
        model, test_loader, criterion, encoder, device,
        cfg["beam_width"], cfg["lexicon_re"], desc="Test",
    )
    print(f"\n  TEST  loss={test_loss:.4f}  exact={test_acc*100:.2f}%  "
          f"char={test_char*100:.2f}%\n")


if __name__ == "__main__":
    main()