import os
import json
import argparse
import random
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import Datasets
from model_slimmable import ImageCompressor_slimmable
import models.synthesis_slimmable as synthesis_slimmable
import pruner.SlimmablePruner as SlimmablePruner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for CBANet slimmable model."
    )
    parser.add_argument("--config", required=True, help="Path to json config file")
    parser.add_argument(
        "--train_data_dir",
        required=True,
        help="Directory that contains training images",
    )
    parser.add_argument(
        "--save_dir",
        default="checkpoints/cbanet_train",
        help="Directory for checkpoints or final .pth.tar path",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (on Windows set 0 if multiprocessing is unstable)")
    parser.add_argument("--lr", type=float, default=None, help="Override config lr.base")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Optional max steps per epoch",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Resume from a train.py checkpoint (optional)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size used to crop/resize variable-size images for batching",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_modules(config: Dict, device: torch.device) -> Dict[str, nn.Module]:
    out_channel_N = config["out_channel_N"]
    out_channel_M = config["out_channel_M"]

    modules = {
        "Encoder1": ImageCompressor_slimmable(out_channel_N=out_channel_N, out_channel_M=out_channel_M),
        "Encoder2": ImageCompressor_slimmable(out_channel_N=out_channel_N, out_channel_M=out_channel_M),
        "Encoder3": ImageCompressor_slimmable(out_channel_N=out_channel_N, out_channel_M=out_channel_M),
        "Encoder4": ImageCompressor_slimmable(out_channel_N=out_channel_N, out_channel_M=out_channel_M),
        "Decoder1": synthesis_slimmable.Synthesis_net_slimmable(
            out_channel_N=int(out_channel_N * config["decoder_width1"]),
            out_channel_M=out_channel_M,
        ),
        "Decoder2": synthesis_slimmable.Synthesis_net_slimmable(
            out_channel_N=int(out_channel_N * config["decoder_width2"]),
            out_channel_M=out_channel_M,
        ),
        "Decoder3": synthesis_slimmable.Synthesis_net_slimmable(
            out_channel_N=int(out_channel_N * config["decoder_width3"]),
            out_channel_M=out_channel_M,
        ),
        "gate21": SlimmablePruner.Gate(),
        "gate31": SlimmablePruner.Gate(),
        "gate32": SlimmablePruner.Gate(),
        "Encoder_AQL1": SlimmablePruner.AQL(out_channel_M),
        "Encoder_AQL2": SlimmablePruner.AQL(out_channel_M),
        "Encoder_AQL3": SlimmablePruner.AQL(out_channel_M),
        "Decoder_IAQL1": SlimmablePruner.IAQL(out_channel_M),
        "Decoder_IAQL2": SlimmablePruner.IAQL(out_channel_M),
        "Decoder_IAQL3": SlimmablePruner.IAQL(out_channel_M),
        "priorEncoder_IAQL1": SlimmablePruner.IAQL(out_channel_M),
        "priorEncoder_IAQL2": SlimmablePruner.IAQL(out_channel_M),
        "priorEncoder_IAQL3": SlimmablePruner.IAQL(out_channel_M),
        "priorDecoder_AQL1": SlimmablePruner.AQL(out_channel_M),
        "priorDecoder_AQL2": SlimmablePruner.AQL(out_channel_M),
        "priorDecoder_AQL3": SlimmablePruner.AQL(out_channel_M),
    }

    for module in modules.values():
        module.to(device)

    return modules


def set_train(modules: Dict[str, nn.Module]):
    for module in modules.values():
        module.train()


def collect_parameters(modules: Dict[str, nn.Module]):
    params = []
    for module in modules.values():
        params.extend(list(module.parameters()))
    return params


def resolve_save_paths(save_dir_or_file: str) -> Tuple[str, str]:
    lower = save_dir_or_file.lower()
    if lower.endswith(".pth") or lower.endswith(".pth.tar") or lower.endswith(".pt"):
        final_path = save_dir_or_file
        save_dir = os.path.dirname(save_dir_or_file) or "."
    else:
        save_dir = save_dir_or_file
        final_path = os.path.join(save_dir, "final_model.pth.tar")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, final_path


def save_checkpoint(path: str, modules: Dict[str, nn.Module], optimizer, epoch: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
    }
    for name, module in modules.items():
        state[f"{name}_state_dict"] = module.state_dict()
    torch.save(state, path)


def load_train_checkpoint(path: str, modules: Dict[str, nn.Module], optimizer=None) -> int:
    ckpt = torch.load(path)
    for name, module in modules.items():
        key = f"{name}_state_dict"
        if key in ckpt:
            module.load_state_dict(ckpt[key])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


def rd_loss(recon: torch.Tensor, target: torch.Tensor, bpp: torch.Tensor, lamb: float) -> torch.Tensor:
    mse = torch.mean((recon - target).pow(2))
    return lamb * mse + bpp


def _resize_if_needed(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, h, w = img.shape
    if h >= patch_size and w >= patch_size:
        return img
    scale = max(patch_size / max(h, 1), patch_size / max(w, 1))
    new_h = max(int(round(h * scale)), patch_size)
    new_w = max(int(round(w * scale)), patch_size)
    img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)
    return img.squeeze(0)


def _random_crop(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    img = _resize_if_needed(img, patch_size)
    _, h, w = img.shape
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return img[:, top:top + patch_size, left:left + patch_size]


def _collate_fn(batch: List[torch.Tensor], patch_size: int) -> torch.Tensor:
    cropped = [_random_crop(x, patch_size) for x in batch]
    return torch.stack(cropped, dim=0)

class FixedPatchCollate:
    """Pickle-friendly collate_fn (works on Windows with num_workers>0)."""

    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def __call__(self, batch: List[torch.Tensor]) -> torch.Tensor:
        return _collate_fn(batch, self.patch_size)


def train_one_step(x: torch.Tensor, modules: Dict[str, nn.Module], lambdas: Tuple[float, float, float, float]):
    Encoder4 = modules["Encoder4"]
    Decoder1 = modules["Decoder1"]
    Decoder2 = modules["Decoder2"]
    Decoder3 = modules["Decoder3"]
    gate21 = modules["gate21"]
    gate31 = modules["gate31"]
    gate32 = modules["gate32"]

    losses = []

    def decode_three_widths(feature):
        r1 = Decoder1(feature)
        r2 = Decoder2(feature)
        r3 = Decoder3(feature)
        out1 = r1
        out2 = gate21(r1) + r2
        out3 = gate31(r1) + gate32(r2) + r3
        return out1, out2, out3

    for idx in (1, 2, 3):
        aql = modules[f"Encoder_AQL{idx}"]
        iaql = modules[f"Decoder_IAQL{idx}"]
        prior_iaql = modules[f"priorEncoder_IAQL{idx}"]
        prior_aql = modules[f"priorDecoder_AQL{idx}"]
        _, _, bpp, feature = Encoder4(x, (aql, iaql, prior_iaql, prior_aql))
        feature = iaql(feature)
        rec1, rec2, rec3 = decode_three_widths(feature)
        lamb = lambdas[idx - 1]
        losses.append(rd_loss(rec1, x, bpp, lamb))
        losses.append(rd_loss(rec2, x, bpp, lamb))
        losses.append(rd_loss(rec3, x, bpp, lamb))

    _, _, bpp4, feature4 = Encoder4(x)
    rec1_4, rec2_4, rec3_4 = decode_three_widths(feature4)
    lamb = lambdas[3]
    losses.append(rd_loss(rec1_4, x, bpp4, lamb))
    losses.append(rd_loss(rec2_4, x, bpp4, lamb))
    losses.append(rd_loss(rec3_4, x, bpp4, lamb))

    return torch.stack(losses).mean()


def sync_encoders(modules: Dict[str, nn.Module]):
    e4 = modules["Encoder4"].state_dict()
    modules["Encoder1"].load_state_dict(e4)
    modules["Encoder2"].load_state_dict(e4)
    modules["Encoder3"].load_state_dict(e4)


def main():
    args = parse_args()
    config = load_config(args.config)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this repository: model forward uses hard-coded .cuda() tensors."
        )

    device = torch.device("cuda")
    save_dir, final_path = resolve_save_paths(args.save_dir)

    batch_size = args.batch_size or config.get("batch_size", 8)
    base_lr = args.lr if args.lr is not None else config.get("lr", {}).get("base", 1e-4)

    dataset = Datasets(args.train_data_dir, image_size=args.patch_size)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in train_data_dir: {args.train_data_dir}")

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=FixedPatchCollate(args.patch_size),
    )

    modules = build_modules(config, device)
    params = collect_parameters(modules)
    optimizer = torch.optim.Adam(params, lr=base_lr)

    start_epoch = 0
    if args.resume:
        start_epoch = load_train_checkpoint(args.resume, modules, optimizer)
        print(f"Resumed from epoch {start_epoch}")

    lambdas = (
        float(config["train_lambda1"]),
        float(config["train_lambda2"]),
        float(config["train_lambda3"]),
        float(config["train_lambda4"]),
    )

    for epoch in range(start_epoch, args.epochs):
        set_train(modules)
        epoch_loss = 0.0
        steps = 0

        for step, x in enumerate(loader):
            if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
                break

            x = x.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = train_one_step(x, modules, lambdas)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            steps += 1

            if (step + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] Step [{step + 1}] "
                    f"Loss: {epoch_loss / max(steps, 1):.6f}"
                )

        sync_encoders(modules)

        avg_loss = epoch_loss / max(steps, 1)
        print(f"Epoch [{epoch + 1}/{args.epochs}] avg loss: {avg_loss:.6f}")

        if ((epoch + 1) % args.save_every) == 0:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pth.tar")
            save_checkpoint(ckpt_path, modules, optimizer, epoch + 1)
            print(f"Saved checkpoint: {ckpt_path}")

    save_checkpoint(final_path, modules, optimizer, args.epochs)
    print(f"Training done. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()

#python train.py --config config/OneEncoderPruner.json --train_data_dir ./datasets/archive/val2017/train --save_dir checkpoints/cbanet_train/cbanet3_1.pth.tar --epochs 200
