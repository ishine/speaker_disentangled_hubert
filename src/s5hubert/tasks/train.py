from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ...sdhubert.utils.syllable import BoundaryDetectionEvaluator
from ..mincut.mincut_utils import parallel_mincut
from ..models.s5hubert import S5Hubert
from ..models.s5hubert_dino import S5HubertDino
from ..utils.data import LibriSpeech
from ..utils.misc import fix_random_seed, get_tri_stage_schedule


@torch.inference_mode()
def validate(config, model, writer: SummaryWriter, step: int):
    model.eval()
    torch.cuda.empty_cache()

    wav_dir = Path(config.dataset.root) / "LibriSpeech"
    segment_dir = Path(config.path.segment_dir)
    segment_paths = []

    with open(config.dataset.dev_file) as f:
        for n, wav_name in enumerate(f):
            wav_name = wav_name.rstrip()
            wav_path = wav_dir / wav_name
            wav_path = str(wav_path)  # for sox backend
            wav, sr = torchaudio.load(wav_path)
            wav = wav.cuda()

            hidden_states, _ = model.student_forward(wav)
            hidden_states = hidden_states[config.model.segmentation_layer].squeeze(0).cpu().numpy()
            outputs = {"hidden_states": hidden_states}

            # save hidden states
            segment_name = wav_name.replace(".flac", ".npy")
            segment_path = segment_dir / segment_name
            segment_path.parent.mkdir(parents=True, exist_ok=True)
            segment_paths.append(segment_path)
            np.save(segment_path, outputs)

            if n < 10:
                similarity_mat = hidden_states @ hidden_states.T
                min_value = np.min(similarity_mat)
                max_value = np.max(similarity_mat)
                similarity_mat = (similarity_mat - min_value) / (max_value - min_value)
                writer.add_image(f"dev/{n}", similarity_mat, step, dataformats="HW")

    parallel_mincut(
        segment_paths,
        config.common.disable_tqdm,
        config.mincut.merge_threshold,
        config.mincut.max_duration,
        config.mincut.num_workers,
    )

    results = BoundaryDetectionEvaluator(
        config.path.segment_dir,
        config.dataset.dev_alignment,
        config.dataset.dev_alignment,
        tolerance=0.05,
        max_val_num=None,
    ).evaluate()

    for key in results:
        writer.add_scalar(f"dev/{key}", results[key], step)

    model.train()
    torch.cuda.empty_cache()

    return results


def train(config):
    fix_random_seed(config.common.seed)

    train_dataset = ConcatDataset(
        [
            LibriSpeech(
                root=config.dataset.root,
                url="train-clean-100",
                download=config.dataset.download,
                max_sample_size=config.dataset.max_sample_size,
            ),
            LibriSpeech(
                root=config.dataset.root,
                url="train-clean-360",
                download=config.dataset.download,
                max_sample_size=config.dataset.max_sample_size,
            ),
            LibriSpeech(
                root=config.dataset.root,
                url="train-other-500",
                download=config.dataset.download,
                max_sample_size=config.dataset.max_sample_size,
            ),
        ]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        collate_fn=LibriSpeech.collate_fn,
    )

    if config.model.model_type == "s5hubert":
        model = S5Hubert(
            model_name_or_path=config.model.model_name_or_path,
            init_last_layer=config.model.init_last_layer,
            head_out_size=config.model.head_out_size,
            head_hidden_size=config.model.head_hidden_size,
            ema_decay=config.model.ema_decay,
        ).cuda()
    elif config.model.model_type == "s5hubert_dino":
        model = S5HubertDino(
            model_name_or_path=config.model.model_name_or_path,
            init_last_layer=config.model.init_last_layer,
            head_out_size=config.model.head_out_size,
            head_hidden_size=config.model.head_hidden_size,
            head_bottleneck_size=config.model.head_bottleneck_size,
            teacher_temp=config.model.teacher_temp,
            student_temp=config.model.student_temp,
            center_momentum=config.model.center_momentum,
            ema_decay=config.model.ema_decay,
        ).cuda()
    else:
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )

    # learning rate scheduler
    assert config.optim.stage_ratio[0] + config.optim.stage_ratio[1] + config.optim.stage_ratio[2] == 1
    T_max = config.optim.epoch * len(train_loader)
    warmup_steps = int(T_max * config.optim.stage_ratio[0])
    hold_steps = int(T_max * config.optim.stage_ratio[1])
    decay_steps = T_max - warmup_steps - hold_steps
    lr_scheduler = get_tri_stage_schedule(
        optimizer,
        config.optim.lr,
        config.optim.lr_min,
        warmup_steps,
        hold_steps,
        decay_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config.common.fp16)
    writer = SummaryWriter(config.path.checkpoint)

    last_epoch = 0
    step = 0

    # resume training
    # if Path(config.path.checkpoint).is_file():
    #     ckpt = torch.load(config.path.checkpoint, weights_only=True)

    #     last_epoch = ckpt["epoch"]
    #     step = ckpt["step"]
    #     model.load_state_dict(ckpt["model"])
    #     optimizer.load_state_dict(ckpt["optimizer"])
    #     lr_scheduler.load_state_dict(ckpt["scheduler"])
    #     scaler.load_state_dict(ckpt["scaler"])

    #     print(f"load from {config.path.checkpoint}")
    #     del ckpt

    if step < warmup_steps:
        model.freeze_pretrained_modules()
    else:
        model.defrost_transformer_encoder()

    for epoch in range(last_epoch + 1, config.optim.epoch + 1):
        model.train()

        for batch in tqdm(train_loader, desc=f"epoch {epoch}", disable=config.common.disable_tqdm):
            with torch.amp.autocast("cuda", enabled=config.common.fp16):
                loss = model(
                    teacher_input_values=batch["teacher_input_values"].cuda(),
                    student_input_values=batch["student_input_values"].cuda(),
                    teacher_attention_mask=batch["teacher_attention_mask"].cuda(),
                    student_attention_mask=batch["student_attention_mask"].cuda(),
                )
            scaler.scale(loss).backward()

            # gradient clipping
            if config.optim.max_norm is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_norm)

            # update student
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            optimizer.zero_grad()

            # update teacher
            model.update_teacher()

            # update learning rate
            lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()

            step += 1

            # tensorboard log
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/scale", scale, step)
            if config.optim.max_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)

            if step == warmup_steps:
                model.defrost_transformer_encoder()

        results = validate(config, model, writer, step)

        # save model
        # ckpt = {
        #     "epoch": epoch,
        #     "step": step,
        #     "model": model.state_dict(),
        #     "optimizer": optimizer.state_dict(),
        #     "scheduler": lr_scheduler.state_dict(),
        #     "scaler": scaler.state_dict(),
        # }
        Path(config.path.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        model.student.save_pretrained(config.path.checkpoint)
        # torch.save(ckpt, config.path.checkpoint)
