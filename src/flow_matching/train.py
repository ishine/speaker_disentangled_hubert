import os
from pathlib import Path

import jiwer
import torch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoModel, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from ..bigvgan.bigvgan import BigVGan, BigVGanConfig
from .configs import FlowMatchingConfig
from .data import get_collate_fn
from .models import FlowMatchingModel
from .utils import fix_random_seed, get_input_embeddings, get_lr_schedule

# register BigVGan
AutoConfig.register("bigvgan", BigVGanConfig)
AutoModel.register(BigVGanConfig, BigVGan)


@torch.inference_mode()
def validate(config, dataloader, model: FlowMatchingModel, step: int, writer: SummaryWriter):
    model.eval()

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    vocoder = AutoModel.from_pretrained(config.vocoder.model_name_or_path).cuda()
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.asr.model_name_or_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(config.asr.model_name_or_path)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=asr,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
    )

    hyps = []

    for n, batch in enumerate(dataloader):
        spectrogram = model.synthesize(
            input_ids=batch["input_ids"].cuda(),
            dt=config.flow_matching.dt,
            truncation_value=config.flow_matching.truncation_value,
        )
        hyp_wav = vocoder(spectrogram)

        hyp_wav = hyp_wav.cpu().squeeze(0).numpy()
        ref_wav = batch["input_values"].squeeze(0).numpy()

        hyp = pipe(hyp_wav, generate_kwargs={"language": "english"}, return_timestamps=True)["text"]

        hyps.append(hyp)

        if n < 5:
            writer.add_audio(f"hyp/{batch['names'][0]}", hyp_wav, step, 16000)
            writer.add_audio(f"ref/{batch['names'][0]}", ref_wav, step, 16000)

    transcripts = [processor.tokenizer.normalize(transcript) for transcript in dataloader.dataset["transcript"]]
    hyps = [processor.tokenizer.normalize(hyp) for hyp in hyps]

    wer_hyp = jiwer.wer(transcripts, hyps) * 100
    cer_hyp = jiwer.cer(transcripts, hyps) * 100

    writer.add_scalar("dev/WER", wer_hyp, step)
    writer.add_scalar("dev/CER", cer_hyp, step)

    del vocoder
    del asr
    torch.cuda.empty_cache()


def train_flow_matching(config):
    fix_random_seed(config.common.seed)

    train_set = load_dataset(config.dataset.name, split="train", keep_in_memory=True).with_format("torch")
    dev_set = load_dataset(config.dataset.name, split="dev", keep_in_memory=True).with_format("torch")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.flow_matching.batch_size,
        shuffle=True,
        num_workers=config.flow_matching.num_workers,
        collate_fn=get_collate_fn(
            ext_audio=config.dataset.ext_audio,
        ),
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        num_workers=config.flow_matching.num_workers,
        collate_fn=get_collate_fn(
            wav_dir=config.dataset.wav_dir,
            ext_audio=config.dataset.ext_audio,
        ),
    )

    model = FlowMatchingModel(
        FlowMatchingConfig(
            vocab_size=config.flow_matching.vocab_size,
            dim_in=config.flow_matching.dim_in,
            dim_cond_emb=config.flow_matching.dim_cond_emb,
            hidden_size=config.flow_matching.hidden_size,
            depth=config.flow_matching.depth,
            heads=config.flow_matching.heads,
            intermediate_size=config.flow_matching.intermediate_size,
            attn_dropout=config.flow_matching.attn_dropout,
            ff_dropout=config.flow_matching.ff_dropout,
            use_unet_skip_connection=config.flow_matching.use_unet_skip_connection,
            mean=config.flow_matching.mean,
            std=config.flow_matching.std,
        ),
        get_input_embeddings(config.speech2unit.model_name_or_path),
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.flow_matching.lr)

    # learning rate scheduler
    lr_scheduler = get_lr_schedule(
        optimizer,
        config.flow_matching.epoch * len(train_loader),
        config.flow_matching.warmup_steps,
        config.flow_matching.lr,
        config.flow_matching.lr_min,
    )

    scaler = torch.amp.GradScaler("cuda", init_scale=1e24)
    writer = SummaryWriter(os.path.join(config.flow_matching.model_name_or_path, "logs"))

    last_epoch = 0
    step = 0

    for epoch in range(last_epoch + 1, config.flow_matching.epoch + 1):
        model.train()

        for batch in train_loader:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(
                    input_ids=batch["input_ids"].cuda(),
                    spectrogram_labels=batch["spectrogram_labels"].cuda(),
                    duration_labels=batch["duration_labels"].cuda(),
                )
            scaler.scale(loss).backward()

            # gradient clipping
            if config.flow_matching.max_norm is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.flow_matching.max_norm)

            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            optimizer.zero_grad()

            # update learning rate
            lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()

            step += 1

            # tensorboard log
            if step % config.flow_matching.summary_interval == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/scale", scale, step)
                if config.flow_matching.max_norm is not None:
                    writer.add_scalar("train/grad_norm", grad_norm.item(), step)

        if epoch % config.flow_matching.save_interval_epoch == 0:
            validate(config, dev_loader, model, step, writer)

            # save model
            Path(config.flow_matching.model_name_or_path).parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(config.flow_matching.model_name_or_path)
