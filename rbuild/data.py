"""
R-Build data — no-fuss batch sources for every model type.

Universal JSON manifest (works with blind, ViT, and encoderless models):

    [
      {"text": "any plain text"},
      {"text": "a caption for one picture", "images": ["pic.jpg"]},
      {"text": "several angles", "images": ["a.jpg", "https://.../b.png"]},
      {"text": "describe the clip", "video": "clip.mp4"},
      {"text": "paint a red square", "image_out": "red.png"},
      {"text": "say hello", "audio_out": "hello.wav"},
      {"text": "make it move", "video_out": "move.mp4"}
    ]

Also accepts .jsonl (one object per line). Links can be local paths or
http(s) URLs. Media is fetched, decoded, resized to `vision.image_size`,
and spliced into the token stream as image-placeholder runs automatically
— the manifest author never counts vision tokens. Blind models use the
text and skip the media (with a one-time warning). The *_out keys are the
v3.1 generative targets (cfg.outgen): the model trains to PRODUCE that
image / waveform / clip after the text — the same placeholder run
model.generate_image / generate_audio / generate_video appends.

HuggingFace streaming (needs: pip install .[train]):

    hf_image_batches("lambdalabs/naruto-blip-captions", cfg)
    hf_video_batches("friedrichor/MSR-VTT", cfg)

Column names are auto-detected from the first streamed row (image = PIL
value, video = path/URL/bytes column, text = first string) — override with
text_column= / image_column= / video_column= for exotic datasets.

Every generator yields (x, y) for blind models and (x, y, images) for
vision models — Trainer.fit() accepts both. Video decoding tries decord,
then imageio, then an ffmpeg/ffprobe subprocess — install whichever you
like; images only need Pillow.
"""

from __future__ import annotations

import io
import json
import os
import random
import subprocess
import tempfile
import urllib.request
from typing import Any, Dict, Iterable, List, Optional

import torch

# --------------------------------------------------------------------------- #
# sources: local path or URL -> bytes
# --------------------------------------------------------------------------- #

_URL_TIMEOUT = 30


def _read_bytes(source: Any) -> bytes:
    """Local path, http(s) URL, raw bytes, or {'path'|'bytes'} dict -> bytes."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, dict):
        if source.get("bytes") is not None:
            return bytes(source["bytes"])
        source = source.get("path")
    if not isinstance(source, str):
        raise ValueError(f"unsupported media source {type(source)}")
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=_URL_TIMEOUT) as r:
            return r.read()
    with open(source, "rb") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# media decoding
# --------------------------------------------------------------------------- #

def _pil_to_tensor(img, image_size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((image_size, image_size))
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return t.view(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0


def load_image(source: Any, image_size: int) -> torch.Tensor:
    """One image -> (3, image_size, image_size) float tensor in [0, 1]."""
    from PIL import Image
    return _pil_to_tensor(Image.open(io.BytesIO(_read_bytes(source))), image_size)


def _uniform_indices(total: int, n: int) -> List[int]:
    if n <= 1:
        return [0]
    if total <= n:
        return [min(i, total - 1) for i in range(n)]
    return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]


def load_audio(source: Any, n_samples: int, sample_rate: int = 16000) -> torch.Tensor:
    """
    One audio file -> (n_samples,) float waveform in [-1, 1] at sample_rate.
    .wav via the stdlib; anything else via an ffmpeg subprocess. Shorter
    clips are zero-padded, longer ones truncated; wrong sample rates are
    linearly resampled.
    """
    raw = _read_bytes(source)
    wav = sr = None
    try:  # stdlib wave — no deps for plain .wav
        import wave
        with wave.open(io.BytesIO(raw)) as w:
            sr = w.getframerate()
            ch, sw = w.getnchannels(), w.getsampwidth()
            frames = w.readframes(w.getnframes())
        if sw == 2:
            t = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
        elif sw == 1:
            t = torch.frombuffer(bytearray(frames), dtype=torch.uint8)
            t = (t.float() - 128.0) / 128.0
        else:  # 32-bit pcm
            t = torch.frombuffer(bytearray(frames), dtype=torch.int32).float() / 2147483648.0
        wav = t.view(-1, ch).mean(dim=1) if ch > 1 else t
    except Exception:
        wav = None
    if wav is None:  # ffmpeg fallback — mp3/ogg/flac/whatever
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            out = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", tmp_path, "-ac", "1", "-ar",
                 str(sample_rate), "-f", "f32le", "-"],
                capture_output=True, check=True)
            wav = torch.frombuffer(bytearray(out.stdout), dtype=torch.float32).clone()
            sr = sample_rate
        finally:
            os.unlink(tmp_path)
    if sr != sample_rate:  # naive linear resample
        n_new = max(1, int(round(wav.shape[0] * sample_rate / sr)))
        wav = torch.nn.functional.interpolate(
            wav.view(1, 1, -1), size=n_new, mode="linear", align_corners=False).view(-1)
    if wav.shape[0] < n_samples:
        wav = torch.cat([wav, torch.zeros(n_samples - wav.shape[0])])
    return wav[:n_samples].clamp(-1.0, 1.0)


def load_video(source: Any, n_frames: int, image_size: int) -> torch.Tensor:
    """
    One video -> (n_frames, 3, image_size, image_size) float tensor.
    Frames are sampled uniformly over the clip. Decoders tried in order:
    decord, imageio, ffmpeg/ffprobe subprocess.
    """
    raw = _read_bytes(source)
    frames = None

    try:  # decord — fastest
        import decord  # type: ignore
        vr = decord.VideoReader(io.BytesIO(raw))
        idx = _uniform_indices(len(vr), n_frames)
        arr = vr.get_batch(idx).asnumpy()                  # (n, H, W, 3)
        frames = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    except Exception:  # decord missing or can't decode — try next
        pass

    if frames is None:
        try:  # imageio (pip install imageio imageio-ffmpeg)
            import imageio.v3 as iio  # type: ignore
            all_frames = [fr for fr in iio.imiter(io.BytesIO(raw), extension=".mp4")]
            idx = _uniform_indices(len(all_frames), n_frames)
            arr = torch.stack([torch.from_numpy(all_frames[i][..., :3]) for i in idx])
            frames = arr.permute(0, 3, 1, 2).float() / 255.0
        except Exception:  # imageio/backend missing or can't decode — try ffmpeg
            pass

    if frames is None:  # ffmpeg/ffprobe subprocess — no python deps at all
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", tmp_path],
                capture_output=True, text=True, check=True)
            dur = max(1e-6, float(probe.stdout.strip()))
            fps = n_frames / dur
            out = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", tmp_path, "-vf", f"fps={fps}",
                 "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                capture_output=True, check=True)
            h = w = None
            # probe frame size
            size_probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0", tmp_path],
                capture_output=True, text=True, check=True)
            w, h = map(int, size_probe.stdout.strip().split(",")[:2])
            n = len(out.stdout) // (h * w * 3)
            if n == 0:
                raise RuntimeError("ffmpeg decoded 0 frames")
            arr = torch.frombuffer(bytearray(out.stdout[:n * h * w * 3]),
                                   dtype=torch.uint8)
            frames = arr.view(n, h, w, 3).permute(0, 3, 1, 2).float() / 255.0
            idx = _uniform_indices(n, n_frames)
            frames = frames[idx]
        finally:
            os.unlink(tmp_path)

    # resize every frame to the model's image_size
    frames = torch.nn.functional.interpolate(
        frames, size=(image_size, image_size), mode="bilinear", align_corners=False)
    if frames.shape[0] < n_frames:                        # short clip: repeat last
        pad = frames[-1:].expand(n_frames - frames.shape[0], -1, -1, -1)
        frames = torch.cat([frames, pad], dim=0)
    return frames[:n_frames]


# --------------------------------------------------------------------------- #
# token splicing — mirrors rbuild.vision.VisionTower exactly
# --------------------------------------------------------------------------- #

def vision_tokens_per_sample(cfg, n_frames: int) -> int:
    """
    How many <image> placeholders a sample with n_frames media frames needs.
    Mirrors VisionTower: grid^2 (+cls in ViT mode) per frame, plus
    vawu_tokens when VaWU is built (vawu and video enabled).
    """
    v = cfg.vision
    tpi = (v.image_size // v.patch_size) ** 2
    if v.mode == "vit" and v.use_cls_token:
        tpi += 1
    whole = v.vawu_tokens if (v.vawu and v.video) else 0
    return n_frames * tpi + whole


def _encode_text(text: str, vocab_size: int) -> List[int]:
    """Byte-level UTF-8 — matches the local .txt path and the rrun backend."""
    if vocab_size < 256:
        return [b % vocab_size for b in text.encode("utf-8")]
    return list(text.encode("utf-8"))


def _out_runs(cfg, out_keys) -> List[int]:
    """Placeholder ids a sample's out-targets need, appended after the text."""
    o = cfg.outgen
    ids: List[int] = []
    if not (o.enabled and out_keys):
        return ids
    if o.image and "image" in out_keys:
        ids += [o.image_token_id] * cfg.n_image_out_tokens()
    if o.video and "video" in out_keys:
        ids += [o.video_token_id] * cfg.n_video_out_tokens()
    if o.tts and "audio" in out_keys:
        ids += [o.audio_token_id] * o.audio_tokens
    return ids


def _build_sample(cfg, text: str, n_media: int, seq: int,
                  out_keys: Optional[set] = None):
    """
    (input_ids, target_ids) for one sample: an <image>-placeholder run of the
    exact length VisionTower will produce, then the byte-level text, then the
    out-placeholder runs for whichever generative heads this sample targets
    (matching generate_image/video/audio, which append the same runs after
    the prompt). Padded to seq+1 with 0 / -100; truncation never cuts any
    placeholder run.
    """
    ids: List[int] = []
    if n_media > 0:
        ids += [cfg.vision.image_token_id] * vision_tokens_per_sample(cfg, n_media)
    out_ids = _out_runs(cfg, out_keys)
    n_reserved = len(ids) + len(out_ids)
    if n_reserved > seq + 1:
        raise ValueError(
            f"placeholders alone need {n_reserved} tokens but seq={seq} — "
            f"increase seq (or max_seq_len), shrink image_size / n_frames / out sizes")
    ids += _encode_text(text, cfg.model.vocab_size)[:seq + 1 - n_reserved]
    ids += out_ids
    x = torch.zeros(seq + 1, dtype=torch.long)
    y = torch.full((seq + 1,), -100, dtype=torch.long)
    n = len(ids)
    x[:n] = torch.tensor(ids, dtype=torch.long)
    y[:n - 1] = torch.tensor(ids[1:], dtype=torch.long)
    return x[:-1], y[:-1]


# --------------------------------------------------------------------------- #
# batch assembly
# --------------------------------------------------------------------------- #

def _collate(cfg, samples, seq: int, warned: dict):
    """
    samples: list of (text, frames_or_None, out_dict). Frames are padded to
    the batch's max frame count (last-frame repeat); text-only samples in a
    mixed batch get black frames — standard mixed-modality padding, their
    loss is unaffected (vision positions never enter the text loss).
    out_dict maps "image"/"video"/"audio" -> target media tensor; samples
    missing a head's target get zero-filled targets and NO placeholder run,
    so the head's loss skips them. Returns (x, y), (x, y, images), or
    (x, y, images, out_targets) depending on what the config needs.
    """
    blind = not cfg.vision.enabled
    has_media = any(f is not None for _, f, _ in samples) and not blind
    n_batch = max((f.shape[0] for _, f, _ in samples if f is not None), default=0) \
        if has_media else 0
    xs, ys, vids = [], [], []
    out_lists: Dict[str, list] = {}
    for text, frames, outs in samples:
        if has_media:
            if frames is None:
                frames = torch.zeros(n_batch, 3, cfg.vision.image_size,
                                     cfg.vision.image_size)
            elif frames.shape[0] < n_batch:
                pad = frames[-1:].expand(n_batch - frames.shape[0], -1, -1, -1)
                frames = torch.cat([frames, pad], dim=0)
            x, y = _build_sample(cfg, text, n_batch, seq,
                                 set(outs) if outs else None)
            vids.append(frames)
        else:
            if frames is not None and not warned.get("blind_media"):
                print("  [data] blind model: media links in the manifest are "
                      "ignored — text only (enable vision to train on them)")
                warned["blind_media"] = True
            x, y = _build_sample(cfg, text, 0, seq,
                                 set(outs) if outs else None)
        xs.append(x)
        ys.append(y)
        for key, t in (outs or {}).items():
            out_lists.setdefault(key, []).append(t)
    x = torch.stack(xs)
    y = torch.stack(ys)
    batch_out: Dict[str, torch.Tensor] = {}
    if cfg.outgen.enabled and out_lists:
        shapes = {"image": (3, cfg.outgen.image_size, cfg.outgen.image_size),
                  "video": (cfg.outgen.video_frames, 3,
                            cfg.outgen.image_size, cfg.outgen.image_size),
                  "audio": (cfg.out_audio_len(),)}
        B = len(samples)
        for key, lst in out_lists.items():
            t = torch.zeros(B, *shapes[key])
            idx = [i for i, (_, _, outs) in enumerate(samples)
                   if outs and key in outs]
            for i, v in zip(idx, lst):
                t[i] = v
            batch_out[key] = t
    if has_media and batch_out:
        return x, y, torch.stack(vids), batch_out
    if has_media:
        return x, y, torch.stack(vids)
    if batch_out:
        return x, y, None, batch_out
    return x, y


# --------------------------------------------------------------------------- #
# universal JSON manifest
# --------------------------------------------------------------------------- #

def load_manifest(path: str) -> List[Dict[str, Any]]:
    """Parse a .json (list of objects) or .jsonl manifest."""
    with open(path) as f:
        raw = f.read().strip()
    if raw.startswith("["):
        entries = json.loads(raw)
    else:  # jsonl
        entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"manifest {path} is empty")
    return entries


def manifest_batches(path: str, cfg, batch: Optional[int] = None,
                     seq: Optional[int] = None, n_frames: Optional[int] = None,
                     shuffle: bool = True) -> Iterable:
    """
    Batch generator over a JSON/JSONL manifest — the manual insert that works
    with every model type. Entries: {"text": ..., "images": [...], "video": ...}.
    v3.1 outgen entries add generation targets: "image_out": <pic>,
    "video_out": <clip>, "audio_out": <sound> — the model learns to PRODUCE
    that media after the text (same run generate_image/video/audio uses).
    Yields (x, y) for blind text configs, (x, y, images) for vision
    configs, (x, y, images, out_targets) when outgen targets are present.
    Media that fails to load is skipped with a warning.
    """
    batch = batch or cfg.train.batch_size
    seq = seq or cfg.model.max_seq_len
    n_frames = min(n_frames or 8, cfg.vision.max_video_frames) \
        if cfg.vision.enabled else 0
    entries = load_manifest(path)
    warned: Dict[str, bool] = {}
    _SKIP = object()

    def frames_for(entry):
        outs = None
        if cfg.outgen.enabled:
            outs = {}
            try:
                o = cfg.outgen
                if o.image and entry.get("image_out"):
                    outs["image"] = load_image(entry["image_out"], o.image_size)
                if o.video and entry.get("video_out"):
                    outs["video"] = load_video(entry["video_out"],
                                               o.video_frames, o.image_size)
                if o.tts and entry.get("audio_out"):
                    outs["audio"] = load_audio(entry["audio_out"],
                                               cfg.out_audio_len(), o.sample_rate)
            except Exception as e:
                if not warned.get("out_load_fail"):
                    print(f"  [data] skipping entries whose out-target media "
                          f"fails to load ({e})")
                    warned["out_load_fail"] = True
                return _SKIP
        if not cfg.vision.enabled:
            if (entry.get("images") or entry.get("video")) and not warned.get("blind_media"):
                print("  [data] blind model: media links in the manifest are "
                      "ignored — text only (enable vision to train on them)")
                warned["blind_media"] = True
            return None, outs
        try:
            if entry.get("video"):
                return load_video(entry["video"], n_frames,
                                  cfg.vision.image_size), outs
            if entry.get("images"):
                imgs = [load_image(s, cfg.vision.image_size) for s in entry["images"]]
                return torch.stack(imgs), outs
        except Exception as e:
            if not warned.get("load_fail"):
                print(f"  [data] skipping entries whose media fails to load ({e})")
                warned["load_fail"] = True
            return _SKIP
        return None, outs

    while True:
        order = list(range(len(entries)))
        if shuffle:
            random.shuffle(order)
        samples = []
        for i in order:
            entry = entries[i]
            text = entry.get("text", "")
            got = frames_for(entry)
            if got is _SKIP:
                continue
            fr, outs = got
            samples.append((text, fr, outs))
            if len(samples) == batch:
                yield _collate(cfg, samples, seq, warned)
                samples = []
        if samples:  # partial tail batch
            yield _collate(cfg, samples, seq, warned)


# --------------------------------------------------------------------------- #
# HuggingFace streaming — image and video, column auto-detection
# --------------------------------------------------------------------------- #

def _detect_columns(row: Dict[str, Any]):
    text_col = image_col = video_col = None
    for k, v in row.items():
        if isinstance(v, str) and text_col is None and not _looks_like_media_path(v):
            text_col = k
        if _is_pil(v) and image_col is None:
            image_col = k
        if video_col is None and _looks_like_video(v, k):
            video_col = k
    return text_col, image_col, video_col


def _is_pil(v) -> bool:
    return hasattr(v, "size") and hasattr(v, "convert") and not isinstance(v, str)


def _looks_like_media_path(v: str) -> bool:
    if not isinstance(v, str):
        return False
    return v.lower().rsplit(".", 1)[-1] in \
        ("jpg", "jpeg", "png", "webp", "mp4", "webm", "mov", "avi", "mkv")


def _looks_like_video(v, key: str) -> bool:
    if isinstance(v, str) and v.lower().rsplit(".", 1)[-1] in \
            ("mp4", "webm", "mov", "avi", "mkv"):
        return True
    if isinstance(v, dict) and ("bytes" in v or "path" in v) and \
            any(t in key.lower() for t in ("video", "mp4", "clip")):
        return True
    if isinstance(v, (bytes, bytearray)) and "video" in key.lower():
        return True
    return False


def _stream(dataset_name: str, split: str):
    from datasets import load_dataset
    return load_dataset(dataset_name, split=split, streaming=True)


def hf_image_batches(dataset_name: str, cfg, batch: Optional[int] = None,
                     seq: Optional[int] = None, split: str = "train",
                     text_column: Optional[str] = None,
                     image_column: Optional[str] = None) -> Iterable:
    """
    No-fuss image training from any HF dataset with an image column and a
    caption/text column. Columns auto-detect from the first row; override
    with text_column=/image_column= if needed. Yields (x, y, images).
    """
    assert cfg.vision.enabled, "hf_image_batches needs cfg.vision.enabled=True"
    batch = batch or cfg.train.batch_size
    seq = seq or cfg.model.max_seq_len
    ds = _stream(dataset_name, split)
    warned: Dict[str, bool] = {}
    t_col = i_col = None
    samples = []
    for row in ds:
        if t_col is None and i_col is None:
            t_col, i_col, _ = _detect_columns(row)
            t_col = text_column or t_col
            i_col = image_column or i_col
            if not i_col:
                raise ValueError(f"no image column found in {dataset_name} "
                                 f"(columns: {list(row)}); pass image_column=")
            if not t_col:
                raise ValueError(f"no text column found in {dataset_name} "
                                 f"(columns: {list(row)}); pass text_column=")
            print(f"  [data] {dataset_name}: image={i_col!r} text={t_col!r}")
        try:
            img = row[i_col]
            if _is_pil(img):                         # PIL -> tensor
                img = _pil_to_tensor(img, cfg.vision.image_size)
            else:                                    # path/URL/bytes -> tensor
                img = load_image(img, cfg.vision.image_size)
            samples.append((str(row[t_col]), img.unsqueeze(0), None))
        except Exception as e:
            if not warned.get("load_fail"):
                print(f"  [data] skipping rows whose image fails ({e})")
                warned["load_fail"] = True
            continue
        if len(samples) == batch:
            yield _collate(cfg, samples, seq, warned)
            samples = []
    if samples:
        yield _collate(cfg, samples, seq, warned)


def hf_video_batches(dataset_name: str, cfg, batch: Optional[int] = None,
                     seq: Optional[int] = None, split: str = "train",
                     n_frames: Optional[int] = None,
                     text_column: Optional[str] = None,
                     video_column: Optional[str] = None) -> Iterable:
    """
    No-fuss video training from any HF dataset with a video column (paths,
    URLs, or mp4 bytes) and a caption column. Columns auto-detect; override
    with text_column=/video_column=. Frames are sampled uniformly, resized,
    and spliced automatically. Yields (x, y, images).
    """
    assert cfg.vision.enabled, "hf_video_batches needs cfg.vision.enabled=True"
    batch = batch or cfg.train.batch_size
    seq = seq or cfg.model.max_seq_len
    n_frames = min(n_frames or 8, cfg.vision.max_video_frames)
    ds = _stream(dataset_name, split)
    warned: Dict[str, bool] = {}
    t_col = v_col = None
    samples = []
    for row in ds:
        if t_col is None and v_col is None:
            t_col, _, v_col = _detect_columns(row)
            t_col = text_column or t_col
            v_col = video_column or v_col
            if not v_col:
                raise ValueError(f"no video column found in {dataset_name} "
                                 f"(columns: {list(row)}); pass video_column=")
            if not t_col:
                raise ValueError(f"no text column found in {dataset_name} "
                                 f"(columns: {list(row)}); pass text_column=")
            print(f"  [data] {dataset_name}: video={v_col!r} text={t_col!r} "
                  f"({n_frames} frames/clip)")
        try:
            frames = load_video(row[v_col], n_frames, cfg.vision.image_size)
            samples.append((str(row[t_col]), frames, None))
        except Exception as e:
            if not warned.get("load_fail"):
                print(f"  [data] skipping rows whose video fails ({e})")
                warned["load_fail"] = True
            continue
        if len(samples) == batch:
            yield _collate(cfg, samples, seq, warned)
            samples = []
    if samples:
        yield _collate(cfg, samples, seq, warned)
