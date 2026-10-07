"""R-Run smoke test — proves the swap contract end to end on any machine
(mock backend, no GPU / no weights needed; rbuild backend on tiny native
checkpoints):

  1. serve model A            -> resident + full KV cache
  2. generate from A          -> works
  3. swap A -> B              -> A fully wiped, B resident, FULL new KV cache
  4. generate from B          -> works, server/engine never restarted
  5. swap B -> C              -> repeatable
  6. HTTP layer               -> /admin/swap + /v1/chat/completions consistent
  7. rbuild backend           -> native checkpoints serve + swap, KV sized
                                 from R-Build geometry, counter-verified

Run:  python tests/smoke_test.py        (or: pytest tests/smoke_test.py)
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rrun.backends import MockBackend                      # noqa: E402
from rrun.engine import RRunEngine                          # noqa: E402

MODEL_A = "GeoThinkAI/R-build-20b-a3.4b"
MODEL_B = "Qwen/Qwen3-32B"
MODEL_C = "GeoThinkAI/R-build-90b-a8.9b"

passed = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        raise AssertionError(name)
    passed.append(name)


def test_engine_swap():
    print("== engine swap contract ==")
    eng = RRunEngine(MockBackend(simulated_vram_gb=18.5),
                     max_context=131072, max_seqs=64)

    # 1. cold serve A
    rep_a = eng.serve(MODEL_A)
    check("serve A resident", eng.backend.loaded)
    check("A full KV cache", eng.kv.status()["full"] is True,
          f"{eng.kv.status()['gb']} GB @ {rep_a.max_context} ctx")

    # 2. inference on A
    out_a = eng.generate("hello from A")
    check("generate from A", MODEL_A in out_a, out_a[:60])

    # 3. swap A -> B
    wipes_before = eng.kv.wipe_count
    t0 = time.perf_counter()
    rep_b = eng.swap(MODEL_B)
    swap_s = time.perf_counter() - t0
    st = eng.status()
    check("swap wiped old cache", eng.kv.wipe_count == wipes_before + 1)
    check("B is resident", st["model"]["model_id"] == MODEL_B)
    check("A gone from cache", st["kv_cache"]["model_id"] == MODEL_B)
    check("B full KV cache", st["kv_cache"]["full"] is True,
          f"{st['kv_cache']['gb']} GB")
    check("swap was fast", swap_s < 2.0, f"{swap_s*1000:.0f} ms (mock)")

    # 4. inference on B, same engine object (no restart)
    out_b = eng.generate("hello from B")
    check("generate from B", MODEL_B in out_b)

    # 5. second swap B -> C, history accumulates
    eng.swap(MODEL_C)
    check("repeatable swap", eng.status()["model"]["model_id"] == MODEL_C)
    check("swap history", eng.status()["swaps"] == 2)


def test_http_layer():
    print("== http layer ==")
    from fastapi.testclient import TestClient
    from rrun.server import create_app

    eng = RRunEngine(MockBackend())
    eng.serve(MODEL_A)
    c = TestClient(create_app(eng))

    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "ping"}]})
    check("chat 200", r.status_code == 200)
    check("chat answered by A", MODEL_A in
          r.json()["choices"][0]["message"]["content"])

    r = c.post("/admin/swap", json={"model_id": MODEL_B})
    check("admin swap 200", r.status_code == 200)
    body = r.json()
    check("admin swap reports full KV", body["kv_cache_full"] is True,
          f"{body['kv_cache_gb']} GB")

    r = c.get("/v1/models")
    check("/v1/models shows B only",
          [m["id"] for m in r.json()["data"]] == [MODEL_B])

    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "ping again"}]})
    check("post-swap chat served by B", MODEL_B in
          r.json()["choices"][0]["message"]["content"])


def test_rbuild_backend():
    """Native R-Build checkpoints through the same serve/swap contract."""
    print("== rbuild backend (native checkpoints) ==")
    import tempfile
    import torch
    from rbuild import RBuildConfig, RBuildModel, Trainer
    from rrun.backends.rbuild_backend import RBuildBackend

    def make_ckpt(path: str, seed: int) -> RBuildConfig:
        cfg = RBuildConfig()
        cfg.model.max_seq_len = 64
        cfg.critic.max_loops = 2
        torch.manual_seed(seed)
        model = RBuildModel(cfg)
        actual = sum(p.numel() for p in model.parameters())
        assert actual == cfg.count_parameters()["total_params"]
        Trainer(model, cfg, device="cpu", log_fn=None).save_checkpoint(path)
        return cfg

    root = tempfile.mkdtemp(prefix="rrun_rbuild_")
    ckpt_a, ckpt_b = os.path.join(root, "ckpt_a"), os.path.join(root, "ckpt_b")
    cfg_a = make_ckpt(ckpt_a, 1)
    make_ckpt(ckpt_b, 2)

    eng = RRunEngine(RBuildBackend(max_context=64), max_context=64, max_seqs=4)

    # serve checkpoint A — KV cache sized from R-Build geometry
    rep_a = eng.serve(ckpt_a)
    expect_layers = (cfg_a.cache_loop.n_layers
                     + cfg_a.parallel.n_stages * cfg_a.parallel.n_branches)
    check("rbuild A resident", eng.backend.loaded)
    check("KV from R-Build geometry",
          rep_a.extra["n_layers"] == expect_layers
          and rep_a.extra["n_kv_heads"] == cfg_a.model.n_kv_heads
          and rep_a.extra["head_dim"] == cfg_a.model.resolved_head_dim(),
          f"{expect_layers} effective layers")
    check("counter-verified load",
          rep_a.extra["total_params"]
          == cfg_a.count_parameters()["total_params"])
    check("rbuild full KV cache", eng.kv.status()["full"] is True)

    # generate (byte-level path), watermark kwarg accepted
    out_a = eng.generate("hello from A", max_tokens=8, watermark=False)
    check("generate from rbuild A", isinstance(out_a, str) and len(out_a) > 0)

    # swap checkpoint A -> B, same engine, full new KV cache
    wipes_before = eng.kv.wipe_count
    eng.swap(ckpt_b)
    st = eng.status()
    check("rbuild swap wiped old cache",
          eng.kv.wipe_count == wipes_before + 1)
    check("rbuild B resident", st["model"]["model_id"] == ckpt_b)
    check("rbuild B full KV cache", st["kv_cache"]["full"] is True)
    out_b = eng.generate("hello from B", max_tokens=8)
    check("generate from rbuild B", isinstance(out_b, str) and len(out_b) > 0)

    # HTTP layer over the native backend
    from fastapi.testclient import TestClient
    from rrun.server import create_app
    c = TestClient(create_app(eng))
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4})
    check("rbuild chat 200", r.status_code == 200)
    r = c.post("/admin/swap", json={"model_id": ckpt_a})
    check("rbuild admin swap 200", r.status_code == 200)
    check("rbuild admin swap reports full KV",
          r.json()["kv_cache_full"] is True)


def test_vl_manifest_data():
    """v3.1 data layer: manual JSON manifest trains vision + blind models."""
    print("== vl manifest data layer ==")
    import json
    import shutil
    import subprocess
    import tempfile
    import torch
    from PIL import Image
    from rbuild import (RBuildConfig, RBuildModel, Trainer, manifest_batches,
                        vision_tokens_per_sample)

    root = tempfile.mkdtemp(prefix="rbuild_data_")
    Image.new("RGB", (32, 32), (200, 30, 30)).save(os.path.join(root, "a.png"))
    Image.new("RGB", (32, 32), (30, 30, 200)).save(os.path.join(root, "b.png"))
    clip = os.path.join(root, "clip.mp4")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        "testsrc=duration=1:size=48x48:rate=8",
                        "-pix_fmt", "yuv420p", clip],
                       capture_output=True, check=True)
    entries = [
        {"text": "red square", "images": [os.path.join(root, "a.png")]},
        {"text": "blue square", "images": [os.path.join(root, "b.png"),
                                           os.path.join(root, "a.png")]},
        {"text": "no media here"},
    ]
    if os.path.exists(clip):
        entries.append({"text": "a test clip", "video": clip})
    mpath = os.path.join(root, "manifest.jsonl")
    with open(mpath, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    cfg = RBuildConfig()
    cfg.model.max_seq_len = 96
    cfg.critic.max_loops = 2
    cfg.vision.enabled = True
    cfg.vision.image_size = 48
    cfg.vision.patch_size = 8
    cfg.vision.image_token_id = 255
    cfg.vision.vit_heads = 8
    cfg.validate()
    cfg.train.max_steps = 2
    cfg.train.batch_size = 4

    model = RBuildModel(cfg)
    check("vl counter-verified",
          sum(p.numel() for p in model.parameters())
          == cfg.count_parameters()["total_params"])

    batches = manifest_batches(mpath, cfg, n_frames=2, shuffle=False)
    x, y, images = next(iter(batches))
    check("manifest yields (x, y, images)",
          x.shape == (4, 96) and images.shape[1] == 2,
          f"images {tuple(images.shape)}")
    K = vision_tokens_per_sample(cfg, 2)
    check("placeholder run == K",
          bool(((x == 255).sum(1) == K).all()), f"K={K}")

    hist = Trainer(model, cfg, device="cpu", log_fn=None).fit(
        manifest_batches(mpath, cfg, n_frames=2, shuffle=False), max_steps=2)
    check("vl manifest training", len(hist["loss"]) == 2,
          f"loss {hist['loss'][-1]:.3f}")
    # Trainer zeroes grads after each step — verify grad flow on a fresh pass
    model.zero_grad(set_to_none=True)
    model(x, targets=y, images=images)[1].backward()
    vgrad = any(p.grad is not None and p.grad.abs().sum() > 0
                for p in model.vision.parameters())
    check("vision tower receives gradients", vgrad)

    # blind model: same manifest, media ignored, (x, y) batches
    cfg_b = RBuildConfig()
    cfg_b.model.max_seq_len = 64
    cfg_b.critic.max_loops = 2
    cfg_b.train.max_steps = 1
    model_b = RBuildModel(cfg_b)
    batch_b = next(iter(manifest_batches(mpath, cfg_b, batch=3, seq=64,
                                         shuffle=False)))
    check("blind manifest yields (x, y)", len(batch_b) == 2)
    hist_b = Trainer(model_b, cfg_b, device="cpu", log_fn=None).fit(
        manifest_batches(mpath, cfg_b, batch=3, seq=64), max_steps=1)
    check("blind manifest training", len(hist_b["loss"]) == 1)


def test_outgen_heads():
    """v3.1 generative OUTPUT heads: TTS / image OUT / video OUT (renderer MoE)."""
    print("== outgen heads (tts / image out / video out) ==")
    import json
    import math
    import struct
    import tempfile
    import wave
    import torch
    from PIL import Image
    from rbuild import RBuildConfig, RBuildModel, Trainer, manifest_batches

    root = tempfile.mkdtemp(prefix="rbuild_outgen_")
    Image.new("RGB", (48, 48), (220, 40, 40)).save(os.path.join(root, "red.png"))
    wav_path = os.path.join(root, "tone.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = b"".join(struct.pack("<h", int(12000 * math.sin(i / 10)))
                          for i in range(16000))
        w.writeframes(frames)
    entries = [
        {"text": "paint red", "image_out": os.path.join(root, "red.png")},
        {"text": "play a tone", "audio_out": wav_path},
        {"text": "plain text"},
    ]
    mpath = os.path.join(root, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(entries, f)

    cfg = RBuildConfig()
    cfg.model.max_seq_len = 192
    cfg.critic.max_loops = 2
    o = cfg.outgen
    o.enabled = True
    o.image = o.video = o.tts = True
    o.image_token_id, o.video_token_id, o.audio_token_id = 250, 251, 252
    o.video_frames, o.audio_tokens = 2, 8
    cfg.validate()

    model = RBuildModel(cfg)
    check("outgen counter-verified",
          sum(p.numel() for p in model.parameters())
          == cfg.count_parameters()["total_params"],
          f"{cfg.count_parameters()['outgen_params']:,} outgen params")

    batch = next(iter(manifest_batches(mpath, cfg, batch=3, seq=192,
                                       shuffle=False)))
    check("manifest yields (x, y, images, out_targets)", len(batch) == 4)
    x, y, images, out_targets = batch
    check("out targets present",
          "image" in out_targets and "audio" in out_targets,
          str({k: tuple(v.shape) for k, v in out_targets.items()}))

    hist = Trainer(model, cfg, device="cpu", log_fn=None).fit(
        manifest_batches(mpath, cfg, batch=3, seq=192, shuffle=False), max_steps=2)
    check("outgen manifest training", len(hist["loss"]) == 2,
          f"loss {hist['loss'][-1]:.3f}")
    model.zero_grad(set_to_none=True)
    model(x, targets=y, images=images, out_targets=out_targets)[1].backward()
    for name, head in (("image", model.outgen.image_head),
                       ("tts", model.outgen.tts_head)):
        g = any(p.grad is not None and p.grad.abs().sum() > 0
                for p in head.parameters())
        check(f"{name} head receives gradients", g)

    model.eval()
    prompt = torch.tensor(list(b"say hi"), dtype=torch.long).unsqueeze(0)
    img = model.generate_image(prompt)
    wav = model.generate_audio(prompt)
    vid = model.generate_video(prompt)
    check("generate_image shape/range",
          img.shape == (1, 3, 48, 48)
          and 0 <= float(img.min()) <= float(img.max()) <= 1)
    check("generate_audio shape/range",
          wav.shape == (1, cfg.out_audio_len())
          and -1 <= float(wav.min()) <= float(wav.max()) <= 1)
    check("generate_video shape/range",
          vid.shape == (1, 2, 3, 48, 48)
          and 0 <= float(vid.min()) <= float(vid.max()) <= 1)

    # checkpoint roundtrip keeps the heads
    ckpt = os.path.join(root, "ckpt")
    Trainer(model, cfg, device="cpu", log_fn=None).save_checkpoint(ckpt)
    reloaded = Trainer.load_checkpoint(ckpt)
    check("outgen checkpoint roundtrip",
          reloaded.outgen is not None
          and reloaded.generate_image(prompt).shape == img.shape)


def test_routgen_model():
    """R-OutGen: media-native model on the EXACT R-Build text trunk."""
    print("== r-outgen (media-native, exact text trunk) ==")
    import json
    import math
    import struct
    import tempfile
    import wave
    import torch
    from PIL import Image
    from rbuild import (RBuildConfig, RBuildModel, ROutGenModel, Trainer,
                        manifest_batches, count_routgen_parameters)
    from rbuild.model import CacheLoopLine, ParallelBundleStage

    root = tempfile.mkdtemp(prefix="routgen_")
    Image.new("RGB", (48, 48), (40, 40, 220)).save(os.path.join(root, "blue.png"))
    wav_path = os.path.join(root, "tone.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = b"".join(struct.pack("<h", int(10000 * math.sin(i / 8)))
                          for i in range(16000))
        w.writeframes(frames)
    mpath = os.path.join(root, "manifest.json")
    with open(mpath, "w") as f:
        json.dump([{"text": "paint blue", "image_out": os.path.join(root, "blue.png")},
                   {"text": "play tone", "audio_out": wav_path}], f)

    cfg = RBuildConfig()
    cfg.model.max_seq_len = 192
    cfg.critic.max_loops = 2
    o = cfg.outgen
    o.enabled = True
    o.image = o.video = o.tts = True
    o.image_token_id, o.video_token_id, o.audio_token_id = 250, 251, 252
    o.video_frames, o.audio_tokens = 2, 8

    model = ROutGenModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    counted = count_routgen_parameters(cfg)["total_params"]
    check("routgen counter-verified", n == counted, f"{n:,} params")

    text_model = RBuildModel(cfg)
    check("routgen trunk IS the text architecture",
          isinstance(model.cache_loop, CacheLoopLine)
          and all(isinstance(s, ParallelBundleStage) for s in model.stages)
          and model.cache_loop.critics is not None
          and all(s.critics is not None for s in model.stages)
          and not hasattr(model, "lm_head"),
          "same CacheLoopLine + ParallelBundleStage + critic panels, no lm_head")
    check("routgen drops only text-output params",
          n == sum(p.numel() for p in text_model.parameters())
               - sum(p.numel() for p in text_model.noting_experts.parameters()),
          "trunk + memory + outgen identical to the text model (tied lm_head = 0)")
    del text_model

    hist = Trainer(model, cfg, device="cpu", log_fn=None).fit(
        manifest_batches(mpath, cfg, batch=2, seq=192, shuffle=False),
        max_steps=2)
    check("routgen trains via stock Trainer", len(hist["loss"]) == 2,
          f"loss {hist['loss'][-1]:.3f}")

    model.zero_grad(set_to_none=True)
    x, y, _, out_targets = next(iter(manifest_batches(
        mpath, cfg, batch=2, seq=192, shuffle=False)))
    _, loss = model(x, targets=y, out_targets=out_targets)
    loss.backward()
    g_head = any(p.grad is not None and p.grad.abs().sum() > 0
                 for p in model.outgen.image_head.parameters())
    g_trunk = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.cache_loop.parameters())
    check("routgen grads reach renderer MoE + trunk", g_head and g_trunk)

    model.eval()
    prompt = torch.tensor(list(b"make art"), dtype=torch.long).unsqueeze(0)
    img = model.generate_image(prompt)
    wav = model.generate_audio(prompt)
    vid = model.generate_video(prompt)
    check("routgen generate_image", img.shape == (1, 3, 48, 48)
          and 0 <= float(img.min()) <= float(img.max()) <= 1)
    check("routgen generate_audio", wav.shape == (1, cfg.out_audio_len())
          and -1 <= float(wav.min()) <= float(wav.max()) <= 1)
    check("routgen generate_video", vid.shape == (1, 2, 3, 48, 48)
          and 0 <= float(vid.min()) <= float(vid.max()) <= 1)

    ckpt = os.path.join(root, "ckpt")
    model.save_checkpoint(ckpt)
    reloaded = ROutGenModel.load_checkpoint(ckpt)
    check("routgen checkpoint roundtrip",
          isinstance(reloaded, ROutGenModel)
          and reloaded.generate_image(prompt).shape == img.shape)


def test_thinking_effort():
    """v3.2: mode effort (float) + reasoning flag + tag-selected effort."""
    print("== thinking effort (reasoning flag + selectable effort) ==")
    import torch
    from rbuild import RBuildConfig, RBuildModel, parse_effort_tag
    from rbuild.thinking import BUILTIN_MODES

    # every built-in mode defines effort (float) + reasoning (bool)
    check("modes define effort + reasoning",
          all(isinstance(m.get("effort"), float)
              and isinstance(m.get("reasoning"), bool)
              for m in BUILTIN_MODES.values()))
    check("instant mode = reasoning off",
          BUILTIN_MODES["instant"]["reasoning"] is False
          and BUILTIN_MODES["instant"]["effort"] == 0.0)

    # selectable effort is picked by tag, unlike the invisible mode effort
    clean, tag = parse_effort_tag("paint a cat {effort:'high'}")
    check("effort tag parsed", clean == "paint a cat" and tag == "high",
          f"{clean!r} / {tag!r}")
    clean2, tag2 = parse_effort_tag("no tag here")
    check("no tag -> None", tag2 is None and clean2 == "no tag here")

    cfg = RBuildConfig()
    cfg.model.max_seq_len = 32
    cfg.critic.max_loops = 8
    cfg.critic.min_loops = 1
    model = RBuildModel(cfg)
    check("effort adds zero parameters (counter still exact)",
          sum(p.numel() for p in model.parameters())
          == cfg.count_parameters()["total_params"])

    # the ladder is built in; default tag is medium
    check("selectable ladder built in",
          model.thinking_mode.selectable_efforts
          == {"low": 0.5, "medium": 1.0, "high": 2.0})
    check("default effort tag", model.thinking_mode.current_effort() == "medium")

    # effective effort = mode's built-in (invisible) effort x selectable
    model.thinking_mode.deep()
    model.thinking_mode.set_effort("high")
    check("effective = built-in x selectable",
          abs(model.thinking_mode.effective_effort() - 2.0 * 2.0) < 1e-9,
          f"deep(2.0) x high(2.0) = {model.thinking_mode.effective_effort():g}")

    # selectable effort ACTUALLY increases depth: with critics that never
    # fire, the loop runs to its cap — and the cap scales with the tag
    class _NeverSatisfied(torch.nn.Module):
        threshold = 0.99

        def forward(self, x):
            z = torch.zeros(x.shape[0], x.shape[1], device=x.device)
            return z, z.long()

    model.cache_loop.critics = _NeverSatisfied()
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    loops, thresholds, temps = {}, {}, {}
    for t in ("low", "medium", "high"):
        model.thinking_mode.set_effort(t)
        with torch.no_grad():
            model.hidden_states(x)
        loops[t] = model.cache_loop.last_halting["mean_loops"]
        thresholds[t] = model.cache_loop.critics.threshold
        temps[t] = model._runtime_sampling["temperature"]
    check("more effort = deeper extraction (actual loops)",
          loops["low"] < loops["medium"] < loops["high"], str(loops))
    check("more effort = stricter critic gate",
          thresholds["low"] < thresholds["medium"] < thresholds["high"],
          str({k: round(v, 3) for k, v in thresholds.items()}))
    check("more effort = sharper sampling",
          temps["high"] < temps["medium"] < temps["low"],
          str({k: round(v, 3) for k, v in temps.items()}))

    # reasoning=False = instant, no thinking: minimum loops, zero effort
    model.thinking_mode.instant()
    with torch.no_grad():
        model.hidden_states(x)
    check("reasoning off = instant (min loops, zero effort)",
          model.cache_loop.last_halting["mean_loops"] == cfg.critic.min_loops
          and model.thinking_mode.effective_effort() == 0.0)

    # per-call effort kwarg applies once and restores the previous tag
    model.thinking_mode.balanced()
    model.thinking_mode.set_effort("medium")
    model.generate(x, max_new_tokens=2, effort="high")
    check("generate(effort=...) is per-call",
          model.thinking_mode.current_effort() == "medium")

    # unknown tags are rejected loudly
    try:
        model.thinking_mode.set_effort("ludicrous")
        rejected = False
    except KeyError:
        rejected = True
    check("unknown effort tag rejected", rejected)

    # create() accepts the new knobs and still rejects unknown ones
    m = model.thinking_mode.create("exam", max_loops=10, effort=1.5,
                                   reasoning=True)
    check("custom mode carries effort + reasoning",
          m["effort"] == 1.5 and m["reasoning"] is True)
    try:
        model.thinking_mode.create("bogus", speed=2)
        rejected = False
    except ValueError:
        rejected = True
    check("unknown mode knob still rejected", rejected)


if __name__ == "__main__":
    test_engine_swap()
    test_http_layer()
    test_rbuild_backend()
    test_vl_manifest_data()
    test_outgen_heads()
    test_routgen_model()
    test_thinking_effort()
    print(f"\nSMOKE TEST PASSED — {len(passed)} checks green")
