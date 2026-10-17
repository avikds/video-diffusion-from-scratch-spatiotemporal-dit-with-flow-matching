"""
Video Diffusion from Scratch: Spatiotemporal DiT with Flow Matching scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Video diffusion from scratch: a spatiotemporal DiT trained with flow matching.

Story: clips of a square moving at constant velocity are generated on the fly, so
motion has a ground truth. A diffusion transformer over space-time patches, with
adaLN-Zero conditioning and attention factorized into spatial and temporal passes,
is trained to predict the flow-matching velocity and sampled with an Euler ODE
solver under classifier-free guidance. The same model with the temporal attention
removed renders frames as well or better and gets the motion wrong: per-frame
quality and motion coherence are different axes, and only the second one needs
the frames to talk to each other.
"""
import torch


def fmt(ev):
    return f"direction acc {ev['direction_acc']:.3f} | jitter {ev['jitter']:.2f} px | clean-square frames {ev['shape']:.3f}"


def main() -> None:
    torch.set_num_threads(8)
    torch.manual_seed(0)
    cfg = dict(seed=0, train_steps=1200, n_eval=128, euler_steps=16, cfg_scale=2.0)

    print("1. The data")
    clips, labels = make_batch(torch.Generator().manual_seed(0), 4)
    print(f"   clips {tuple(clips.shape)}: {T} frames of {H}x{W}, a {K}x{K} square moving one pixel per frame")
    print(f"   labels {labels.tolist()} -> directions {[DIRECTIONS[int(l)] for l in labels]}")
    print(f"   tracker on clip 0: (dx, dy, jitter) = {tuple(round(v, 3) for v in clip_velocity(clips[0, 0]))}")

    print("\n2. The model")
    S = (H // 4) * (W // 4)
    full, fact = attention_cost(T, S)
    print(f"   {T * S} space-time tokens of {4 * 4} pixels; full attention scores {full} pairs, factorized {fact} ({full / fact:.1f}x cheaper)")
    print(f"   at 16 frames x 1024 patches the saving is {attention_cost(16, 1024)[0] / attention_cost(16, 1024)[1]:.0f}x")
    torch.manual_seed(0)
    probe = VideoDiT()
    print(f"   VideoDiT d=64, depth=2, heads=4: {sum(p.numel() for p in probe.parameters())} parameters; adaLN-Zero makes it the identity at init")

    rep = video_experiment(**cfg)
    f, s = rep["full"], rep["spatial_only"]

    print(f"\n3. Training ({cfg['train_steps']} steps, batch 64, logit-normal times, EMA 0.99)")
    print(f"   full model:    loss {f['loss_first']:.3f} -> {f['loss_last']:.3f} ({f['params']} params)")
    print(f"   spatial-only:  loss {s['loss_first']:.3f} -> {s['loss_last']:.3f} ({s['params']} params)")
    print("   (first number = mean of the first 20 steps; step 0 itself is the target variance, 2, because the")
    print("   zero-initialised model predicts zero velocity.) The spatial-only model plateaus higher: without")
    print("   the other frames it cannot resolve where the square is.")

    print(f"\n4. Conditional samples ({cfg['n_eval']} clips, {cfg['euler_steps']} Euler steps)")
    print(f"   full,  cfg {cfg['cfg_scale']}: {fmt(f['cond'])}")
    print(f"   full,  no cfg: {fmt(f['plain'])}")
    print(f"   spatial-only, cfg {cfg['cfg_scale']}: {fmt(s['cond'])}")
    print(f"   spatial-only, no cfg: {fmt(s['plain'])}")
    print("   Frame quality is similar (the spatial-only model spends all its capacity on frames).")
    print("   Motion is not: the spatial-only model places each frame's square independently, so its")
    print("   jitter is several pixels per frame. Its direction accuracy is still above chance only")
    print("   because the data's start-corner rule correlates position with direction.")

    print("\n5. Image-to-video (first frame clamped to a real frame)")
    print(f"   full:          {fmt(f['i2v'])}")
    print(f"   spatial-only:  {fmt(s['i2v'])}")
    print("   Read the jitter: the full model continues the given frame with a steady velocity, while the")
    print("   spatial-only model cannot see frame 0 while denoising the others and jumps around as before.")

    print("\n6. Distribution distance (Frechet distance on fixed 3-D conv features)")
    print(f"   real vs real: {rep['fd_floor']:.3f} | full: {f['fd']:.3f} | spatial-only: {s['fd']:.3f}")
    print("   A 3-D feature extractor sees motion, so frames that look right but move wrong push the")
    print("   distribution away from the data even when every per-frame metric is fine.")

    verdict = (f["cond"]["jitter"] < s["cond"]["jitter"] and f["fd"] < s["fd"])
    print(f"\n7. Verdict: {'temporal attention bought motion coherence, not frame quality' if verdict else 'the ablation did not separate at this budget; rerun with more steps'}")
    print("   Shape score alone would call the two models equal. Jitter, image-to-video and the 3-D Frechet")
    print("   distance are the metrics that see time, and they are the ones a video model is judged on.")


if __name__ == "__main__":
    main()

