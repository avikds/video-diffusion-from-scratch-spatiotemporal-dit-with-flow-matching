# Video Diffusion from Scratch: Spatiotemporal DiT with Flow Matching

Video generators such as Sora, Veo and Seedance share one backbone: a diffusion transformer over space-time patches, trained with a flow-matching objective and sampled by integrating an ODE. This project builds that backbone in pure PyTorch on a dataset you generate yourself, clips of a shape moving at constant velocity, so that motion has a ground truth and every claim can be measured. You write the clip generator and a centroid tracker that recovers velocity from pixels, then the token pipeline: 3-D patchify and unpatchify, factorized space-time positional embeddings, sinusoidal timestep embeddings and a label embedder with a null class. The model is a DiT with adaLN-Zero conditioning whose attention is factorized into spatial attention within each frame and temporal attention across frames at each location, and you compute why that factorization is what makes long clips affordable. Training uses the flow-matching objective with logit-normal timestep sampling and an EMA of the weights; sampling uses an Euler ODE solver, classifier-free guidance on the motion label, and image-to-video generation by clamping the first frame. The evaluation separates per-frame quality from motion coherence with a direction-accuracy metric, a jitter metric and a Frechet distance on fixed spatiotemporal features, and the final experiment trains the same model with the temporal attention removed to show that frames can look right while the motion is wrong, which is the failure video models are judged on.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_clip
- [x] **2.** make_batch
- [x] **3.** clip_velocity
- [x] **4.** patchify
- [x] **5.** pos_embed
- [x] **6.** TimestepEmbedder
- [x] **7.** LabelEmbedder
- [x] **8.** Attention
- [x] **9.** attend_spatial
- [x] **10.** modulate
- [x] **11.** DiTBlock
- [x] **12.** FinalLayer
- [x] **13.** VideoDiT
- [x] **14.** sample_timesteps
- [x] **15.** flow_interpolant
- [x] **16.** flow_matching_loss
- [x] **17.** ema_update
- [x] **18.** train_video_dit
- [x] **19.** cfg_velocity
- [x] **20.** euler_sample
- [x] **21.** sample_from_first_frame
- [x] **22.** evaluate_samples
- [x] **23.** frechet_distance
- [x] **24.** train_variant
- [x] **25.** video_experiment

## Results

```
1. The data
   clips (4, 1, 6, 12, 12): 6 frames of 12x12, a 4x4 square moving one pixel per frame
   labels [4, 7, 5, 0] -> directions [(-1, 0), (1, -1), (-1, -1), (1, 0)]
   tracker on clip 0: (dx, dy, jitter) = (-1.0, 0.0, 0.0)

2. The model
   54 space-time tokens of 16 pixels; full attention scores 2916 pairs, factorized 810 (3.6x cheaper)
   at 16 frames x 1024 patches the saving is 16x
   VideoDiT d=64, depth=2, heads=4: 226960 parameters; adaLN-Zero makes it the identity at init

3. Training (1200 steps, batch 64, logit-normal times, EMA 0.99)
   full model:    loss 1.385 -> 0.063 (226960 params)
   spatial-only:  loss 1.375 -> 0.112 (193680 params)
   (first number = mean of the first 20 steps; step 0 itself is the target variance, 2, because the
   zero-initialised model predicts zero velocity.) The spatial-only model plateaus higher: without
   the other frames it cannot resolve where the square is.

4. Conditional samples (128 clips, 16 Euler steps)
   full,  cfg 2.0: direction acc 0.984 | jitter 0.62 px | clean-square frames 0.807
   full,  no cfg: direction acc 0.930 | jitter 0.86 px | clean-square frames 0.775
   spatial-only, cfg 2.0: direction acc 0.656 | jitter 3.11 px | clean-square frames 0.871
   spatial-only, no cfg: direction acc 0.547 | jitter 3.25 px | clean-square frames 0.849
   Frame quality is similar (the spatial-only model spends all its capacity on frames).
   Motion is not: the spatial-only model places each frame's square independently, so its
   jitter is several pixels per frame. Its direction accuracy is still above chance only
   because the data's start-corner rule correlates position with direction.

5. Image-to-video (first frame clamped to a real frame)
   full:          direction acc 0.734 | jitter 1.19 px | clean-square frames 0.832
   spatial-only:  direction acc 0.703 | jitter 3.25 px | clean-square frames 0.874
   Read the jitter: the full model continues the given frame with a steady velocity, while the
   spatial-only model cannot see frame 0 while denoising the others and jumps around as before.

6. Distribution distance (Frechet distance on fixed 3-D conv features)
   real vs real: 0.062 | full: 0.085 | spatial-only: 0.126
   A 3-D feature extractor sees motion, so frames that look right but move wrong push the
   distribution away from the data even when every per-frame metric is fine.

7. Verdict: temporal attention bought motion coherence, not frame quality
   Shape score alone would call the two models equal. Jitter, image-to-video and the 3-D Frechet
   distance are the metrics that see time, and they are the ones a video model is judged on.
```
