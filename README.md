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

---

Built on Deep-ML.
