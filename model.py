"""
Video Diffusion from Scratch: Spatiotemporal DiT with Flow Matching

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_clip
import torch

T, H, W, K = 6, 12, 12, 4

DIRECTIONS = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]


def make_clip(gen, direction):
    # Get the per-frame displacement for the specified direction.
    dx, dy = DIRECTIONS[direction]

    # Maximum displacement over the complete clip.
    span = T - 1

    # Determine valid starting x-coordinates so the square
    # remains inside the frame throughout all T frames.
    if dx == 1:
        x_lo, x_hi = 0, W - K - span
    elif dx == -1:
        x_lo, x_hi = span, W - K
    else:
        x_lo, x_hi = 0, W - K

    # Determine valid starting y-coordinates.
    if dy == 1:
        y_lo, y_hi = 0, H - K - span
    elif dy == -1:
        y_lo, y_hi = span, H - K
    else:
        y_lo, y_hi = 0, H - K

    # Sample x0 first, then y0, using the supplied generator.
    x0 = torch.randint(
        x_lo, x_hi + 1, (1,), generator=gen
    ).item()

    y0 = torch.randint(
        y_lo, y_hi + 1, (1,), generator=gen
    ).item()

    # Initialize the video with background pixel values.
    clip = torch.full((T, H, W), -1.0, dtype=torch.float32)

    # Draw the moving K x K square in every frame.
    for f in range(T):
        x = x0 + f * dx
        y = y0 + f * dy

        clip[f, y:y + K, x:x + K] = 1.0

    return clip

# Step 2 - make_batch
def make_batch(gen, n):
    # Sample all direction labels first, using the supplied generator.
    labels = torch.randint(0, 8, (n,), generator=gen)

    # Generate clips sequentially in the same order as their labels.
    clips = torch.stack(
        [make_clip(gen, int(label)) for label in labels],
        dim=0
    )

    # Add the channel dimension: (n, T, H, W) -> (n, 1, T, H, W).
    clips = clips.unsqueeze(1)

    return clips, labels

# Step 3 - clip_velocity
def clip_velocity(clip):
    # Accept either (T, H, W) or (1, T, H, W).
    if clip.ndim == 4:
        if clip.shape[0] != 1:
            raise ValueError("A 4D clip must have shape (1, T, H, W).")
        clip = clip.squeeze(0)
    elif clip.ndim != 3:
        raise ValueError("Clip must have shape (T, H, W) or (1, T, H, W).")

    # Compute the centroid (cx, cy) of the bright pixels in each frame.
    centroids = []

    for frame in clip:
        rows, cols = torch.where(frame > 0)

        if rows.numel() == 0:
            # This frame contains no bright pixels.
            centroids.append(None)
        else:
            cx = cols.float().mean().item()
            cy = rows.float().mean().item()
            centroids.append((cx, cy))

    # Compute displacements only between adjacent frames
    # when both frames have valid centroids.
    displacements = []

    for i in range(len(centroids) - 1):
        previous = centroids[i]
        current = centroids[i + 1]

        if previous is not None and current is not None:
            step_x = current[0] - previous[0]
            step_y = current[1] - previous[1]
            displacements.append((step_x, step_y))

    # No valid consecutive frame pair means no measurable motion.
    if not displacements:
        return (0.0, 0.0, 0.0)

    # Mean displacement across valid consecutive frame pairs.
    dx = sum(step[0] for step in displacements) / len(displacements)
    dy = sum(step[1] for step in displacements) / len(displacements)

    # Mean absolute deviation from the average velocity.
    jitter = sum(
        abs(step_x - dx) + abs(step_y - dy)
        for step_x, step_y in displacements
    ) / len(displacements)

    return (float(dx), float(dy), float(jitter))


def direction_index(dx, dy):
    # Find the direction with minimum squared Euclidean distance.
    # min() preserves the lowest index when distances are tied.
    return min(
        range(len(DIRECTIONS)),
        key=lambda i: (
            (DIRECTIONS[i][0] - dx) ** 2
            + (DIRECTIONS[i][1] - dy) ** 2
        )
    )

# Step 4 - patchify
def patchify(video, p):
    # Input shape: (B, C, T, H, W)
    B, C, T, H, W = video.shape

    # Number of patches along the height and width.
    Hp = H // p
    Wp = W // p

    # Split each frame into non-overlapping p x p patches.
    # Shape: (B, C, T, Hp, p, Wp, p)
    video = video.reshape(B, C, T, Hp, p, Wp, p)

    # Reorder dimensions to:
    # (B, T, Hp, Wp, C, p, p)
    # This preserves frame-major and row-major patch ordering.
    video = video.permute(0, 2, 3, 5, 1, 4, 6)

    # Flatten the patch grid into a token sequence and
    # flatten each patch's (C, p, p) values into one token.
    # Output shape: (B, T * Hp * Wp, C * p * p)
    tokens = video.reshape(B, T * Hp * Wp, C * p * p)

    return tokens


def unpatchify(tokens, T, H, W, p, C):
    # Input shape: (B, T * Hp * Wp, C * p * p)
    B = tokens.shape[0]

    # Number of patches along the height and width.
    Hp = H // p
    Wp = W // p

    # Restore the patch grid and individual patch dimensions.
    # Shape: (B, T, Hp, Wp, C, p, p)
    video = tokens.reshape(B, T, Hp, Wp, C, p, p)

    # Restore the original channel-first video layout:
    # (B, C, T, Hp, p, Wp, p)
    video = video.permute(0, 4, 1, 2, 5, 3, 6)

    # Merge the patch-grid and within-patch dimensions.
    # Output shape: (B, C, T, H, W)
    video = video.reshape(B, C, T, H, W)

    return video

# Step 5 - pos_embed
import math


def sincos_1d(positions, dim):
    # Number of sine/cosine frequency pairs.
    half = dim // 2

    # Compute frequencies: omega_k = 10000 ** (-k / half).
    k = torch.arange(
        half,
        dtype=torch.float32,
        device=positions.device
    )
    omega = torch.exp(-k * math.log(10000.0) / half)

    # Compute the position-frequency products using broadcasting.
    positions = positions.to(dtype=torch.float32).reshape(-1, 1)
    angles = positions * omega.unsqueeze(0)

    # Concatenate all sine features followed by all cosine features.
    return torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=1
    )


def pos_embed(T, Hp, Wp, d):
    # Generate position indices for frames, patch rows, and patch columns.
    frame_positions = torch.arange(T)
    row_positions = torch.arange(Hp)
    col_positions = torch.arange(Wp)

    # Encode each frame's temporal position.
    temporal = sincos_1d(frame_positions, d)  # (T, d)

    # Encode patch-row and patch-column positions independently.
    row_embed = sincos_1d(row_positions, d // 2)  # (Hp, d // 2)
    col_embed = sincos_1d(col_positions, d // 2)  # (Wp, d // 2)

    # Broadcast row and column embeddings into a spatial grid.
    # Shape: (Hp, Wp, d)
    spatial = torch.cat(
        [
            row_embed[:, None, :].expand(Hp, Wp, -1),
            col_embed[None, :, :].expand(Hp, Wp, -1),
        ],
        dim=-1
    )

    # Combine temporal and spatial embeddings through broadcasting.
    # Shape: (T, Hp, Wp, d)
    embeddings = temporal[:, None, None, :] + spatial[None, :, :, :]

    # Flatten in frame-major, row-major order to match patchify().
    return embeddings.reshape(T * Hp * Wp, d)

# Step 6 - TimestepEmbedder
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, d, freq_dim=64):
        super().__init__()

        # Store the dimension of the sinusoidal timestep features.
        self.freq_dim = freq_dim

        # Project sinusoidal features into the model dimension.
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

    def forward(self, t):
        # Scale flow times from [0, 1] to [0, 1000].
        # Use the sinusoidal embedding function from Step 5.
        t_freq = sincos_1d(t * 1000.0, self.freq_dim)

        # Return timestep conditioning vectors of shape (B, d).
        return self.mlp(t_freq)

# Step 7 - LabelEmbedder
class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, d, drop_prob):
        super().__init__()

        # Store the number of conditional classes and dropout probability.
        self.num_classes = num_classes
        self.drop_prob = drop_prob

        # Reserve an additional class index for unconditional generation.
        # The null class has index num_classes.
        self.table = nn.Embedding(num_classes + 1, d)

    def forward(self, y, gen=None):
        # Apply classifier-free label dropout only during training.
        if self.training and self.drop_prob > 0:
            # Independently decide whether to drop each label.
            drop_mask = (
                torch.rand(y.shape[0], generator=gen) < self.drop_prob
            )

            # Replace dropped labels with the null-class index.
            y = torch.where(
                drop_mask.to(y.device),
                torch.full_like(y, self.num_classes),
                y,
            )

        # Return label embeddings of shape (B, d).
        return self.table(y)

# Step 8 - Attention
class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()

        # Store the number of attention heads.
        self.heads = heads

        # Project each token into queries, keys, and values.
        self.qkv = nn.Linear(d, 3 * d)

        # Project the concatenated attention heads back to dimension d.
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        # Input shape: (B, N, d)
        B, N, d = x.shape

        # Compute queries, keys, and values.
        # Shape: (B, N, 3 * d)
        qkv = self.qkv(x)

        # Split into heads:
        # Shape: (B, N, 3, heads, head_dim)
        head_dim = d // self.heads
        qkv = qkv.reshape(B, N, 3, self.heads, head_dim)

        # Rearrange to (3, B, heads, N, head_dim).
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Extract queries, keys, and values.
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention.
        # Attention scores have shape (B, heads, N, N).
        scale = math.sqrt(head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        # Normalize attention weights over the key-token dimension.
        attn = attn.softmax(dim=-1)

        # Apply attention weights to the values.
        # Shape: (B, heads, N, head_dim)
        out = attn @ v

        # Merge the heads back into the embedding dimension.
        # Shape: (B, N, d)
        out = out.transpose(1, 2).reshape(B, N, d)

        # Final output projection.
        return self.proj(out)

# Step 9 - attend_spatial
def attend_spatial(attn, x, T, S):
    # Input shape: (B, T * S, d)
    B, _, d = x.shape

    # Group tokens by frame so each frame attends independently.
    # Shape: (B, T, S, d) -> (B * T, S, d)
    x = x.reshape(B, T, S, d)
    x = x.reshape(B * T, S, d)

    # Apply self-attention across the S spatial tokens of each frame.
    x = attn(x)

    # Restore frame-major token order.
    # Shape: (B * T, S, d) -> (B, T * S, d)
    x = x.reshape(B, T, S, d)
    x = x.reshape(B, T * S, d)

    return x


def attend_temporal(attn, x, T, S):
    # Input shape: (B, T * S, d)
    B, _, d = x.shape

    # Restore the space-time grid.
    # Shape: (B, T * S, d) -> (B, T, S, d)
    x = x.reshape(B, T, S, d)

    # Group tokens by spatial location across all frames.
    # Shape: (B, T, S, d) -> (B, S, T, d)
    x = x.permute(0, 2, 1, 3)

    # Treat each spatial location as an independent sequence.
    # Shape: (B, S, T, d) -> (B * S, T, d)
    x = x.reshape(B * S, T, d)

    # Apply self-attention across the T frames at each location.
    x = attn(x)

    # Restore the batch and spatial dimensions.
    # Shape: (B * S, T, d) -> (B, S, T, d)
    x = x.reshape(B, S, T, d)

    # Return to frame-major order.
    # Shape: (B, S, T, d) -> (B, T, S, d)
    x = x.permute(0, 2, 1, 3)

    # Flatten the space-time grid back into a token sequence.
    # Output shape: (B, T * S, d)
    x = x.reshape(B, T * S, d)

    return x


def attention_cost(T, S):
    # Full attention computes scores between every pair of T * S tokens.
    full = (T * S) ** 2

    # Factorized attention:
    # Spatial: T frames, each with S spatial tokens.
    # Temporal: S spatial locations, each spanning T frames.
    factorized = T * S * S + S * T * T

    return (full, factorized)

# Step 10 - modulate
import torch.nn.functional as F


def modulate(x, shift, scale):
    # x: (B, N, d)
    # shift, scale: (B, d)
    # Add a token dimension so conditioning broadcasts across all N tokens.
    shift = shift.unsqueeze(1)
    scale = scale.unsqueeze(1)

    # Apply adaptive scale and shift.
    return x * (1 + scale) + shift


class AdaLNModulation(nn.Module):
    def __init__(self, d, n):
        super().__init__()

        # Number of conditioning vectors to generate.
        self.n = n

        # Project conditioning into n separate d-dimensional vectors.
        self.lin = nn.Linear(d, n * d)

        # Zero-initialize the projection's weight and bias.
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, c):
        # Apply SiLU activation before the conditioning projection.
        out = self.lin(F.silu(c))

        # Split along the feature dimension into n chunks.
        return out.chunk(self.n, dim=-1)

# Step 11 - DiTBlock
class DiTBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4, temporal=True):
        super().__init__()

        # Store whether temporal attention is enabled.
        self.temporal = temporal

        # 1. Normalization before spatial attention.
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)

        # 2. Spatial self-attention within each frame.
        self.attn_s = Attention(d, heads)

        # 3. Normalization before temporal attention.
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)

        # 4. Temporal self-attention across frames, if enabled.
        self.attn_t = Attention(d, heads) if temporal else None

        # 5. Normalization before the feed-forward network.
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False)

        # 6. Feed-forward MLP.
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d),
            nn.GELU(),
            nn.Linear(mlp_ratio * d, d),
        )

        # 7. Generate nine adaLN-Zero conditioning vectors:
        # (shift1, scale1, gate1, shift2, scale2, gate2,
        #  shift3, scale3, gate3).
        self.ada = AdaLNModulation(d, 9)

    def forward(self, x, c, T, S):
        # x: (B, T * S, d)
        # c: (B, d)

        # Split the conditioning output into nine vectors.
        (
            shift1, scale1, gate1,
            shift2, scale2, gate2,
            shift3, scale3, gate3,
        ) = self.ada(c)

        # Spatial attention: each frame attends to its own spatial tokens.
        spatial_input = modulate(
            self.norm1(x), shift1, scale1
        )
        spatial_output = attend_spatial(
            self.attn_s, spatial_input, T, S
        )
        x = x + gate1[:, None, :] * spatial_output

        # Temporal attention: each spatial location attends across frames.
        if self.temporal:
            temporal_input = modulate(
                self.norm2(x), shift2, scale2
            )
            temporal_output = attend_temporal(
                self.attn_t, temporal_input, T, S
            )
            x = x + gate2[:, None, :] * temporal_output

        # Feed-forward MLP with adaptive normalization and a gated residual.
        mlp_input = modulate(
            self.norm3(x), shift3, scale3
        )
        mlp_output = self.mlp(mlp_input)
        x = x + gate3[:, None, :] * mlp_output

        return x

# Step 12 - FinalLayer
class FinalLayer(nn.Module):
    def __init__(self, d, out_dim):
        super().__init__()

        # Normalize tokens without learnable affine parameters.
        self.norm = nn.LayerNorm(d, elementwise_affine=False)

        # Generate shift and scale vectors from the conditioning.
        self.ada = AdaLNModulation(d, 2)

        # Project each token to its patch-pixel values.
        self.lin = nn.Linear(d, out_dim)

        # Zero-initialize the final projection's weight and bias.
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, c):
        # Generate the conditioning-dependent shift and scale.
        shift, scale = self.ada(c)

        # Apply normalization, adaptive modulation, and projection.
        return self.lin(modulate(self.norm(x), shift, scale))

# Step 13 - VideoDiT
class VideoDiT(nn.Module):
    def __init__(
        self,
        d=64,
        depth=2,
        heads=4,
        p=4,
        num_classes=8,
        drop_prob=0.1,
        temporal=True,
    ):
        super().__init__()

        # Store the patch size and space-time grid dimensions.
        self.p = p
        self.T, self.Hp, self.Wp = T, H // p, W // p

        # 1. Embed flattened single-channel spatial patches into d dimensions.
        self.embed = nn.Linear(p * p, d)

        # 2. Register fixed space-time positional embeddings as a buffer.
        # This tensor moves with the model but is not a trainable parameter.
        self.register_buffer(
            "pos",
            pos_embed(self.T, self.Hp, self.Wp, d),
        )

        # 3. Embed continuous flow timesteps.
        self.t_emb = TimestepEmbedder(d)

        # 4. Embed motion labels, including the unconditional null class.
        self.y_emb = LabelEmbedder(num_classes, d, drop_prob)

        # 5. Construct the transformer blocks.
        # Setting temporal=False enables the spatial-only ablation.
        self.blocks = nn.ModuleList(
            [
                DiTBlock(d, heads, temporal=temporal)
                for _ in range(depth)
            ]
        )

        # 6. Project each token back to its p x p pixel values.
        # The output projection is zero-initialized in FinalLayer.
        self.final = FinalLayer(d, p * p)

    def forward(self, x, t, y, gen=None):
        # x: (B, 1, T, H, W)
        # t: (B,)
        # y: (B,)

        # Convert the video into a sequence of flattened spatial patches.
        tokens = patchify(x, self.p)

        # Embed patches and add fixed positional information.
        tokens = self.embed(tokens) + self.pos.unsqueeze(0)

        # Combine timestep and motion-label conditioning.
        c = self.t_emb(t) + self.y_emb(y, gen)

        # Apply spatial attention, temporal attention (if enabled),
        # and the feed-forward network in each DiT block.
        S = self.Hp * self.Wp
        for block in self.blocks:
            tokens = block(tokens, c, self.T, S)

        # Predict the flattened patch values of the velocity field.
        tokens = self.final(tokens, c)

        # Reconstruct the video-shaped velocity field.
        return unpatchify(
            tokens,
            self.T,
            H,
            W,
            self.p,
            C=1,
        )

# Step 14 - sample_timesteps
def sample_timesteps(n, gen, mode="uniform"):
    if mode == "uniform":
        # Sample timesteps uniformly from [0, 1).
        return torch.rand(n, generator=gen)

    elif mode == "logit":
        # Sample from a standard normal distribution, then apply sigmoid.
        # This concentrates more samples around t = 0.5.
        z = torch.randn(n, generator=gen)
        return torch.sigmoid(z)

    else:
        raise ValueError(
            f"Unsupported timestep sampling mode: {mode}"
        )

# Step 15 - flow_interpolant
def flow_interpolant(x1, x0, t):
    # Broadcast each timestep across the channel, frame, height, and width axes.
    t = t.view(-1, 1, 1, 1, 1)

    # Interpolate along the straight path from Gaussian noise x0 to data x1.
    x_t = (1.0 - t) * x0 + t * x1

    # The velocity along this straight path is constant with respect to time.
    v = x1 - x0

    return x_t, v

# Step 16 - flow_matching_loss
def flow_matching_loss(model, x1, y, t, x0, gen=None):
    # Construct the interpolated video and its target velocity.
    x_t, v = flow_interpolant(x1, x0, t)

    # Predict the velocity, forwarding the generator to make label
    # dropout reproducible when the model is in training mode.
    prediction = model(x_t, t, y, gen)

    # Compute the mean squared error over all batch and video elements.
    return F.mse_loss(prediction, v)

# Step 17 - ema_update
def ema_update(ema, model, decay):
    # Update EMA parameters in place without building a computation graph.
    with torch.no_grad():
        for p_ema, p_model in zip(ema.parameters(), model.parameters()):
            # p_ema = decay * p_ema + (1 - decay) * p_model
            p_ema.mul_(decay).add_(p_model, alpha=1.0 - decay)

# Step 18 - train_video_dit
def train_video_dit(
    model,
    ema,
    steps,
    batch,
    lr,
    seed,
    t_mode="logit",
    ema_decay=0.99,
):
    # Create a seeded generator for reproducible training randomness.
    gen = torch.Generator().manual_seed(seed)

    # Initialize the AdamW optimizer with no weight decay.
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0,
    )

    # Enable training mode.
    model.train()

    # Store the loss from each optimization step as a Python float.
    losses = []

    for _ in range(steps):
        # 1. Generate a batch of synthetic videos and motion labels.
        x1, y = make_batch(gen, batch)

        # 2. Sample Gaussian noise with the same shape as the videos.
        x0 = torch.randn(x1.shape, generator=gen)

        # 3. Sample flow timesteps using the requested distribution.
        t = sample_timesteps(batch, gen, t_mode)

        # 4. Compute the flow-matching velocity prediction loss.
        loss = flow_matching_loss(
            model, x1, y, t, x0, gen
        )

        # 5. Clear gradients from the previous optimization step.
        opt.zero_grad()

        # 6. Backpropagate the loss.
        loss.backward()

        # 7. Update the model parameters.
        opt.step()

        # 8. Update EMA weights when an EMA model is provided.
        if ema is not None:
            ema_update(ema, model, ema_decay)

        # Record the loss as a Python float.
        losses.append(float(loss.detach().item()))

    return losses

# Step 19 - cfg_velocity
@torch.no_grad()
def cfg_velocity(model, x, t, y, scale):
    # At guidance scale 1, return the conditional prediction directly.
    if scale == 1.0:
        return model(x, t, y)

    # Create null-class labels for unconditional generation.
    null = torch.full_like(y, model.y_emb.num_classes)

    # Double the batch so conditional and unconditional predictions
    # can be computed in a single model forward pass.
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0)
    y_in = torch.cat([y, null], dim=0)

    # Compute both predictions together.
    predictions = model(x_in, t_in, y_in)

    # The first half is conditional; the second half is unconditional.
    v_cond, v_uncond = predictions.chunk(2, dim=0)

    # Apply classifier-free guidance.
    return v_uncond + scale * (v_cond - v_uncond)

# Step 20 - euler_sample
@torch.no_grad()
def euler_sample(model, y, steps, scale, gen):
    # Use evaluation mode to disable training-time label dropout.
    model.eval()

    # Number of videos to generate.
    n = y.shape[0]

    # Initialize the video state with Gaussian noise.
    # Shape: (n, 1, T, H, W)
    x = torch.randn(
        n, 1, T, H, W,
        generator=gen,
    )

    # Create a time grid from noise (t=0) to data (t=1).
    ts = torch.linspace(0.0, 1.0, steps + 1)

    # Integrate the learned velocity field using Euler's method.
    for i in range(steps):
        # Expand the scalar timestep to one value per video.
        t = ts[i].expand(n)

        # Predict the classifier-free-guided velocity.
        v = cfg_velocity(model, x, t, y, scale)

        # Euler update: x(t + dt) = x(t) + dt * v(x(t), t).
        dt = ts[i + 1] - ts[i]
        x = x + dt * v

    return x

# Step 21 - sample_from_first_frame
@torch.no_grad()
def sample_from_first_frame(model, first, y, steps, scale, gen):
    # Disable training-time behavior such as classifier-free label dropout.
    model.eval()

    # Determine the batch size.
    n = y.shape[0]

    # Draw initial Gaussian noise and preserve it for the interpolation.
    # Shape: (n, 1, T, H, W)
    x0 = torch.randn(
        n, 1, T, H, W,
        generator=gen,
    )

    # Start the sampling trajectory from the initial noise.
    x = x0.clone()

    # Create the integration time grid from noise (t=0) to data (t=1).
    ts = torch.linspace(0.0, 1.0, steps + 1)

    # Integrate the velocity field using Euler's method.
    for i in range(steps):
        t = ts[i]

        # Clamp the first frame to the noise-to-image interpolant:
        # at t=0 it is initial noise; at t=1 it becomes the given image.
        x[:, :, 0] = (
            (1.0 - t) * x0[:, :, 0]
            + t * first
        )

        # Predict the classifier-free-guided velocity.
        v = cfg_velocity(
            model,
            x,
            t.expand(n),
            y,
            scale,
        )

        # Take one Euler integration step.
        dt = ts[i + 1] - ts[i]
        x = x + dt * v

    # Ensure the first frame matches the given image exactly.
    x[:, :, 0] = first

    return x

# Step 22 - evaluate_samples
def shape_score(samples):
    # samples has shape (n, 1, T, H, W).
    n, _, T, _, _ = samples.shape

    # Handle an empty batch safely.
    if n == 0 or T == 0:
        return 0.0

    valid_frames = 0
    total_frames = n * T

    # Check every frame of every generated clip.
    for clip in samples:
        for frame in clip[0]:
            # Locate all bright pixels in the frame.
            rows, cols = torch.where(frame > 0)

            # A clean square must contain exactly K*K bright pixels.
            if rows.numel() != K * K:
                continue

            # Calculate the bounding-box height and width.
            box_height = int(rows.max().item() - rows.min().item() + 1)
            box_width = int(cols.max().item() - cols.min().item() + 1)

            # Count frames whose bright mask forms a clean K x K square.
            if box_height == K and box_width == K:
                valid_frames += 1

    return float(valid_frames / total_frames)


def evaluate_samples(samples, labels):
    # Evaluate frame quality independently of motion coherence.
    shape = shape_score(samples)

    n = samples.shape[0]

    # Handle an empty batch safely.
    if n == 0:
        return {
            "direction_acc": 0.0,
            "jitter": 0.0,
            "shape": round(shape, 4),
        }

    correct_directions = 0
    total_jitter = 0.0

    # Track the motion of each generated clip from its pixel values.
    for i in range(n):
        dx, dy, jitter = clip_velocity(samples[i])

        # Recover the closest direction index from the tracked velocity.
        predicted_direction = direction_index(dx, dy)

        # Compare the recovered direction with the conditioning label.
        if predicted_direction == int(labels[i].item()):
            correct_directions += 1

        total_jitter += jitter

    # Average the metrics across the batch.
    direction_acc = correct_directions / n
    mean_jitter = total_jitter / n

    return {
        "direction_acc": round(float(direction_acc), 4),
        "jitter": round(float(mean_jitter), 4),
        "shape": round(float(shape), 4),
    }

# Step 23 - frechet_distance
class ClipFeatures(nn.Module):
    def __init__(self, seed=0, dim=32):
        super().__init__()

        # Build the 3D convolutional feature extractor.
        self.c1 = nn.Conv3d(
            1, 8, kernel_size=3, stride=2, padding=1
        )
        self.c2 = nn.Conv3d(
            8, 16, kernel_size=3, stride=2, padding=1
        )

        # For an input video of shape (1, 6, 12, 12):
        # First convolution: (8, 3, 6, 6)
        # Second convolution: (16, 2, 3, 3)
        # Flattened feature dimension: 16 * 2 * 3 * 3 = 288.
        self.lin = nn.Linear(16 * 2 * 3 * 3, dim)

        # Reinitialize all parameters deterministically using the given seed.
        g = torch.Generator().manual_seed(seed)

        with torch.no_grad():
            for p in self.parameters():
                if p.ndim > 1:
                    # Initialize weights using the specified fan-in scaling.
                    p.copy_(
                        torch.randn(p.shape, generator=g, dtype=p.dtype)
                        / math.sqrt(p[0].numel())
                    )
                else:
                    # Initialize all biases to zero.
                    p.zero_()

                # Freeze the feature extractor.
                p.requires_grad_(False)

    def forward(self, x):
        # Extract spatiotemporal features using two convolutional layers.
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))

        # Flatten the feature maps and project to the feature dimension.
        x = x.flatten(start_dim=1)
        return self.lin(x)


def frechet_distance(f1, f2):
    # Calculate the feature means.
    mu1 = f1.mean(dim=0)
    mu2 = f2.mean(dim=0)

    # Estimate the covariance matrices across samples.
    c1 = torch.cov(f1.T)
    c2 = torch.cov(f2.T)

    # Compute the symmetric positive-semidefinite square root of c1.
    eigvals1, eigvecs1 = torch.linalg.eigh(c1)
    sqrt_eigvals1 = torch.sqrt(eigvals1.clamp_min(0.0))

    s1 = (
        eigvecs1
        @ torch.diag(sqrt_eigvals1)
        @ eigvecs1.T
    )

    # Compute the symmetric matrix s1 @ c2 @ s1.
    middle = s1 @ c2 @ s1
    middle = 0.5 * (middle + middle.T)

    # The trace of the square root equals the sum of square roots
    # of the eigenvalues, with negative roundoff values clamped to zero.
    eigvals_middle = torch.linalg.eigvalsh(middle)
    tr_sqrt = torch.sqrt(
        eigvals_middle.clamp_min(0.0)
    ).sum()

    # Squared distance between the means.
    mean_term = (mu1 - mu2).square().sum()

    # Fréchet (Wasserstein-2) distance between the fitted Gaussians.
    distance = (
        mean_term
        + torch.trace(c1)
        + torch.trace(c2)
        - 2.0 * tr_sqrt
    )

    # Prevent small numerical errors from yielding a negative distance.
    return float(distance.clamp_min(0.0).item())

# Step 24 - train_variant
def train_variant(
    temporal,
    steps,
    seed,
    d=64,
    depth=2,
    heads=4,
    batch=64,
    lr=3e-3,
):
    # Seed PyTorch before constructing either model.
    torch.manual_seed(seed)

    # Construct the live model and its EMA copy with identical settings.
    model = VideoDiT(
        d=d,
        depth=depth,
        heads=heads,
        temporal=temporal,
    )

    ema = VideoDiT(
        d=d,
        depth=depth,
        heads=heads,
        temporal=temporal,
    )

    # Initialize EMA with exactly the same state as the live model.
    ema.load_state_dict(model.state_dict())

    # Train the live model and update EMA throughout training.
    losses = train_video_dit(
        model=model,
        ema=ema,
        steps=steps,
        batch=batch,
        lr=lr,
        seed=seed,
    )

    # The EMA model is intended for inference and evaluation.
    ema.eval()

    # Count trainable model parameters.
    params = sum(p.numel() for p in model.parameters())

    # Return the complete reproducible training artifact.
    return {
        "ema": ema,
        "model": model,
        "losses": losses,
        "temporal": temporal,
        "params": params,
    }

