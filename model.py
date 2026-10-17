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

