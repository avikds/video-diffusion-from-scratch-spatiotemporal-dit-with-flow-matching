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

