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

