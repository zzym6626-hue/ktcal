"""Consistent geometric and event-count augmentation for training records."""
import math

import torch
from torch.nn import functional as F


def warp_streaming_records(frames, labels, teacher, angles_degrees, scales, count_gains):
    """One inverse image warp and matching forward point transform per record.

    Coordinates are physical pixel centers, normalized by (80,60). The same
    transform applies to every timestep and polarity; no coordinate chooses it.
    """
    if (frames.ndim != 5 or frames.shape[2:] != (2, 60, 80)
            or min(frames.shape[:2]) < 1 or not frames.is_floating_point()
            or labels.shape != frames.shape[:2] + (2,) or teacher.shape != labels.shape
            or labels.device != frames.device or teacher.device != frames.device):
        raise ValueError('Aligned nonempty B,T,2,60,80 frames and B,T,2 targets required')
    parameters = [torch.as_tensor(value, dtype=frames.dtype, device=frames.device)
                  for value in (angles_degrees, scales, count_gains)]
    if any(value.shape != (len(frames),) or not torch.isfinite(value).all() for value in parameters):
        raise ValueError('Finite per-record angle, scale and gain arrays required')
    angle, scale, gain = parameters
    if (scale <= 0).any() or (gain <= 0).any():
        raise ValueError('Positive scale and event-count gain required')
    identity = bool((angle == 0).all() and (scale == 1).all())
    if identity:
        images, points, targets = frames, labels, teacher
    else:
        radians = angle * (math.pi / 180.)
        cosine, sine = radians.cos(), radians.sin()
        y, x = torch.meshgrid(torch.arange(60, dtype=frames.dtype, device=frames.device),
                              torch.arange(80, dtype=frames.dtype, device=frames.device), indexing='ij')
        dx, dy = x + .5 - 40., y + .5 - 30.
        c, s, size = (value[:, None, None] for value in (cosine, sine, scale))
        source_x = (c * dx + s * dy) / size + 39.5
        source_y = (-s * dx + c * dy) / size + 29.5
        grid = torch.stack((source_x * (2. / 79.) - 1., source_y * (2. / 59.) - 1.), -1)
        # Timesteps become channels so the fixed record grid needs no expansion.
        images = F.grid_sample(frames.reshape(len(frames), -1, 60, 80), grid,
                               mode='bilinear', padding_mode='zeros', align_corners=True).reshape_as(frames)

        def transform(coordinates):
            pixels = coordinates * coordinates.new_tensor([80., 60.])
            px, py = pixels[..., 0] - 40., pixels[..., 1] - 30.
            c, s, size = (value[:, None] for value in (cosine, sine, scale))
            moved = torch.stack((size * (c * px - s * py) + 40.,
                                 size * (s * px + c * py) + 30.), -1)
            return moved / coordinates.new_tensor([80., 60.])

        points, targets = transform(labels), transform(teacher)
    if not (gain == 1).all():
        images = images * gain[:, None, None, None, None]
    return images, points, targets
