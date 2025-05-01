"""
Copyright (c) 2013--2023, librosa development team.

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


***This file extracted from librosa package since we use only the trim() function and librosa requires many dependencies***

Reference:
    - https://gist.github.com/evq/82e95a363eeeb75d15dd62abc1eb1bde
    - https://github.com/librosa/librosa/blob/894942673d55aa2206df1296b6c4c50827c7f1d6/librosa/effects.py#L612
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import as_strided


class LibrosaError(Exception):
    pass


class ParameterError(LibrosaError):
    pass

def _cabs2(x):  # pragma: no cover
    """Efficiently compute abs2 on complex inputs"""
    return x.real**2 + x.imag**2


def abs2(x, dtype):
    if np.iscomplexobj(x):
        # suppress type check, mypy doesn't like vectorization
        y = _cabs2(x)
        if dtype is None:
            return y  # type: ignore
        else:
            return y.astype(dtype)  # type: ignore
    else:
        # suppress type check, mypy doesn't know this is real
        return np.square(x, dtype=dtype)  # type: ignore


def amplitude_to_db(
    S,
    *,
    ref: float | Callable = 1.0,
    amin: float = 1e-5,
    top_db: float | None = 80.0,
) -> np.floating[Any] | np.ndarray:
    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "amplitude_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call amplitude_to_db(np.abs(S)) instead.",
            stacklevel=2,
        )

    magnitude = np.abs(S)

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    out_array = magnitude if isinstance(magnitude, np.ndarray) else None
    power = np.square(magnitude, out=out_array)

    db: np.ndarray = power_to_db(power, ref=ref_value**2, amin=amin**2, top_db=top_db)
    return db


def _signal_to_frame_nonsilent(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
    top_db: float = 60,
    ref: Callable | float = np.max,
    aggregate: Callable = np.max,
) -> np.ndarray:
 
    # Compute the MSE for the signal
    mse = rms(y=y, frame_length=frame_length, hop_length=hop_length)

    # Convert to decibels and slice out the mse channel
    db: np.ndarray = amplitude_to_db(mse[..., 0, :], ref=ref, top_db=None)

    # Aggregate everything but the time dimension
    if db.ndim > 1:
        db = np.apply_over_axes(aggregate, db, range(db.ndim - 1))
        # Squeeze out leading singleton dimensions here
        # We always want to keep the trailing dimension though
        db = np.squeeze(db, axis=tuple(range(db.ndim - 1)))

    return db > -top_db


def trim(
    y: np.ndarray,
    *,
    top_db: float = 60,
    ref: float | Callable = np.max,
    frame_length: int = 2048,
    hop_length: int = 512,
    aggregate: Callable = np.max,
) -> tuple[np.ndarray, np.ndarray]:
    non_silent = _signal_to_frame_nonsilent(
        y,
        frame_length=frame_length,
        hop_length=hop_length,
        ref=ref,
        top_db=top_db,
        aggregate=aggregate,
    )

    nonzero = np.flatnonzero(non_silent)

    if nonzero.size > 0:
        # Compute the start and end positions
        # End position goes one frame past the last non-zero
        start = int(frames_to_samples(nonzero[0], hop_length=hop_length))
        end = min(
            y.shape[-1],
            int(frames_to_samples(nonzero[-1] + 1, hop_length=hop_length)),
        )
    else:
        # The entire signal is trimmed here: nothing is above the threshold
        start, end = 0, 0

    # Slice the buffer and return the corresponding interval
    return y[..., start:end], np.asarray([start, end])


def rms(
    *,
    y: np.ndarray | None = None,
    S: np.ndarray | None = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode="constant",
    dtype=np.float32,
) -> np.ndarray:
    if y is not None:
        if center:
            padding = [(0, 0) for _ in range(y.ndim)]
            padding[-1] = (int(frame_length // 2), int(frame_length // 2))
            y = np.pad(y, padding, mode=pad_mode)

        x = frame(y, frame_length=frame_length, hop_length=hop_length)

        # Calculate power
        power = np.mean(abs2(x, dtype=dtype), axis=-2, keepdims=True)
    elif S is not None:
        # Check the frame length
        if S.shape[-2] != frame_length // 2 + 1:
            raise ParameterError(
                f"Since S.shape[-2] is {S.shape[-2]}, "
                f"frame_length is expected to be {S.shape[-2] * 2 - 2} or {S.shape[-2] * 2 - 1}; "
                f"found {frame_length}"
            )

        # power spectrogram
        x = abs2(S, dtype=dtype)

        # Adjust the DC and sr/2 component
        x[..., 0, :] *= 0.5
        if frame_length % 2 == 0:
            x[..., -1, :] *= 0.5

        # Calculate power
        power = 2 * np.sum(x, axis=-2, keepdims=True) / frame_length**2
    else:
        raise ParameterError("Either `y` or `S` must be input.")

    rms_result: np.ndarray = np.sqrt(power)
    return rms_result


def frame(
    x: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
    writeable: bool = False,
    subok: bool = False,
) -> np.ndarray:

    x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise ParameterError(
            f"Input is too short (n={x.shape[axis]:d}) for frame_length={frame_length:d}"
        )

    if hop_length < 1:
        raise ParameterError(f"Invalid hop_length: {hop_length:d}")

    out_strides = x.strides + tuple([x.strides[axis]])

    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]


def power_to_db(
    S,
    *,
    ref: float | Callable = 1.0,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> np.floating[Any] | np.ndarray:
    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def frames_to_samples(
    frames,
    *,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> np.integer[Any] | np.ndarray:

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)
