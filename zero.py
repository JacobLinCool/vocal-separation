from typing import Callable
from functools import partial
import gradio as gr
import spaces
import spaces.config
from spaces.zero.decorator import P, R


def _dynGPU(
    fn: Callable[P, R] | None, duration: Callable[P, int], min=30, max=300, step=10
) -> Callable[P, R]:
    if not spaces.config.Config.zero_gpu:
        return fn

    funcs = [
        (t, spaces.GPU(duration=t)(lambda *args, **kwargs: fn(*args, **kwargs)))
        for t in range(min, max + 1, step)
    ]

    def wrapper(*args, **kwargs):
        requirement = duration(*args, **kwargs)

        # find the function that satisfies the duration requirement
        for t, func in funcs:
            if t >= requirement:
                gr.Info(f"Acquiring ZeroGPU for {t} seconds")
                return func(*args, **kwargs)

        # if no function is found, return the last one
        gr.Info(f"Acquiring ZeroGPU for {funcs[-1][0]} seconds")
        return funcs[-1][1](*args, **kwargs)

    return wrapper


def dynGPU(
    fn: Callable[P, R] | None = None,
    duration: Callable[P, int] = lambda: 60,
    min=30,
    max=300,
    step=10,
) -> Callable[P, R]:
    if fn is None:
        return partial(_dynGPU, duration=duration, min=min, max=max, step=step)
    return _dynGPU(fn, duration, min, max, step)
