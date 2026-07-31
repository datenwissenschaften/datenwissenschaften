import cv2
import numpy as np

SCENE_SIZE = 84


def scene(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif frame.ndim == 2:
        grayscale = frame
    else:
        raise ValueError(f"Frame must have two or three dimensions, got {frame.ndim}")
    resized = cv2.resize(grayscale, (SCENE_SIZE, SCENE_SIZE), interpolation=cv2.INTER_AREA)
    return resized[np.newaxis, :].astype(np.uint8, copy=False)
