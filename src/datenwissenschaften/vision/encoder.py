import cv2
import numpy as np


class FixedVisualEncoder:
    _pooled_size = (4, 4)

    _kernels = np.asarray(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 1, 0], [1, 4, 1], [0, 1, 0]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
            [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        ],
        dtype=np.float32,
    )

    output_size = len(_kernels) * _pooled_size[0] * _pooled_size[1]

    _kernel_norms = np.maximum(
        np.abs(_kernels).sum(axis=(1, 2), keepdims=True),
        1.0,
    )

    def encode(self, observation: np.ndarray) -> list[float]:
        image = np.asarray(observation, dtype=np.float32)

        if image.ndim == 3:
            if image.shape[0] != 1:
                raise ValueError("FixedVisualEncoder expects a grayscale observation with one channel.")
            image = image[0]

        if image.ndim != 2:
            raise ValueError(f"Expected observation shape (1, H, W) or (H, W), got {image.shape}.")

        image = image / 255.0

        features = []

        for kernel, norm in zip(self._kernels, self._kernel_norms, strict=True):
            response = cv2.filter2D(
                image,
                ddepth=cv2.CV_32F,
                kernel=kernel / norm,
                borderType=cv2.BORDER_REFLECT,
            )
            pooled = cv2.resize(
                response,
                self._pooled_size,
                interpolation=cv2.INTER_AREA,
            )
            features.extend(pooled.reshape(-1))

        return np.asarray(features, dtype=np.float32).clip(-1.0, 1.0).tolist()
