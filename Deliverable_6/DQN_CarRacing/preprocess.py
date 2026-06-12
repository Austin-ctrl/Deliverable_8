import cv2
import numpy as np
import torch
from collections import deque

def preprocess_frame(frame):

    # RGB -> grayscale important because we don't care about colors
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2GRAY
    )
    # 96x96 -> 84x84
    frame = cv2.resize(
        frame,
        (84,84)
    )

    # normalize
    frame = frame / 255.0

    return torch.tensor(
        frame,
        dtype=torch.float32
    )
    

