import numpy as np

def visnp(image_tensor):
    npa = image_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return npa