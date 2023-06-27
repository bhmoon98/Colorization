import sys, os, csv, cv2
sys.path.append("/mnt/e/Program/Python/Colorization")
import torch


if __name__=='__main__':
    import gc

    gc.collect()
    torch.cuda.empty_cache()
