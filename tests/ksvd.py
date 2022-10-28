import numpy as np
import sys
sys.path.append("../")
import PyZooSVD

if __name__ == "__main__":
    
    dtype='float64'
    size = (5,3)
    A = np.random.random(size).astype(dtype) #+ 1j*np.random.random(size).astype(dtype)
    
    PyZooSVD.KSVD(A)
    
