import numpy as  np
import scipy as scp

class Image:

    def __init__(self):
        self.epithelial = []
        self.fibroblast = []
        self.inflammatory = []
        self.others = []
        self.image = np.zeros([500,500,3])


