import numpy as np


class SubImage:

    def __init__(self, img=None ,label=None):
        self.label = label
        self.image = img


    def translateLabel(self):
        if self.label == 0:
            return 'epithelial'

        if self.label == 1:
            return 'fibroblast'

        if self.label == 2:
            return 'inflammatory'

        if self.label == 3:
            return 'other'

        return 'Has no label'
