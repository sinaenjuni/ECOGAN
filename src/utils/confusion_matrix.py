import torch
import numpy as np
# np.seterr(invalid='ignore')
class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confmat = np.zeros((num_classes, num_classes))
        self.minlength = num_classes**2
    def update(self, target, pred):
        assert len(target) == len(pred)
        unique_mapping = target.reshape(-1) * self.num_classes + pred.reshape(-1)
        bins = np.bincount(unique_mapping, minlength=self.minlength)
        self.confmat += bins.reshape(self.num_classes, self.num_classes)
    def getConfusionMactrix(self):
        return self.confmat
    def getAccuracy(self):
        return np.nan_to_num(self.confmat.trace() / self.confmat.sum())
    def getAccuracyPerClass(self):
        return np.nan_to_num(self.confmat.diagonal() / self.confmat.sum(1))
    def reset(self):
        self.confmat = np.zeros((self.num_classes, self.num_classes))

# Usage
# conf = ConfusionMatrix(10)
# target = (np.random.rand(100, 1) * 10).astype(np.long)
# pred = (np.random.rand(100, 1) * 9).astype(np.long)
# conf.update(target, pred)
#
# sum(target == pred) / len(target)
#
# np.unique(target, return_counts=True)
# conf.getAccuracy()
# conf.getAccuracyPerClass()