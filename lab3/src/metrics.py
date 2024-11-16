import numpy as np


class AccuracyMetricFromProbs:
    def __init__(self, n_classes):
        self.name = "Accuracy"
        self.printout_name = "acc"

        self.n_classes = n_classes

        self.y_pred = np.empty(shape=(0, self.n_classes))
        self.y_corr = np.empty(shape=(0, self.n_classes))
        
        self.worst_value = 0

    def update_per_batch(self, batch_data, correct_data):
        self.y_pred = np.concatenate((self.y_pred, batch_data), axis=0)
        self.y_corr = np.concatenate((self.y_corr, correct_data), axis=0)
    
    def calculate(self):
        correct_samples = sum(self.y_pred.argmax(axis=1) == self.y_corr.argmax(axis=1))
        return correct_samples / len(self.y_pred)
    
    def reset(self):
        self.y_pred = np.empty(shape=(0, self.n_classes))
        self.y_corr = np.empty(shape=(0, self.n_classes))

    def compare_metrics(self, best, candidate):
        return candidate > best
    
