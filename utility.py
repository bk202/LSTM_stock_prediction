import numpy as np

def reformat_labels(labels, num_labels):
    labels = (np.arange(num_labels)) == labels[:, None].astype(np.float32)
    return labels

def accuracy(predictions, labels):
    # match = 0
    # for i in range(len(predictions)):
    #     if predictions[i] == labels[i]:
    #         match += 1
    #
    # return 100 * match / predictions.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def fbrn_norm_filter(dataset, labels, fbrn_norm=8):
    return_dataset = []
    return_labels = []
    for i in range(len(dataset)):
        norm = np.linalg.norm(dataset[i])
        if(norm > fbrn_norm and norm != float('Inf') and norm != float('inf') and norm != float('NaN')):
            return_dataset.append(dataset[i])
            return_labels.append(labels[i])

    return_dataset = np.asarray(return_dataset)
    return_labels = np.asarray(return_labels)
    return return_dataset, return_labels

def rgb2grey(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

class AverageMeter:
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


