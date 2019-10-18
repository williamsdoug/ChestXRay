import numpy as np
import torch
import random
import itertools

def reset_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
# modification of: https://github.com/fastai/fastai/blob/master/fastai/train.py#L181
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap="Blues", slice_size=1,
                          norm_dec=2, plot_txt=True, return_fig=None, **kwargs):
    "Plot the confusion matrix, with `title` and using `cmap`."
    
    import matplotlib.pyplot as plt

    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(**kwargs)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes, rotation=0)

    if plot_txt:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
            plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.grid(False)
    return fig if return_fig else None
