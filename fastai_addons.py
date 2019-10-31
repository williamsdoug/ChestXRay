import numpy as np
from fastai.basic_train import Recorder
from fastai.core import ifnone, defaults, Any
from fastai.torch_core import to_np
from fastai.vision import *
import matplotlib.pyplot as plt
from typing import Optional
import scipy
import itertools


def silent_validate(model:nn.Module, dl:DataLoader, loss_func:OptLossFunc=None, cb_handler:Optional[CallbackHandler]=None,
             average=True, n_batch:Optional[int]=None)->Iterator[Tuple[Union[Tensor,int],...]]:
    """Calculate `loss_func` of `model` on `dl` in evaluation mode.  
       Note:  This version does not overwrite results from training"""
    model.eval()
    with torch.no_grad():
        val_losses,nums = [],[]
        if cb_handler: cb_handler.set_dl(dl)
        for xb,yb in dl:
            if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
            val_loss = loss_batch(model, xb, yb, loss_func, cb_handler=cb_handler)
            val_losses.append(val_loss)
            if not is_listy(yb): yb = [yb]
            nums.append(first_el(yb).shape[0])
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
            if n_batch and (len(nums)>=n_batch): break
        nums = np.array(nums, dtype=np.float32)
        if average: return (to_np(torch.stack(val_losses)) * nums).sum() / nums.sum()
        else:       return val_losses


def _svalidate(self, dl=None, callbacks=None, metrics=None):
        "Validate on `dl` with potential `callbacks` and `metrics`."
        dl = ifnone(dl, self.data.valid_dl)
        metrics = ifnone(metrics, self.metrics)
        cb_handler = CallbackHandler(self.callbacks + ifnone(callbacks, []), metrics)
        cb_handler.on_train_begin(1, None, metrics); cb_handler.on_epoch_begin()
        val_metrics = silent_validate(self.model, dl, self.loss_func, cb_handler)
        cb_handler.on_epoch_end(val_metrics)
        return cb_handler.state_dict['last_metrics']
    
Learner.svalidate = _svalidate

def get_val_stats(learner):
    rec = learner.recorder
    ret = {'loss':float(rec.val_losses[-1])}
    for i, name in enumerate(rec.metrics_names):
        ret[name] = float(rec.metrics[-1][i])
    return ret


def get_best_stats(learner):
    rec = learner.recorder
    keys = ['loss'] + rec.metrics_names
    results = []
    for i, loss in enumerate(rec.val_losses):
        entry = [loss] + [float(v) for v in rec.metrics[i]]
        results.append(dict(zip(keys, entry)))
    return sorted(results, key=lambda x:x['error_rate'])[0]


def my_smooth(sig, w=2):
    sig_p = np.pad(sig, (w,w), 'edge')
    return np.array([np.mean(sig_p[i:i+2*w+1]) for i in range(len(sig_p)-2*w)])


def plot2(self, skip_start:int=10, skip_end:int=5, suggestion:bool=True, return_fig:bool=None, win=3,
         **kwargs)->Optional[plt.Figure]:
    "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`. Optionally plot and return min gradient"
    lrs = self._split_list(self.lrs, skip_start, skip_end)
    losses = self._split_list(self.losses, skip_start, skip_end)
    losses = [x.item() for x in losses]
    all_losses = [losses]
    
    #if 'k' in kwargs: losses = self.smoothen_by_spline(lrs, losses, **kwargs)
    fig, ax = plt.subplots(1,1)
    ax.plot(lrs, losses)
    
    if win is not None: 
        losses2 = my_smooth(losses, w=win)
        all_losses.append(losses2)
        ax.plot(lrs, losses2, 'g', lw=0.5)
    
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    if suggestion:
        for i, l in enumerate(all_losses):
            tag = '' if i == 0 else ' (smoothed)'
            try: mg = (np.gradient(np.array(l))).argmin()
            except:
                print(f"Failed to compute the gradients{tag}, there might not be enough points.")
                return
            print(f"Min numerical gradient: {lrs[mg]:.2E} {tag}")
            color = 'r' if i == 0 else 'g'
            ax.plot(lrs[mg],losses[mg],markersize=10,marker='o',color=color)
            if i == 0: 
                self.min_grad_lr = lrs[mg]
                ml = np.argmin(l)
                ax.plot(lrs[ml],losses[ml],markersize=8,marker='o',color='k')
                print(f"Min loss divided by 10: {lrs[ml]/10:.2E}")
                ax.plot([lrs[ml]/10, lrs[ml]/10], [np.min(l), np.max(l)], 'k--', alpha=0.5)
                #print(np.min(l), np.max(l))
            elif i == 1:
                self.min_grad_lr_smoothed = lrs[mg]
        
    if ifnone(return_fig, defaults.return_fig): return fig
    try:
        if not IN_NOTEBOOK: plot_sixel(fig)
    except: pass
        
Recorder.plot2 = plot2


def threshold_confusion_matrix(interp, thresh=0):
    preds = to_np(interp.preds)
    y_true = to_np(interp.y_true)
    n = preds.shape[1]
    cm = np.zeros((n,n), dtype=np.int64)
    
    for j, p in enumerate(preds):
        i = np.argmax(p)
        y = y_true[j]
        if p[i] >= thresh:
            cm[y, i] += 1

    return cm


# fixed version -- consider submitting PR
def plot_confusion_matrix_thresh(self, normalize:bool=False, title:str='Confusion matrix', cmap:Any="Blues", slice_size:int=1,
                                 thresh:float=0.0,
                                 norm_dec:int=2, plot_txt:bool=True, return_fig:bool=None, **kwargs)->Optional[plt.Figure]:
        "Plot the confusion matrix, with `title` and using `cmap`."
        # This function is mainly copied from the sklearn docs
        if thresh == 0:
            cm = self.confusion_matrix(slice_size=slice_size)
        else:
            cm = threshold_confusion_matrix(self, thresh=thresh)
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = plt.figure(**kwargs)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(self.data.c)
        plt.xticks(tick_marks, self.data.y.classes, rotation=90)
        plt.yticks(tick_marks, self.data.y.classes, rotation=0)

        if plot_txt:
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
                plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.ylim(-0.5, self.data.c-0.5)
        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)
        if ifnone(return_fig, defaults.return_fig): return fig


# work-around version
def plot_confusion_matrix(interp, *args, **kwargs):
    interp.plot_confusion_matrix(*args, **kwargs)
    plt.ylim(-0.5, interp.data.c-0.5)


def _summarize_cm(cm, classes):
    n = len(classes)
    
    print(f'{" "*10}  Precision')
    print(f'{" "*10} Specificity     Recall   Sensitivity    F1')
    total = np.sum(cm)
    for i, c in enumerate(classes):
        predicted = np.sum(cm[:, i])
        actual = np.sum(cm[i, :])  

        TP = cm[i,i]
        FP = predicted - TP
        TN = total - actual - predicted + TP
        all_actual_neg = total - actual
          
        precision = TP/predicted
        recall = TP/actual
        sensitivity = TN/all_actual_neg
        F1 = scipy.stats.hmean([precision, recall])
        
        print(f'{c:10} {" "*3} {100*precision:3.0f}% {" "*6} {100*recall:3.0f}% {" "*6}' +
              f'{100*sensitivity:3.0f}% {" "*6} {F1:3.2f}')


def _consolidate_cm(cm, groupings, class_map):
    """consolidate confusion matrix by combining class groups"""
    # Group anong first axis
    cons = []
    for group, members in groupings.items():
        members = members if isinstance(members, (list, tuple)) else [members]
        cons.append(np.sum([cm[class_map[m],:] for m in members], axis=0))
    cons = np.stack(cons)
    # group along second axis
    grouped_cm = []
    for group, members in groupings.items():
        members = members if isinstance(members, (list, tuple)) else [members]
        grouped_cm.append(np.sum([cons[:,class_map[m]] for m in members], axis=0))
    grouped_cm = np.stack(grouped_cm).T
    return grouped_cm


def interpretation_summary(interp, groupings=None, strict=False, thresh:float=0.0):
    if thresh == 0:
        cm = interp.confusion_matrix()
    else:
        cm = threshold_confusion_matrix(interp, thresh=thresh)
              
    _summarize_cm(cm, interp.data.classes)

    if isinstance(groupings, dict):
        class_map = {c:i for i, c in enumerate(interp.data.classes)}
        grouped_cm = _consolidate_cm(cm, groupings, class_map)
        print('\nSummary by group')
        _summarize_cm(grouped_cm, list(groupings.keys()))

    print('\n')
    _get_accuracy(cm)


def _get_accuracy(cm, do_print=True):
    total = np.sum(cm)
    TP = np.sum([cm[i,i] for i in range(len(cm))])
    acc = float(TP)/total
    if do_print:
        print(f'Overall Accuracy: {acc*100:3.2f}%')
    else:
        return acc

def get_accuracy(interp, do_print=True):
    cm = interp.confusion_matrix()
    _get_accuracy(cm, do_print=do_print)


def analyze_confidence(interp, thresh = 0.0, do_plot=True, plot_args={'figsize':(10, 5)}, return_raw=False):
    p = to_np(interp.preds)
    y_true = to_np(interp.y_true)
    
    all_correct = [p[j, i] for j in range(len(y_true)) for i in [np.argmax(p[j])] 
                   if i ==y_true[j] and p[j, i] > thresh]
    all_wrong = [p[j, i] for j in range(len(y_true)) for i in [np.argmax(p[j])] 
                 if i !=y_true[j] and p[j, i] > thresh]
    
    total_predicted = len(all_correct) + len(all_wrong)
    acc = len(all_correct) / total_predicted
    missing = len(y_true) - total_predicted
    pct_unknown = missing / len(y_true)
    
    if do_plot and not return_raw:
        print(f'Accuracy: {100*acc:3.2f}%   Error: {100*(1-acc):3.2f}%   Unknown: {100*pct_unknown:3.2f}%   @ Threshold: {thresh:0.2f}')
        print('')

        colors = ['green', 'red']
        print(f'Confidence Histograms @ t={thresh}')
        fig, axs = plt.subplots(2, 2, **plot_args)

        ax = axs[0, 0]
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_title(f'Correct & Incorrect')
        ax.hist([all_correct, all_wrong], range=(0,1), bins=20, stacked=True, color=colors)
        ax.set_xlim(thresh, 1)

        ax = axs[0, 1]
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_title(f'Incorrect Only')
        ax.hist(all_wrong, range=(0,1), bins=20, color=colors[1])
        ax.set_xlim(thresh, 1)

        ax = axs[1, 0]
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_title(f'Log Correct & Incorrect')
        ax.hist([all_correct, all_wrong], range=(0,1), bins=20, log=True, stacked=True, color=colors)
        ax.set_xlim(thresh, 1)

        ax = axs[1, 1]
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_title(f'Log Incorrect Only')
        ax.hist(all_wrong, range=(0,1), bins=20, log=True, color=colors[1])
        ax.set_xlim(thresh, 1)
        fig.tight_layout()
        plt.show()
    elif return_raw:
        return total_predicted, len(all_correct), len(all_wrong), missing
    else:
        return acc, pct_unknown


def accuracy_vs_threshold(interp, threshold_range=(0, 0.90, 18), plot_args={'figsize':(8, 2)}):
    t_range = np.linspace(*threshold_range)
    results = np.array([analyze_confidence(interp, thresh=t, do_plot=False) for t in t_range])
              
    fig, axs = plt.subplots(1, 2, **plot_args)
    ax = axs[0]
    ax.set_title('Error Rate vs Threshold')
    ax.plot(t_range, 1 - results[:, 0])
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    ax = axs[1]
    ax.set_title('Witheld Predictions  vs Threshold')
    ax.plot(t_range, results[:, 1])
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
              
    fig.tight_layout()
    plt.show()


def show_incremental_accuracy(interp, plot_args={'figsize':(8,3)}):
    all_thresholds = np.linspace(0, 0.99, 25)
    results = np.array([analyze_confidence(interp, thresh=t, return_raw=True) for t in all_thresholds])
    all_predicted, all_correct, all_wrong,all_missing = results[:, 0], results[:, 1], results[:, 2], results[:, 3], 
    d_predicted, d_correct, d_wrong, d_missing = -np.diff(results[:, 0]), -np.diff(results[:, 1]), -np.diff(results[:, 2]), np.diff(results[:, 3])
              
    fig, axs = plt.subplots(1, 2, **plot_args)
    ax = axs[0]
    ax.set_title('Incremental Accuracy vs Threshold')
    ax.scatter([np.mean(all_thresholds[i:i+1]) for i, p in enumerate(d_predicted) if p >= 10],
                [d_correct[i]/p for i, p in enumerate(d_predicted) if p >= 10], s=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Incremental Accuracy')

    ax = axs[1]
    ax.set_title('Total Predicted vs Threshold')
    ax.plot([all_thresholds[i] for i, p in enumerate(all_predicted)],
             [p/all_predicted[0] for i, p in enumerate(all_predicted)])  
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Predicted')
    fig.tight_layout()
    plt.show()


def old_analyze_low_confidence(interp, thresh=0.0, thresh_min=0.0, display_mode='delta'):
    assert display_mode in ['delta', 'index', 'prediction']
    
    p = to_np(interp.preds)
    y_true = to_np(interp.y_true)
    
    wrong = [(j,p[j, i], p[j, y_true[j]],  p[j, i]-p[j, y_true[j]] )  for j in range(len(y_true)) for i in [np.argmax(p[j])] 
                 if i !=y_true[j] and p[j, i] < thresh and p[j, i] >= thresh_min]
    med_delta = np.median([w[3] for w in wrong])
    print(f'Total predictions in range: {len(wrong)}  Median delta: {med_delta:3.2f}')
    
    if display_mode == 'index':
        print('\nLow Confidence Predictions sorted by image number')
    elif display_mode == 'delta':
        wrong = sorted(wrong, key=lambda x:x[3])
        print('\nLow Confidence Predictions sorted by difference from correct answer')
    elif display_mode == 'prediction':
        wrong = sorted(wrong, key=lambda x:x[1])
        print('\nLow Confidence Predictions sorted by prediction confidence')
    
    
    #print('\nLow Confidence Predictions sorted by image number')
    for j, p_predict, p_correct, delta in wrong:
        print(f'{j:4d}:   p_predict: {p_predict:3.2f}   p_correct: {p_correct:3.2f}    delta: {delta:3.2f}')
        
    return


def analyze_row(j, preds, y):
    results = sorted([(i, p) for i, p in enumerate(preds)], key=lambda x:-x[1])
    i, p_pred = results[0]    # high top prediction
    i2, p_second = results[1]  # second prediction
    p_actual = preds[y]
    
    # row, p_pred, p_actual, delta_actual, is_second, p_second, delta_second
    return j,p_pred, p_actual,  p_pred-p_actual, i2==y, p_second, p_pred-p_second


def analyze_low_confidence(interp, thresh=0.0, thresh_min=0.0, display_mode='delta'):
    assert display_mode in ['delta', 'index', 'prediction']
    
    p = to_np(interp.preds)
    y_true = to_np(interp.y_true)
    
    wrong = [analyze_row(j, p[j], y_true[j])
             for j in range(len(y_true)) for i in [np.argmax(p[j])] 
             if i !=y_true[j] and p[j, i] < thresh and p[j, i] >= thresh_min]
    
    med_delta = np.median([w[3] for w in wrong])
    med_delta_next = np.median([w[6] for w in wrong])
    print(f'Total predictions in range: {len(wrong)}  Median del_act: {med_delta:3.2f}  Median del_next: {med_delta_next:3.2f}')
    
    if display_mode == 'index':
        print('\nLow Confidence Predictions sorted by image number')
    elif display_mode == 'delta':
        wrong = sorted(wrong, key=lambda x:x[3])
        print('\nLow Confidence Predictions sorted by difference from correct answer')
    elif display_mode == 'prediction':
        wrong = sorted(wrong, key=lambda x:x[1])
        print('\nLow Confidence Predictions sorted by prediction confidence')
        
    for j, p_predict, p_correct, delta_actual, is_second, p_second, delta_second in wrong:
        is_sec = 'T' if is_second else 'F'
        print(f'{j:4d}:   p_pred: {p_predict:3.2f}   p_corr: {p_correct:3.2f}    del_act: {delta_actual:3.2f}',end = '')
    
        print(f'    is_sec: {is_sec}   p_sec: {p_second:3.2f}   del_sec: {delta_second:3.2f}')       
