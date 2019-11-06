from fastai.vision import *
from fastai.callbacks import SaveModelCallback
from fastai.metrics import error_rate
import os
import pickle
import scipy
from scipy.stats import gmean, hmean


from fastai_addons import model_cutter
from fastai_addons import interpretation_summary, plot_confusion_matrix

# from fastai_addons import interpretation_summary, plot_confusion_matrix, \
#                           get_accuracy, analyze_confidence, accuracy_vs_threshold, \
#                           show_incremental_accuracy,  analyze_low_confidence, \
#                           plot_confusion_matrix_thresh, get_val_stats, model_cutter


def get_best_stats(learner):
    rec = learner.recorder
    keys = ['loss'] + rec.metrics_names
    results = []
    for i, loss in enumerate(rec.val_losses):
        entry = [loss] + [float(v) for v in rec.metrics[i]]
        results.append(dict(zip(keys, entry)))
    return sorted(results, key=lambda x:x['error_rate'])[0]


def show_results(results, key=None, show_details=True, limit=None, sort_param='error_rate'):
    if key is not None:
        results = [x for x in results if key in x[0]]
        
    if len(results) > 1:
        err = [stats['error_rate'] for key, stats in results]
        loss = [stats['loss'] for key, stats in results]
        title = 'Overall' if key is None else key
        
        print(f'{title:14}  Error -- best: {np.min(err):.3f}  med: {np.median(err):.3f}   Loss -- best: {np.min(loss):.3f}  med: {np.median(loss):.3f}')
        if not show_details: return
        print('')
    results = sorted(results, key=lambda x:x[1][sort_param])
    
    if limit is None: limit = len(results)
    for key, stats in results[:limit]:
        print(f"{key:20}  error_rate: {stats['error_rate']:.3f}   accuracy: {stats['accuracy']:.3f}   loss:{stats['loss']:.4f}")


def _get_learner(db=None, model=None, model_dir=None, unfreeze=False, cut=None, **kwargs):
    # verity below params have been replaced using partial
    assert db is not None
    assert model is not None
    assert model_dir is not None

    if cut:
        assert isinstance(cut, int)
        mc = partial(model_cutter, select= [cut])
        my_split_on = lambda m: (m[0][cut],m[1])
        #def my_split_on(m): return (m[0][cut],m[1])
        learn = cnn_learner(db, model, metrics=[error_rate, accuracy], 
                            cut=mc, split_on=my_split_on,
                            model_dir=model_dir, **kwargs).to_fp16()
    else:
        learn = cnn_learner(db, model, metrics=[error_rate, accuracy], 
                        model_dir=model_dir, **kwargs).to_fp16()
    return learn



def _do_train(key, cycles, ps=None, mixup=False, unfreeze=False, cut=None, use_label_smoothing=False,
             get_learner=None, stats_repo=None, monitor='accuracy', **kwargs):
    assert stats_repo is not None
    assert get_learner is not None
    
    global all_results

    learn_args = {}
    if cut is not None:
        learn_args['cut'] = cut
        key = f'{key}_cut{cut}'
    if ps is not None:
        learn_args['ps'] = ps
        key = f'{key}_ps_{ps}'
    if use_label_smoothing:
        learn_args['loss_func'] = LabelSmoothingCrossEntropy()
        key = f'{key}_ls'
    learn = get_learner(**learn_args)

    # if ps is None:
    #     learn = get_learner(cut=cut)
    # else:
    #     learn = get_learner(ps=ps, cut=cut)
    #     key = f'{key}_ps_{ps}'

    if unfreeze == 'all':
        key = key + '_ufa' 
        learn.freeze()
    elif unfreeze:
        key = f'{key}_uf{unfreeze}' 
        learn.freeze_to(unfreeze)
    if mixup:
        if isinstance(mixup, float):
            learn = learn.mixup(mixup)
            key = key + f'_m{mixup}'
        else:
            learn = learn.mixup()
            key = key + '_m'
    
    print(key)
    learn.fit_one_cycle(cycles, callbacks=[SaveModelCallback(learn, every='improvement', 
                                                             monitor=monitor, name='best')], **kwargs)
    learn.recorder.plot_losses()
    plt.show()
    stats =  get_best_stats(learn)   
    learn.save(key)
    stats_repo.add([key, stats])
    return learn


def _get_interp(name, use_tta=False, get_learner=None):
    assert get_learner is not None
    learn = get_learner()
    learn.load(name)
    interp = learn.to_fp32().interpret(tta=use_tta)
    return interp


def analyze_interp(interp, include_norm=True):
    interpretation_summary(interp)
    plot_confusion_matrix(interp)
    plt.show()
    if include_norm:
        plot_confusion_matrix(interp, normalize=True)
        plt.show() 


def compute_acc(preds, y_true):
    yy = np.argmax(preds, axis=-1)
    return np.mean(yy==y_true)
    

def combine_predictions(all_interp):
    y_true = to_np(all_interp[0][1].y_true)
    all_preds = np.stack([to_np(interp.preds) for _, interp in all_interp])
    
    preds = np.mean(all_preds, axis=0)
    acc_m = compute_acc(preds, y_true) 
    
    preds = np.median(all_preds, axis=0)
    acc_med = compute_acc(preds, y_true)
    
    preds = gmean(all_preds, axis=0)
    acc_g = compute_acc(preds, y_true)
    
    preds = hmean(all_preds, axis=0)
    acc_h = compute_acc(preds, y_true)
    
    print(f'accuracy -- mean: {acc_m:0.3f}   median: {acc_med:0.3f}   gmean: {acc_g:0.3f}   hmean: {acc_h:0.3f}')
    return acc_m, acc_med, acc_g, acc_h


class StatsRepo:
    def __init__(self, prefix, force_init=False, stats_fn=None, checkpoint=False, verbose=False):
        self.prefix = prefix
        self.checkpoint = checkpoint
        self.verbose = verbose
        self.stats_fn = Path(stats_fn) if stats_fn else Path('stats')/f'{self.prefix}_stats.p'
        if not force_init and self.stats_fn.exists():
            self.restore()
        else:
            self.clear()
            
    def add(self, val):
        self.all_results.append(val)
        if self.checkpoint:
            self.save()
            
    def clear(self):
        if self.verbose: print('initialializing stats')
        self.all_results = []
        if self.checkpoint:
            self.save()
            
    def get(self):
        return self.all_results
        
    def save(self):
        with open(self.stats_fn, 'wb') as f:
            pickle.dump(self.all_results, f)
        if self.verbose: print('saved stats to:', self.stats_fn)
        
    def restore(self):
        with open(self.stats_fn, 'rb') as f:
            self.all_results = pickle.load(f)
        if self.verbose: print('restored stats from:', self.stats_fn)
    

def stats_repo_unit_test(prefix='unit_test'):
    stats = StatsRepo(prefix, force_init=True, stats_fn=None, checkpoint=False, verbose=True)
    print('** expected *', 'initialializing stats')

    print()
    print(stats.stats_fn)
    print('** expected *', 'stats/18_448_stats.p')

    stats.add('foobar')

    print()
    stats.save()
    print('** expected *', 'saved stats to: stats\18_448_stats.p')

    print()
    stats.restore()
    print('** expected *', 'saved stats to: stats\18_448_stats.p')

    print()
    print(stats.get())
    print('** expected *', "['foobar']")

    print()
    stats = StatsRepo(prefix, force_init=False, stats_fn=None, checkpoint=True, verbose=True)
    print('** expected *', 'restored stats from: stats\18_448_stats.p')

    print()
    stats.add('bar')
    print('** expected *', 'saved stats to: stats\18_448_stats.p')

    print()
    print(stats.get())
    print('** expected *', "['foobar', 'bar']")

    stats = StatsRepo(prefix, force_init=True, stats_fn=None, checkpoint=True, verbose=True)

    print()
    print(stats.get())
    print('** expected *', '[]') 
   