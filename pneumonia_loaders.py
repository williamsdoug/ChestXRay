#!/usr/bin/env python
# coding: utf-8


import collections

from fastai.vision import *
# from fastai.callbacks import SaveModelCallback
# from fastai.metrics import error_rate
import os

#import fastai_addons   #add plot2 extension -- learn.recorder.plot2()
#from fastai_addons import interpretation_summary, plot_confusion_matrix,                           get_accuracy, analyze_confidence, accuracy_vs_threshold,                           show_incremental_accuracy,  analyze_low_confidence,                           plot_confusion_matrix_thresh, get_val_stats

# model = models.resnet18
# prefix = 'other_classifier2_'
# size=500
# bs = 64

# path = Path()/'data'/'chest_xray'
# path.ls()



def default_transforms():
    tfms = get_transforms(max_rotate=5.0)
#     tfms = get_transforms(do_flip=True, flip_vert=False,
#                           max_zoom=1.3, max_lighting=0.3)
    return tfms


def get_labels(file_path, known_classes=('normal', 'bacteria', 'virus'), default='normal'): 
    base = file_path.stem
    for k in known_classes:
        if k in base:
            return k
    return default


def filter_files(file_path, selected=('bacteria', 'virus')):
    base = file_path.stem
    for s in selected:
        if s in base:
            return True
    return False        


def sample_files(file_path, probs={'NORMAL':0.5, 'PNEUMONIA':0.25}, default_prob=0.125):
    fp = str(file_path)
    for k, v in probs.items():
        if k in fp:
            return random.random() < v
    return random.random() < default_prob


def get_labellist(path, filter_func=None, sample_func=None, p_sample=1.0, ll_only=False):
    ll = ImageList.from_folder(path)
    if filter_func is not None:ll = ll.filter_by_func(filter_func)
    if sample_func is not None:ll = ll.filter_by_func(sample_func)
    if p_sample != 1.0: ll = ll.filter_by_rand(p_sample)

    return ll


def characterize_labellist(*args, label_func=None, **kwargs):
    ll = get_labellist(*args, **kwargs)

    ll = ll.split_none()
    if label_func is None:
        ll = ll.label_from_folder()
    else:
        ll = ll.label_from_func(label_func)

    total = len(ll.y)
    c = collections.Counter([str(x) for x in ll.y])
    c = dict(c)
    c['_total'] = total
    return c


def show_categories(y):
    all_y = [x for x in y]
    tot = len(all_y)
    c = collections.Counter(all_y)
    keys = sorted(c.keys(), key=lambda k:str(k))
    for k in keys:
        v = c[k]
        print(f'  {str(k):10}:  {v:4d}     {100*v/tot:4.1f}%')
    print(f'  {"Total":10}:  {tot:4d}')


def get_db_np(path, tfms=None, size=224, bs=64, scale=1.0, workers=7):
    if tfms is None: tfms = default_transforms()
        
    sample_pneumonia = partial(sample_files, probs={'NORMAL':1.0, 'PNEUMONIA':0.346}, default_prob=0)
    #sample_pneumonia_v = partial(sample_files, probs={'NORMAL':1.0, 'PNEUMONIA':1.0}, default_prob=0)
    sample_pneumonia_t = partial(sample_files, probs={'NORMAL':1.0, 'PNEUMONIA':0.6}, default_prob=0)
        
    il_train = get_labellist(path/'train', sample_func=sample_pneumonia, p_sample=0.0373*scale)
    il_valid = get_labellist(path/'test', sample_func=sample_pneumonia_t, p_sample=0.216*scale)

    ils = ItemLists(path, train=il_train, valid=il_valid).label_from_folder()
    db = ImageDataBunch.create_from_ll(ils, size=size, ds_tfms=tfms, bs=bs, num_workers=workers)
    return db


def get_db_vb(path, tfms=None, size=224, bs=64, scale=1.0, label_func=get_labels, workers=7):
    if tfms is None: tfms = default_transforms()
    scale *= 1.5   # adjust size for missing normal
        
    sample_vb = partial(sample_files, probs={'bacteria':0.5316, 'virus':1.0}, default_prob=0)
    #sample_vb_v = partial(sample_files, probs={'bacteria':1.0, 'virus':1.0}, default_prob=0)
    sample_vb_t = partial(sample_files, probs={'bacteria':0.6115, 'virus':1.0}, default_prob=0)

    il_train = get_labellist(path/'train', sample_func=sample_vb, p_sample=0.0249*scale)
    il_valid = get_labellist(path/'test', sample_func=sample_vb_t, p_sample=0.2273*scale)
    ils = ItemLists(path, train=il_train, valid=il_valid).label_from_func(label_func)
    db = ImageDataBunch.create_from_ll(ils, size=size, ds_tfms=tfms, bs=bs, num_workers=workers)
    return db


def get_db_nvb(path, tfms=None, size=224, bs=64, scale=1.0, label_func=get_labels, workers=7):
    if tfms is None: tfms = default_transforms()
        
    sample_vb = partial(sample_files, probs={'bacteria':0.5316, 'virus':1.0}, default_prob=1.0)
    #sample_vb_v = partial(sample_files, probs={'bacteria':1.0, 'virus':1.0}, default_prob=1.0)
    sample_vb_t = partial(sample_files, probs={'bacteria':0.6115, 'virus':1.0}, default_prob=0.6325)

    il_train = get_labellist(path/'train', sample_func=sample_vb, p_sample=0.0249*scale)
    il_valid = get_labellist(path/'test', sample_func=sample_vb_t, p_sample=0.2273*scale)
    ils = ItemLists(path, train=il_train, valid=il_valid).label_from_func(label_func)
    db = ImageDataBunch.create_from_ll(ils, size=size, ds_tfms=tfms, bs=bs, num_workers=workers)
    return db
