#!/usr/bin/env python
# coding: utf-8

# # Pneumonia Balanced Databunches

import collections
import pickle
from fastai.vision import *
import os
import pandas as pd 


def default_transforms():
    tfms = get_transforms(do_flip=True, flip_vert=False,
                          max_zoom=1.3, max_lighting=0.3)
    return tfms


def get_labels(file_path, known_classes=('normal', 'bacteria', 'virus'), default='normal'): 
    """get labels from filepath"""
    base = file_path.stem
    for k in known_classes:
        if k in base:
            return k
    return default


def filter_files(file_path, selected=('bacteria', 'virus')):
    """filter files based on filepath"""
    base = file_path.stem
    for s in selected:
        if s in base:
            return True
    return False  


def stringify(x):
    return [str(y) for y in x]


def get_rel_fn(x, base):
    return [y.relative_to(base) for y in x]


def list_to_ll(path, subset):
    """Converts list of features with label to fastai imagelist"""
    df = pd.DataFrame(subset, columns = ['path', 'label'])
    data = ImageList.from_df(df, path=path).split_none().label_from_df()
    #print(type(data), type(data.train))
    return data.train


def get_labellist_contents(path, filter_func=None, label_func=None):
    ll = ImageList.from_folder(path)
    if filter_func is not None:ll = ll.filter_by_func(filter_func)

    ll = ll.split_none()
    if label_func is None:
        ll = ll.label_from_folder()
    else:
        ll = ll.label_from_func(label_func)

    return ll.train


def get_dir_contents(path, subdir='train', **kwargs):
    ll = get_labellist_contents(path/subdir, **kwargs)
    all_items = list(zip(get_rel_fn(ll.items, path), stringify(ll.y)))
    grouped_items = {}
    keys = set([y for x, y in all_items])
    for k in keys:
        entries =  [(x, y) for x, y in all_items if y == k]
        random.shuffle(entries)
        grouped_items[k] = entries
    
    #print([(k, len(grouped_items[k])) for k in keys])
    return grouped_items


def get_sampled_data(path, n_samples=100, balance_valid=True, **kwargs):
    """generates new data set after sampling and scaling"""
    contents_tr = get_dir_contents(path, subdir='train', **kwargs)
    new_train = []
    for k, v in contents_tr.items():
        n = len(v)
        if n >= n_samples:
            new_train += v[:n_samples]
        else:
            times = n_samples//n
            mod = n_samples%n
            new_train += v * times
            if mod > 0: 
                new_train += v[:mod]
        
    contents_tst = get_dir_contents(path, subdir='test', **kwargs)
    new_test, new_valid = [], []
    
    if balance_valid:
        min_len = np.min([len(v) for v in contents_tst.values()])
        max_valid = min(min_len//2, n_samples)
    else:
        max_valid = n_samples
    
    for k, v in contents_tst.items():
        n = min(len(v) // 2,  max_valid)
        new_test += v[:n]
        
        v = v[n:]
        n = min(len(v),  max_valid)
        new_valid += v[:n]
        
    random.shuffle(new_train)
    random.shuffle(new_test) 
    random.shuffle(new_valid)
        
    return new_train, new_test, new_valid


def get_xray_databunch(path, scale=1, size=None, tfms=None, cache=None, **kwargs):
    if tfms is None:
        tfms = default_transforms()
        
    p_name = Path(f'{cache}.pkl')
    if cache is not None and p_name.exists():
        with open(p_name, 'rb') as f:
            new_train, new_test, new_valid = pickle.load(f)
    else:
        new_train, new_test, new_valid = get_sampled_data(path, **kwargs)
        if cache is not None:
            with open(p_name, 'wb') as f:
                pickle.dump([new_train, new_test, new_valid], f)

    if scale > 1:     # Duplicate entries, if requested 
        new_train = new_train * scale
        random.shuffle(new_train)

    ll_tr = list_to_ll(path, new_train)
    ll_val = list_to_ll(path, new_valid)
    ll_tst = list_to_ll(path, new_test)

    ils = ItemLists(path=path, train=ll_tr, valid=ll_val); ils

    db = ils.transform(tfms, size=size).databunch()
    return db

