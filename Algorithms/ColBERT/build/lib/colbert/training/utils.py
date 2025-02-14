import math
import os
from typing import *

import torch
import tqdm
import os
import torch
import diskcache as dc
# from colbert.utils.runs import Run
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS
from colbert.infra.run import Run

# Create a cache object
cache = dc.Cache("/tmp/diskcache")

def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)

def get_score_avg(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    return positive_avg, negative_avg

def manage_checkpoints(args, colbert, optimizer, batch_idx, savepath=None, consumed_all_triples=False):
    # arguments = dict(args)

    # TODO: Call provenance() on the values that support it??

    checkpoints_path = savepath or os.path.join(Run().path_, 'checkpoints')
    name = None

    try:
        save = colbert.save
    except:
        save = colbert.module.save

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    
    path_save = None

    if consumed_all_triples or (batch_idx % 2000 == 0):
        # name = os.path.join(path, "colbert.dnn")
        # save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
        path_save = os.path.join(checkpoints_path, "colbert")

    if batch_idx in SAVED_CHECKPOINTS or (batch_idx % 20 == 0):
        # name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        # save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
        path_save = os.path.join(checkpoints_path, f"colbert-{batch_idx}")

    if path_save:
        print(f"#> Saving a checkpoint to {path_save} ..")

        checkpoint = {}
        checkpoint['batch'] = batch_idx
        # checkpoint['epoch'] = 0
        # checkpoint['model_state_dict'] = model.state_dict()
        # checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        # checkpoint['arguments'] = arguments

        save(path_save)

    return path_save

# Decorator to cache function results
def disk_cache():
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = (func.__name__, args, tuple(kwargs.items()))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result

        return wrapper

    return decorator

@disk_cache()
def get_indices_to_avoid_repeated_qids_in_minibatch(
    qids: List[int], batch_size: int
) -> List[int]:
    """Returns a list of indices to avoid repeated qids in the minibatch."""

    def get_item(dic: Dict, get_unique_qid: bool = False) -> int:
        """Return the qid that appears the most in the dictionary."""
        max_key = 1 if get_unique_qid else max(dic.keys())
        qid = dic[max_key].pop(0)
        if len(dic[max_key]) == 0:
            dic.pop(max_key)
        if max_key > 1:
            # Append the qid back to the dictionary with a count of max_key-1
            if max_key - 1 in dic:
                dic[max_key - 1].append(qid)
            else:
                dic[max_key - 1] = [qid]
        return qid

    # List of indices of the qids
    qid_indices = {}
    for i, qid in enumerate(qids):
        if qid not in qid_indices:
            qid_indices[qid] = [i]
        else:
            qid_indices[qid].append(i)
    # Create dictionary that counts the number of times each qid appears
    dic: Dict[int, int] = {}
    for qid in qids:
        dic[qid] = dic.get(qid, 0) + 1

    # Inverted index of the dictionary
    new_dic: Dict[int, int] = {}
    for key, value in dic.items():
        if value not in new_dic:
            new_dic[value] = [key]
        else:
            new_dic[value].append(key)

    indices = []
    for i in tqdm.tqdm(
        range(0, math.ceil(len(qids) // batch_size)), desc="Shuffling train indices"
    ):
        # Add the index of the qid to the list of indices
        get_unique_qid = False
        tmp = []
        for _ in range(i * batch_size, min((i + 1) * batch_size, len(qids))):
            qid = get_item(new_dic, get_unique_qid)
            indices.append(qid_indices[qid].pop())
            get_unique_qid = True
            if qid in tmp:
                stop = 1
            tmp.append(qid)
    # Append the remaining indices
    for qid, item_indices in qid_indices.items():
        if dic:
            indices.extend(item_indices)
    return indices