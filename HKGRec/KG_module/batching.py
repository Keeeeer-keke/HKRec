import numpy as np
import random


def replace_val(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples):
    """
    随机替换value以获得负样本
    """
    rmd_dict = key_val
    new_range = (last_idx * num_negative_samples)

    for cur_idx in range(new_range):
        key_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
        tmp_key = new_facts_indexes[last_idx + cur_idx, key_ind]
        tmp_len = len(rmd_dict[tmp_key])
        rdm_w = np.random.randint(0, tmp_len)

        times = 1
        tmp_array = new_facts_indexes[last_idx + cur_idx]
        tmp_array[key_ind+1] = rmd_dict[tmp_key][rdm_w]
        while (tuple(tmp_array) in whole_train_facts):
            if (tmp_len == 1) or (times > 2*tmp_len) or (times > 100):
                tmp_array[key_ind+1] = np.random.randint(0, n_values)
            else:
                rdm_w = np.random.randint(0, tmp_len)
                tmp_array[key_ind+1] = rmd_dict[tmp_key][rdm_w]
            times = times + 1
        new_facts_indexes[last_idx + cur_idx, key_ind+1] = tmp_array[key_ind+1]
        new_facts_values[last_idx + cur_idx] = [-1]

def replace_key(n_keys, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, keyH2keyT, num_negative_samples):
    """
    随机替换key以获得负样本
    """

    new_range = (last_idx*num_negative_samples)
    rdm_ws = np.random.randint(0, n_keys, new_range)

    for cur_idx in range(new_range):
        key_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
        # Sample a random key
        tmp_array = new_facts_indexes[last_idx + cur_idx]
        if key_ind == 0 or key_ind == 2:
            tmp_array[0] = rdm_ws[cur_idx]
            tmp_array[2] = rdm_ws[cur_idx]
        else:
            tmp_array[key_ind] = rdm_ws[cur_idx]

        while (tuple(tmp_array) in whole_train_facts):
            rnd_key = np.random.randint(0, n_keys)
            if key_ind == 0 or key_ind == 2:
                tmp_array[0] = rnd_key
                tmp_array[2] = rnd_key
            else:
                tmp_array[key_ind] = rnd_key

        new_facts_indexes[last_idx + cur_idx, key_ind] = tmp_array[key_ind]
        new_facts_values[last_idx + cur_idx] = [-1]

'''根据正样本数据生成相应数量的负样本，并返回生成的新事实索引和值'''

def Batch_Loader(train_i_indexes, train_i_values, n_values, n_keys, key_val, batch_size, arity, whole_train_facts, indexes_values, indexes_keys, keyH2keyT, num_negative_samples):
    # train_i_indexes: all facts with the same arity
    new_facts_indexes = np.empty((batch_size+(batch_size*num_negative_samples), 2*arity)).astype(np.int32)
    new_facts_values = np.empty((batch_size+(batch_size*num_negative_samples), 1)).astype(np.float32)
    idxs = np.random.randint(0, len(train_i_values), batch_size)
    new_facts_indexes[:batch_size, :] = train_i_indexes[idxs, :]
    new_facts_values[:batch_size] = train_i_values[idxs, :]
    last_idx = batch_size
    new_facts_indexes[last_idx:last_idx+(last_idx*num_negative_samples), :] = np.tile(new_facts_indexes[:last_idx, :], (num_negative_samples, 1))
    new_facts_values[last_idx:last_idx+(last_idx*num_negative_samples)] = np.tile(new_facts_values[:last_idx], (num_negative_samples, 1))
    val_key = random.uniform(0, 1)
    if val_key < 0.5:  # 如果小于0.5，则替换value
        replace_val(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples)
    else:  # 替换key
        replace_key(n_keys, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, keyH2keyT, num_negative_samples)
    last_idx += batch_size

    return new_facts_indexes, new_facts_values
