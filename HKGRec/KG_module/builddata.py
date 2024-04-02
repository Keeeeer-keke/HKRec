import tensorflow as tf
import numpy as np
import pickle
import time
import sys
ISOTIMEFORMAT = '%Y-%m-%d %X'  # 年月日时

tf.compat.v1.flags.DEFINE_string("data_dir", "./data", "The data dir.")
tf.compat.v1.flags.DEFINE_string("bin_postfix", "_permutate", "bin文件的新后缀.")
tf.compat.v1.flags.DEFINE_boolean("if_permutate", True, "If permutate for test filter.")
# tf.compat.v1.flags.DEFINE_string("bin_postfix", "", "bin文件的新后缀.")
# tf.compat.v1.flags.DEFINE_boolean("if_permutate", False, "If permutate for test filter.")



FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)

# 对数组进行全排列
def permutations(arr, position, end, res):
    """
    arr：要进行排列组合的数组
    position：当前递归的位置
    end：递归的结束位置
    res：存储结果的列表
    """
    if position == end:  # 遍历结束
        res.append(tuple(arr))
    else:
        for index in range(position, end):
            arr[index], arr[position] = arr[position], arr[index]  # 位置交换
            permutations(arr, position+1, end, res)  # 对剩余元素排列组合
            arr[index], arr[position] = arr[position], arr[index]
    return res

def load_data_from_txt(filenames, values_indexes = None, roles_indexes = None, ary_permutation = None):
    """
    获取文件名列表并建立相应的事实字典
    """
    if values_indexes is None:
        values_indexes = dict()  # 创建新的空字典用于存储索引
        values = set()  # 将values设为空集合，即暂时没有被任何值索引
        next_val = 0  # 下一个值的索引为0
    else:  # 使用提供的字典作为索引
        values = set(values_indexes)
        next_val = max(values_indexes.values()) + 1  # 下一个值的索引应该是最大值加1

    if roles_indexes is None:
        roles_indexes = dict()
        roles = set()
        next_role = 0
    else:
        roles = set(roles_indexes)
        next_role = max(roles_indexes.values()) + 1
    if ary_permutation is None:
        ary_permutation = dict()

    max_n = 2  # The maximum arity of the facts
    for filename in filenames:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                xx_dict = eval(line)
                xx = xx_dict['N']
                if xx > max_n:
                    max_n = xx  # N, arity
    data = []
    for i in range(max_n-1):
        data.append(dict())

    for filename in filenames:
        with open(filename) as f:
            lines = f.readlines()

        for _, line in enumerate(lines):  # 对每一行fact
            aline = ()
            xx_dict = eval(line)  # {'P0001_h': 'Q10001', 'P0001_t': 'Q10145', 'N': 4, 'P0005': ['Q10078'], 'P0006': ['Q10137']}
            for k in xx_dict:  # 'P0001_h'
                if k == 'N':  # 如果是N则继续
                    continue
                if k in roles:  # role在已有role列表中
                    role_ind = roles_indexes[k]
                else:
                    role_ind = next_role
                    next_role += 1
                    roles_indexes[k] = role_ind
                    roles.add(k)  # {'P0001_h'}
                if type(xx_dict[k]) == str:
                    val = xx_dict[k]  # Q
                    if val in values:  # val在已有val列表中
                        val_ind = values_indexes[val]
                    else:  # 没有则加上
                        val_ind = next_val
                        next_val += 1
                        values_indexes[val] = val_ind
                        values.add(val)
                    aline = aline + (role_ind,)
                    aline = aline + (val_ind,)
                else:
                    for val in xx_dict[k]:  # Multiple values
                        if val in values:
                            val_ind = values_indexes[val]
                        else:
                            val_ind = next_val
                            next_val += 1
                            values_indexes[val] = val_ind
                            values.add(val)
                        aline = aline + (role_ind,)
                        aline = aline + (val_ind,)

            if FLAGS.if_permutate == True:  # Permutate the elements in the fact for negative sampling or further computing the filtered metrics in the test process
                if xx_dict['N'] in ary_permutation:
                    res = ary_permutation[xx_dict['N']]
                else:
                    res = []
                    arr = np.array(range(xx_dict['N']))
                    res = permutations(arr, 0, len(arr), res)
                    ary_permutation[xx_dict['N']] = res
                for tpl in res:
                    tmpline = ()
                    for tmp_ind in tpl:
                        tmpline = tmpline + (aline[2*tmp_ind], aline[2*tmp_ind+1])
                    data[xx_dict['N']-2][tmpline] = [1]
            else:
                data[xx_dict['N']-2][aline] = [1]  # 未替换

    return data, values_indexes, roles_indexes, ary_permutation

def get_neg_candidate_set(folder, values_indexes, roles_indexes):
    """
    Get negative candidate set for replacing value
    """
    role_val = {}
    with open(folder + 'n-ary-1.11.json') as f:
        lines = f.readlines()
    for _, line in enumerate(lines):
        n_dict = eval(line)
        for k in n_dict:
            if k == 'N':
                continue
            k_ind = roles_indexes[k]
            if k_ind not in role_val:
                role_val[k_ind] = []
            v = n_dict[k]
            if type(v) == str:
                v_ind = values_indexes[v]
                if v_ind not in role_val[k_ind]:
                    role_val[k_ind].append(v_ind)
            else:  # Multiple values
                for val in v:
                    val_ind = values_indexes[val]
                    if val_ind not in role_val[k_ind]:
                        role_val[k_ind].append(val_ind)
    return role_val

def build_data(folder='data/'):
    """
    Build data and save to files
    """
    # 加载训练集
    train_facts, values_indexes, roles_indexes, ary_permutation = load_data_from_txt([folder + 'n-ary-1.11.json'])

    # 创建空字典，存储数据加载结果
    data_info = {}
    data_info["train_facts"] = train_facts
    data_info['values_indexes'] = values_indexes
    data_info['roles_indexes'] = roles_indexes
    # 没有被替换，则将负例候选集存储到data_info中，并保存到文件
    if FLAGS.if_permutate == False:
        role_val = get_neg_candidate_set(folder, values_indexes, roles_indexes)
        data_info['role_val'] = role_val
    with open(folder + "/dictionaries_and_facts" + FLAGS.bin_postfix + ".bin", 'wb') as f:
        pickle.dump(data_info, f)

if __name__ == '__main__':
    afolder = FLAGS.data_dir + '/'
    build_data(folder=afolder)
    print('build data completed!')
