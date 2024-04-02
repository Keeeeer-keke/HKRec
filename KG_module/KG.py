import argparse, sys, pickle
from model import *
from random import shuffle
from copy import deepcopy
from pytictoc import TicToc
from batching import *


def chunks(L, n):
    """ Yield successive n-sized chunks from L."""
    for i in range(0, len(L), n):
        yield L[i:i+n]

def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main():
    outdir = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default='False', help='test mode(default: False)')
    parser.add_argument('--epochs', default=70)
    parser.add_argument('--lr', default=0.015)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--emb_size', default=64)
    parser.add_argument('--num_filters', default=150, type=int, help='number of filters CNN(default: 200)')
    parser.add_argument('--num_negative_samples', default=1, type=int, help='number of negative samples for each positive sample')
    parser.add_argument('--outdir', default=outdir, type=str, help='Output dir of model')
    args = parser.parse_args()
    for e in vars(args):
        print(e, getattr(args, e))
    device = torch.device('cpu')

    # Load training data
    with open("./data/dictionaries_and_facts.bin", 'rb') as fin:
        data_info = pickle.load(fin)
    train = data_info["train_facts"]
    entity_value_id = data_info['values_indexes']  # values_indexes
    file1 = open('entity_id.txt', 'w')
    for i, n in entity_value_id.items():
        file1.write(str(i) + ' ' + str(n) + '\n')
    file1.close()
    relation_key_id = data_info['roles_indexes']  # keys_indexes
    file2 = open('relation_id.txt', 'w')
    for j, m in relation_key_id.items():
        file2.write(str(j) + ' ' + str(m) + '\n')
    file2.close()
    key_val = data_info['role_val']

    id_entity_value = {}  # indexes_values
    for tmpkey in entity_value_id:
        id_entity_value[entity_value_id[tmpkey]] = tmpkey
    id_relation_key = {}  # indexes_roles
    for tmpkey in relation_key_id:
        id_relation_key[relation_key_id[tmpkey]] = tmpkey

    n_entity_value = len(entity_value_id)
    n_relation_key = len(relation_key_id)
    print("Unique number of relations and keys:", n_relation_key)
    print("Unique number of entities and values:", n_entity_value)

    head2id = {}
    tail2id = {}
    id2head = {}
    id2tail = {}
    keyH2keyT = {}
    print("\n**************creating extra dictionaries for hyper-relation kg**************\n")
    for r in relation_key_id:
        if r.endswith("_h"):
            head2id[r] = relation_key_id[r]
            id2head[relation_key_id[r]] = r
        elif r.endswith("_t"):
            tail2id[r] = relation_key_id[r]
            id2tail[relation_key_id[r]] = r
    for r_h_id in id2head:
        r_h_string = id2head[r_h_id]
        r_t_string = r_h_string.replace("_h", "_t")
        r_t_id = tail2id[r_t_string]
        keyH2keyT[r_h_id] = r_t_id

    list_of_head_ids = list(id2head.keys())
    list_of_tail_ids = []
    for r_h_id in list_of_head_ids:
        r_h_string = id2head[r_h_id]
        r_t_string = r_h_string.replace("_h", "_t")
        r_t_id = tail2id[r_t_string]
        list_of_tail_ids.append(r_t_id)

    with open("./data/dictionaries_and_facts_permutate.bin", 'rb') as fin:
        data_info1 = pickle.load(fin)
    whole_train = data_info1["train_facts"]

    model = KG(len(relation_key_id), len(entity_value_id), int(args.emb_size), int(args.num_filters))
    model.init()
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    t1 = TicToc()
    best_epoch, best_loss = 0, 100

    n_batches_per_epoch = []
    for i in train:
        ll = len(i)
        if ll == 0:
            n_batches_per_epoch.append(0)
        else:
            n_batches_per_epoch.append(int((ll - 1) / args.batch_size) + 1)
    for epoch in range(1, int(args.epochs)+1):
        t1.tic()
        model.train()
        model.to(device)
        train_loss = 0

        for i in range(len(train)):
            train_i_indexes = np.array(list(train[i].keys())).astype(np.int32)
            train_i_values = np.array(list(train[i].values())).astype(np.float32)

            for batch_num in range(n_batches_per_epoch[i]):
                arity = i + 2
                x_batch, y_batch = Batch_Loader(train_i_indexes, train_i_values, n_entity_value, n_relation_key, key_val, args.batch_size, arity, whole_train[i], id_entity_value, id_relation_key, keyH2keyT, args.num_negative_samples)
                pred_x, trained_emb = model(x_batch, arity, "training", device, id_relation_key, id_entity_value)
                pred = pred_x * torch.FloatTensor(y_batch).to(device) * (-1)
                loss = model.loss(pred).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()

                with open('train_emb/' + 'epoch_' + format(epoch) + 'batch_' + format(batch_num) + '.txt', 'w') as file:
                    for k, v in trained_emb.items():
                        file.write(str(k) + ' ' + str(v) + '\n')
                    file.close()
        t1.toc()
        print("End of epoch", epoch, "- train_loss:", train_loss, "- training time (seconds):", t1.elapsed)
        if epoch != 0 and best_loss > train_loss:
            best_epoch = epoch
            best_loss = train_loss
        print("best_epoch: {}".format(best_epoch), "best_loss: {}".format(best_loss))
    print("训练结束")

    # 保存最后模型
    file_name = "KG_" + str(args.batch_size) + "_" + str(args.epochs) + "_" + str(args.emb_size) + "_" + str(args.lr)
    print("Saving the model trained at epoch", epoch, "in:", './saved/' + file_name)
    if not os.path.exists('./saved/'):
        os.makedirs('./saved/')
    torch.save(model, './saved/' + file_name)
    print("Model saved")


if __name__ == '__main__':
    main()