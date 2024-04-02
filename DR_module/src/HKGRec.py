import torch
import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
from models import HKGRec
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params
import pandas as pd

torch.manual_seed(1203)

model_name = 'HKGRec'
resume_path = 'saved/HKGRec/'  

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument("--ddi", action="store_true", default=True, help="using ddi")
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
parser.add_argument("--T", type=float, default=2.0, help="T")
parser.add_argument("--decay_weight", type=float, default=0.85, help="decay weight")
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimension size')

args = parser.parse_args()

# evaluate
def eval(model, ICD_tensor_dict, ATC_tensor_dict, data_eval, voc_size, epoch):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        if len(input) < 2:
            continue
        for i in range(1, len(input)):
            target_output = model(input[:i], ICD_tensor_dict, ATC_tensor_dict)

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[input[i][2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prob
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(y_pred_label_tmp)
            med_cnt += len(y_pred_label_tmp)
            visit_cnt += 1

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path="../data/output/ddi_A_final.pkl")

    llprint(
        "\nDDI Rate: {:.4f}, Jaccard: {:.4f},  PRAUC: {:.4f}, AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )

def main():

    # load data
    data_path = '../data/output/records_final.pkl'  # 0,1,2,3...编码后
    voc_path = '../data/output/voc_final.pkl'  # 原编码
    ehr_adj_path = '../data/output/ehr_adj_final.pkl'
    ddi_adj_path = '../data/output/ddi_A_final.pkl'

    device = torch.device('cuda')
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))  # diag_voc, med_voc, pro_voc
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    kg_tensor = pd.read_csv('../data/output/ATC_tensor.txt')

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    # dict
    med_voc_idx, diag_voc_idx = voc['med_voc'].idx2word, voc['diag_voc'].idx2word
    diag_voc_df = pd.DataFrame.from_dict(diag_voc_idx, orient='index', columns=['ATC'])
    ICD_tensor = pd.merge(diag_voc_df, kg_tensor, how='left', on='ATC')
    ICD_tensor_dict = dict(zip(ICD_tensor['ATC'], ICD_tensor['tensor']))
    for key in ICD_tensor_dict:
        value = ICD_tensor_dict[key]
        if isinstance(value, float):
            ICD_tensor_dict[key] = torch.zeros(64)
        else:
            tensor_set = torch.tensor([float(x) for x in value.strip('[]').split(',')]).unsqueeze(0)
            ICD_tensor_dict[key] = tensor_set.reshape(64)

    med_voc_df = pd.DataFrame.from_dict(med_voc_idx, orient='index', columns=['ATC'])
    ATC_tensor = pd.merge(med_voc_df, kg_tensor, how='left', on='ATC')
    ATC_tensor_dict = dict(zip(ATC_tensor['ATC'], ATC_tensor['tensor']))
    for key in ATC_tensor_dict:
        value = ATC_tensor_dict[key]
        if isinstance(value, float):
            ATC_tensor_dict[key] = torch.zeros(64)
        else:
            tensor_set = torch.tensor([float(x) for x in value.strip('[]').split(',')]).unsqueeze(0)
            ATC_tensor_dict[key] = tensor_set.reshape(64)

    # 数据集划分
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))  

    model = HKGRec(voc_size, ehr_adj, ddi_adj, ICD_tensor_dict, ATC_tensor_dict, emb_dim=args.emb_dim, device=device)

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, "rb")))
        model.to(device=device)
        tic = time.time()
        result = []
        for _ in range(10):
            test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, ICD_tensor_dict, ATC_tensor_dict, test_sample, voc_size, 0)
            result.append([ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print(outstring)
        print("test time: {}".format(time.time() - tic))
        return

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=args.lr)
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    # 训练模式
    EPOCH = 30
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch))
        prediction_loss_cnt, neg_loss_cnt = 0, 0
        model.train()
        for step, input in enumerate(data_train):  # 对多个患者数据的列表进行迭代
            # 对当前患者的每次就诊记录进行迭代
            # input:[[[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2], [0]], [[1, 3, 9, 10, 6, 5], [0], [0]]]
            if len(input) < 2:
                continue
            for i in range(1, len(input)):
                loss_bce_target = np.zeros((1, voc_size[2]))
                # input[i]: [[1, 3, 9, 10, 6, 5], [0], [0]]
                loss_bce_target[:, input[i][2]] = 1
                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(input[i][2]):
                    loss_multi_target[0][idx] = item
                # 当前患者的之前就诊记录 input[:i]:[[[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2], [0]]]
                target_output1, loss_ddi = model(input[:i], ICD_tensor_dict, ATC_tensor_dict)
                # 计算损失
                loss_bce = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss_bce_target).to(device))
                loss_multi = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss_multi_target).to(device))
                if args.ddi:
                    target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                    target_output1[target_output1 >= 0.5] = 1
                    target_output1[target_output1 < 0.5] = 0
                    y_label = np.where(target_output1 == 1)[0]
                    current_ddi_rate = ddi_rate_score([[y_label]], path="../data/output/ddi_A_final.pkl")
                    if current_ddi_rate <= args.target_ddi:
                        loss = 0.9 * loss_bce + 0.1 * loss_multi
                        prediction_loss_cnt += 1
                    else:
                        rnd = np.exp((args.target_ddi - current_ddi_rate) / args.T)
                        if np.random.rand(1) < rnd:
                            loss = loss_ddi
                            neg_loss_cnt += 1
                        else:
                            loss = 0.9 * loss_bce + 0.1 * loss_multi
                            prediction_loss_cnt += 1
                else:
                    loss = 0.9 * loss_bce + 0.1 * loss_multi

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

        args.T *= args.decay_weight

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, ICD_tensor_dict, ATC_tensor_dict, data_eval, voc_size, epoch)
        print("training time: {}, test time: {}".format(time.time() - tic, time.time() - tic2))

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )

        torch.save(
            model.state_dict(),
            open(
                os.path.join("saved", args.model_name,
                    "Epoch_{}_JA_{:.4f}_DDI_{:.4f}.model".format(epoch, ja, ddi_rate),
                ),
                "wb",
            ),
        )

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print("best_epoch: {}".format(best_epoch))

    dill.dump(
        history,
        open(
            os.path.join("saved", args.model_name, "history_{}.pkl".format(args.model_name)), "wb",
        ),
    )

if __name__ == "__main__":
    main()
