  # 划分测试集训练集
from sklearn.model_selection import train_test_split

def write_data(datapath, line_sen_list):
    with open(datapath, 'w', encoding='utf-8') as o:
        o.write(''.join(line_sen_list))
        o.close()

def main():
    raw_data_path = './n-ary.json'
    train_data_path = './n-ary_train.json'
    validate_data_path = './n-ary_valid.json'
    test_data_path = './n-ary_test.json'

    line_sen_list = []

    with open(raw_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 按某种规律选取固定大小的数据集
        for line in lines[0:100000]:
            line_sen_list.append(''.join(line))
        f.close()

    label_list = [0] * 76  # 由于该数据形式为文本，且形式为数据和标签在一起，所以train_test_split()中标签可以给一个相同大小的0值列表，无影响。

    # 先将1.训练集，2.验证集+测试集，按照8：2进行随机划分
    X_train, X_validate_test, _, y_validate_test = train_test_split(line_sen_list, label_list, test_size=0.2,
                                                                    random_state=42)
    # 再将1.验证集，2.测试集，按照1：1进行随机划分
    X_validate, X_test, _, _ = train_test_split(X_validate_test, y_validate_test, test_size=0.5, random_state=42)

    # 分别将划分好的训练集，验证集，测试集写入到指定目录
    write_data(train_data_path, X_train)
    write_data(validate_data_path, X_validate)
    write_data(test_data_path, X_test)


if __name__ == '__main__':
    main()


