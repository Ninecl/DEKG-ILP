import os
import argparse
import pickle

import numpy as np

from utils.data_utils import load_raw_data, read_triplets2id, count_entity_relation_set, write_triplets


def load_grail_data(dataset, version, entity2id, relation2id):
    """读取grail的数据

    Args:
        dataset (str): 读取哪个数据集
        version (int): 读取v几的数据
        entity2id (dict): entity到id的映射字典
        relation2id (dict): relation到id的映射字典

    Returns:
        original_triplets (list): original的所有triplets，二维列表
        original_entity_set (set): original中的所有entity
        original_relation_set (set): original中的所有relation
        emerging_triplets (list): emerging的所有triplets，二维列表
        emerging_entity_set (set): emerging中的所有entity
        emerging_relation_set (set): emerging中的所有relation
    """
    original_data_path = f'./data/grail/{dataset}_v{version}'
    emerging_data_path = f'./data/grail/{dataset}_v{version}_ind'

    original_train_triplets, entity2id, relation2id = read_triplets2id(os.path.join(original_data_path, 'train.txt'), entity2id, relation2id)
    original_valid_triplets, entity2id, relation2id = read_triplets2id(os.path.join(original_data_path, 'valid.txt'), entity2id, relation2id)
    original_test_triplets, entity2id, relation2id = read_triplets2id(os.path.join(original_data_path, 'test.txt'), entity2id, relation2id)
    original_triplets = original_train_triplets + original_valid_triplets + original_test_triplets
    original_entity_set , original_relation_set = count_entity_relation_set(original_triplets)
    

    emerging_train_triplets, entity2id, relation2id = read_triplets2id(os.path.join(emerging_data_path, 'train.txt'), entity2id, relation2id)
    emerging_valid_triplets, entity2id, relation2id = read_triplets2id(os.path.join(emerging_data_path, 'valid.txt'), entity2id, relation2id)
    emerging_test_triplets, entity2id, relation2id = read_triplets2id(os.path.join(emerging_data_path, 'test.txt'), entity2id, relation2id)
    emerging_triplets = emerging_train_triplets + emerging_valid_triplets + emerging_test_triplets
    emerging_entity_set, emerging_relation_set = count_entity_relation_set(emerging_triplets)

    return original_train_triplets, original_valid_triplets, original_test_triplets, original_entity_set, original_relation_set, \
           emerging_train_triplets, emerging_valid_triplets, emerging_test_triplets, emerging_entity_set, emerging_relation_set, \
           entity2id, relation2id


def screen_u2s_triplets(raw_triplets, original_entity_set, emerging_entity_set):

    u2s_triplets = []
    
    for triplet in raw_triplets:
        h, r, t = triplet
        if h in original_entity_set and t in emerging_entity_set:
            u2s_triplets.append(triplet)
        elif t in original_entity_set and h in emerging_entity_set:
            u2s_triplets.append(triplet)
    
    return u2s_triplets


def main(params):

    # 读取raw数据
    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_raw_data('./data/{}'.format(params.dataset))

    all_triplets = train_triplets + valid_triplets + test_triplets

    entity_set = set(range(len(entity2id)))
    relation_set = set(range(len(relation2id)))


    version = [1, 2, 3]

    for v in version:
        
        # 读取grail中的数据
        # 注意，这里处理数据时发现了一个问题，grail提供的entity2id和relation2id，在nell和wn上均出现了不全的情况，因此这里重新统计并做了补全
        original_train_triplets, original_valid_triplets, original_test_triplets, original_entity_set, original_relation_set, \
        emerging_train_triplets, emerging_valid_triplets, emerging_test_triplets, emerging_entity_set, emerging_relation_set, \
        entity2id, relation2id = load_grail_data(params.dataset, v, entity2id, relation2id)
        # 也更新一下id2
        id2entity = {v: k for k, v in entity2id.items()}
        id2relation = {v: k for k, v in relation2id.items()}
        print('For dataset {}_v{}, there are {} original train triplets, {} emerging train triplets.'.format(params.dataset, v, len(original_train_triplets), len(emerging_train_triplets)))

        # 筛选所有u2s数据
        u2s_triplets = screen_u2s_triplets(all_triplets, original_entity_set, emerging_entity_set)
        # 确保所有u2s数据中的relation是已知的
        u2s_triplets = np.array([triplet for triplet in u2s_triplets if triplet[1] in original_relation_set])
        print("There are {} triplets as the u2s triplets".format(len(u2s_triplets)))

        train_triplets = np.array(original_train_triplets + emerging_train_triplets)
        np.random.shuffle(train_triplets)
        print("There are {} triplets as the all train triplets.".format(len(train_triplets)))
        
        # v1 对应构建 EQ, enclosing links 和 bridging links 比例为1:1
        # v2 对应构建 MB, enclosing links 和 bridging links 比例为1:2
        # v3 对应构建 ME, enclosing links 和 bridging links 比例为2:1
        if v == 1:
            num_bri = len(emerging_test_triplets)
            l = "EQ"
        elif v == 2:
            num_bri = int(len(emerging_test_triplets) * 2)
            l = "MB"
        else:
            num_bri = int(len(emerging_test_triplets) / 2)
            l = "ME"
        
        if num_bri > len(u2s_triplets):
            num_bri = len(u2s_triplets)
        # 采样    
        bri_idx = np.random.choice(np.array(range(len(u2s_triplets))), num_bri, False)
        bri_triplets = u2s_triplets[bri_idx]
        np.random.shuffle(bri_triplets)
        print("Sample {} triplets as bridge triplets.\n".format(len(valid_triplets)))

        # 写原始数据
        data_path = './data/{}_{}'.format(params.dataset, l)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        write_triplets(os.path.join(data_path, 'train.txt'), original_train_triplets, id2entity, id2relation)
        write_triplets(os.path.join(data_path, 'valid.txt'), original_valid_triplets, id2entity, id2relation)
        
        # 写enc数据
        data_path_enc = './data/{}_{}_enc'.format(params.dataset, l)
        if not os.path.exists(data_path_enc):
            os.makedirs(data_path_enc)
        write_triplets(os.path.join(data_path_enc, 'train.txt'), emerging_train_triplets, id2entity, id2relation)
        write_triplets(os.path.join(data_path_enc, 'test.txt'), emerging_test_triplets, id2entity, id2relation)

        # 写bri数据
        data_path_bri = './data/{}_{}_bri'.format(params.dataset, l)
        if not os.path.exists(data_path_bri):
            os.makedirs(data_path_bri)
        write_triplets(os.path.join(data_path_bri, 'train.txt'), train_triplets, id2entity, id2relation)
        write_triplets(os.path.join(data_path_bri, 'test.txt'), bri_triplets, id2entity, id2relation)

        # 写mix数据
        data_path_mix = './data/{}_{}_mix'.format(params.dataset, l)
        mix_triplets = np.concatenate((bri_triplets, emerging_test_triplets))
        np.random.shuffle(mix_triplets)
        if not os.path.exists(data_path_mix):
            os.makedirs(data_path_mix)
        write_triplets(os.path.join(data_path_mix, 'train.txt'), train_triplets, id2entity, id2relation)
        write_triplets(os.path.join(data_path_mix, 'test.txt'), mix_triplets, id2entity, id2relation)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data_generate')

    # 数据集设置
    parser.add_argument('--dataset', '-d', type=str,
                        help='用哪个数据集')
    params = parser.parse_args()

    main(params) 