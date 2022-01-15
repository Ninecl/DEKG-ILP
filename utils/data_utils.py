import os
import pdb
import logging
import numpy as np
from scipy.sparse import csc_matrix


def process_files(files, saved_relation2id=None):
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}
    all_triplets = []

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.readlines()]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])
            
            # store all triplets
            all_triplets.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to each relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    # construct relation-component tables, only from train data
    num_entity = len(entity2id)
    num_relation = len(relation2id)
    relation_component_table = np.zeros((num_entity, num_relation*2))
    for h, t, r in triplets['train']:
        relation_component_table[h][r] += 1
        relation_component_table[t][r+num_relation] += 1


    return adj_list, relation_component_table, triplets, all_triplets, entity2id, relation2id, id2entity, id2relation


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def load_raw_data(file_path):

    logging.info("\nLoad raw data from {}".format(file_path))

    entity2id_path = os.path.join(file_path, 'entity2id.txt')
    relation2id_path = os.path.join(file_path, 'relation2id.txt')

    # directly read 2id
    if os.path.exists(entity2id_path) and os.path.exists(relation2id_path):
        print("There is existing entity2id and relation2id, loading...")
        with open(entity2id_path, 'r') as f:
            entity2id = dict()
            for line in f.readlines():
                entity, eid = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(relation2id_path, 'r') as f:
            relation2id = dict()
            for line in f.readlines():
                relation, rid = line.strip().split('\t')
                relation2id[relation] = int(rid)

        train_triplets, entity2id, relation2id = read_triplets2id(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
        valid_triplets, entity2id, relation2id = read_triplets2id(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
        test_triplets, entity2id, relation2id = read_triplets2id(os.path.join(file_path, 'test.txt'), entity2id, relation2id)
    
    # re-generate 2id
    else:
        print("There is no entity2id and relation2id, generating...")
        train_triplets = read_triplets(os.path.join(file_path, 'train.txt'))
        valid_triplets = read_triplets(os.path.join(file_path, 'valid.txt'))
        test_triplets = read_triplets(os.path.join(file_path, 'test.txt'))
        all_triplets = train_triplets + valid_triplets + test_triplets

        entity2id = dict()
        relation2id = dict()

        entity_cnt = 0
        relation_cnt = 0

        for triplet in all_triplets:
            h, r, t = triplet
            if h not in entity2id:
                entity2id[h] = entity_cnt
                entity_cnt += 1
            if r not in relation2id:
                relation2id[r] = relation_cnt
                relation_cnt += 1
            if t not in entity2id:
                entity2id[t] = entity_cnt
                entity_cnt += 1
        
        train_triplets = [[entity2id[triplet[0]], relation2id[triplet[1]], entity2id[triplet[2]]] for triplet in train_triplets]
        valid_triplets = [[entity2id[triplet[0]], relation2id[triplet[1]], entity2id[triplet[2]]] for triplet in valid_triplets]
        test_triplets = [[entity2id[triplet[0]], relation2id[triplet[1]], entity2id[triplet[2]]] for triplet in test_triplets]

        # save entity2id and relation2id
        with open(entity2id_path, 'w') as f:
            for k, v in entity2id.items():
                f.write('{}\t{}\n'.format(k, v))
        
        with open(relation2id_path, 'w') as f:
            for k, v in relation2id.items():
                f.write('{}\t{}\n'.format(k, v))
        

    logging.info('num_entity: {}'.format(len(entity2id)))
    logging.info('num_relation: {}'.format(len(relation2id)))
    logging.info('num_train_triples: {}'.format(len(train_triplets)))
    logging.info('num_valid_triples: {}'.format(len(valid_triplets)))
    logging.info('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets


def read_triplets2id(file_path, entity2id, relation2id):
    triplets = []
    entity_cnt = len(entity2id)
    relation_cnt = len(relation2id)

    with open(file_path, 'r') as f:
        for line in f.readlines():
            head, relation, tail = line.strip().split('\t')
            if head not in entity2id:
                entity2id[head] = entity_cnt
                entity_cnt += 1
            if tail not in entity2id:
                entity2id[tail] = relation_cnt
                relation_cnt += 1
            if relation not in relation2id:
                relation2id[relation] = entity_cnt
                entity_cnt += 1
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return triplets, entity2id, relation2id


def read_triplets(file_path):
    triplets = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            head, relation, tail = line.strip().split('\t')
            triplets.append([head, relation, tail])

    return triplets


def count_entity_relation_set(triplets, triplet_type='hrt'):
    entity_set = set()
    relation_set = set()

    if triplet_type == 'hrt':
        for h, r, t in triplets:
            entity_set.add(h)
            entity_set.add(t)
            relation_set.add(r)
    
    elif triplet_type == 'htr':
        for h, t, r in triplets:
            entity_set.add(h)
            entity_set.add(t)
            relation_set.add(r)
    
    else:
        raise('Wrong triplet type "{}"'.format(triplet_type))
    
    return entity_set, relation_set


def write_triplets(file_path, triplets, id2entity, id2relation):

    with open(file_path, 'w') as f:
        for triplet in triplets:
            h, r, t = triplet
            f.write('{}\t{}\t{}\n'.format(id2entity[h], id2relation[r], id2entity[t]))
        f.close()