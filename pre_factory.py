'''
@File name: pre_factory.py
@Author: Yuefei Wu
@Version: 1.0
@Creation time: 2019/6/8 - 19:31
@Describetion:
'''

import json
import numpy as np
import pickle
import math
import tensorflow as tf
from define_parameter import FLAGS

seq_length=FLAGS.seq_length



def padding_E2E_bert(left_list):
    pad_embedding=pickle.load(open("./data_emb/pad_emb","rb"))

    left_temp=left_list
    length=left_temp.shape[0]

    if (length == FLAGS.seq_length):
        pass
    if (length > FLAGS.seq_length):
        left_temp=left_temp[:FLAGS.seq_length]
    if (length < FLAGS.seq_length):
        pad_array=np.reshape(np.tile(pad_embedding,(FLAGS.seq_length-length)),(-1,pad_embedding.shape[-1]))
        left_temp=np.concatenate([left_temp,pad_array],axis=0)
    result_reshape=np.reshape(np.concatenate(left_temp,axis=0),(-1,FLAGS.bert_emb_dim))
    return result_reshape

def generate_new_emb_two(d, e):
    new_emb = []

    for d_node, e_node in zip(d, e):
        temp = np.concatenate([d_node, e_node], axis=1)
        new_emb.append(temp)

    sdf = np.reshape(new_emb, [-1, FLAGS.seq_length, 768*2])

    return sdf

def generate_new_emb(d, e,f):
    new_emb = []

    for d_node, e_node,f_node in zip(d, e,f):
        temp = np.concatenate([d_node, e_node,f_node], axis=1)
        new_emb.append(temp)

    sdf = np.reshape(new_emb, [-1, FLAGS.seq_length, 768*3])

    return sdf

def convert_emb(path):

    test = pickle.load(open(path, "rb"))

    test_sentence = []
    test_emb = []

    for node in test:
        emb = node[-1]

        temp = padding_E2E_bert(emb)

        test_emb.append(temp)

    test_emb_array = np.array(test_emb)
    return test_emb_array

def convert_emb_new(test):

    test_emb = []

    for node in test:
        emb = node
        temp = padding_E2E_bert(emb)
        test_emb.append(temp)

    test_emb_array = np.array(test_emb)
    return test_emb_array

def paddding(seqence):

    if(len(seqence)>seq_length):
        seqence=seqence[:seq_length]
    if (len(seqence) < seq_length):
        seqence = seqence+(seq_length-len(seqence))*[0]
    return seqence

def relative_coordinate(index):
    relative=[]
    for index_node in index:
        aa=[node/seq_length for node in index_node]
        relative.append(aa)
    return relative

def label_to_index(label):
    label_to_index = {
        "Background": 0,
        "VEH": 1,
        "LOC": 2,
        "WEA": 3,
        "GPE": 4,
        "PER": 5,
        "ORG": 6,
        "FAC": 7,
    }
    new_label_index=[]
    for node in label:
        new_label_index.append(label_to_index[node])
    return new_label_index


def limit_length(index, label):
    new_index = []
    new_label = []

    for index_ndoe, label_node in zip(index, label):
        if (index_ndoe[-1] - index_ndoe[0] <=FLAGS.max_entity_length)and(index_ndoe[-1]<FLAGS.seq_length):
            new_index.append(index_ndoe)
            new_label.append(label_node)
        else:
            continue
    return new_index, new_label


def load_data(file_path):

    sentence=[]
    total_windeows=[]
    label=[]

    linelist=open(file_path,"r",encoding="utf-8").readlines()

    for node in linelist:
        node_dict = json.loads(node)
        if (len(node_dict["index"]) == 0):
            continue
        sentence_temp=node_dict["sentence"]

        index = [[node[0],node[-1]+1] for node in node_dict["index"]]
        label_text = node_dict["type"]

        new_index,new_label=limit_length(index,label_text)
        # new_index=index
        # new_label=label_text
        if(len(new_label)==0):
            continue
        # new_index=index
        # new_label=label_text
        windows = relative_coordinate(new_index)
        label_index = label_to_index(new_label)

        sentence.append(sentence_temp)
        total_windeows.append(windows)

        label.append(label_index)

    return sentence,total_windeows,label

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def char_mapping(sentence):

    char=""
    for node in sentence:
        char+=node
    item_list=list(char)
    dico={}
    for item in item_list:
        if item not in dico:
            dico[item] = 1
        else:
            dico[item] += 1

    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (len(dico), sum(len(x) for x in item_list)))
    return id_to_char,char_to_id

def prepare_dataset(sentences, char_to_id):

    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[w if w in char_to_id else '<UNK>'] for w in string]
        chars=paddding(chars)
        data.append(chars)
    return data


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset,
                         dtype=np.float32):

    x = np.mgrid[0:feat_shape]
    x = (x.astype(dtype) + offset) * step / img_shape
    x = np.expand_dims(x, axis=-1)
    num_anchors = FLAGS.box_num
    w = np.zeros((num_anchors, ), dtype=dtype)
    w[0] = sizes[0]
    w[1] = sizes[1]
    w[2] = sizes[2]
    w[3] = sizes[3]
    w[4] = sizes[4]
    w[5] = sizes[5]
    return x,w

def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset,
                           dtype=np.float32):
    """
    :param img_shape:
    :param layers_shape:
    :param anchor_sizes:
    :param anchor_ratios:
    :param anchor_steps:
    :param offset:
    :param dtype:
    :return:
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios,
                                             anchor_steps,
                                             offset[i], dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors



def get_jaccard(x,y,box):
    int_x=np.maximum(x,box[0])
    int_y=np.minimum(y,box[1])
    inter_vol=np.maximum(int_y-int_x,0)
    union_vol=y-x-inter_vol+box[1]-box[0]
    jaccard=inter_vol/union_vol
    return jaccard

def get_max_with_threshold(jaccard,threshold):
    jaccard=np.where(jaccard <threshold, 0, jaccard)
    result=np.argmax(jaccard,axis=0)
    new_jaccard=np.zeros(jaccard.shape)
    for i ,index in enumerate(result):
        new_jaccard[index][i]=jaccard[index][i]
    final=np.where(new_jaccard==0,jaccard,new_jaccard)
    return final

def get_max(jaccard):
    result=np.argmax(jaccard,axis=0)
    new_jaccard=np.zeros(jaccard.shape)
    for i ,index in enumerate(result):
        new_jaccard[index][i]=jaccard[index][i]
    return new_jaccard

def get_max_threshols(jaccard):
    result=np.argmax(jaccard,axis=0)
    new_jaccard=np.zeros(jaccard.shape)
    for i ,index in enumerate(result):
        temp=jaccard[index][i]
        if(temp>=0.5):
            new_jaccard[index][i]=jaccard[index][i]
    return new_jaccard

def get_only_one(jaccard):
    result=np.argwhere(jaccard>=0.9)
    new_jaccard=np.zeros(jaccard.shape)
    for index in result:
        i=index[0]
        j=index[1]
        new_jaccard[i][j]=jaccard[i][j]
    return new_jaccard

def encode_box(anchor_layer,windows,label):
    anchor_box = anchor_layer[0]
    width = anchor_layer[1] / FLAGS.seq_length
    xref=anchor_box
    wref=width
    x = anchor_box - width / 2
    x = np.round(x,2)
    y = anchor_box + width / 2
    y = np.round(y, 2)

    temp_labels = []
    temp_localizations = []
    temp_scores = []

    for windows_node, label_node in zip(windows, label):
        feat_labels = np.zeros(anchor_box.shape)
        feat_scores = np.zeros(anchor_box.shape)

        feat_x = np.zeros(anchor_box.shape)
        feat_y = np.ones(anchor_box.shape)
        for win_node, lab_node in zip(windows_node, label_node):
            jaccard = get_jaccard(x, y, win_node)
            jaccard=get_max(jaccard)
            mask = np.greater(jaccard, feat_scores)
            mask = np.logical_and(mask, feat_scores > -0.5)
            imask = mask.astype(np.int64)
            fmask = mask.astype(np.float32)

            feat_labels = imask * lab_node + (1 - imask) * feat_labels
            feat_scores = np.where(mask, jaccard, feat_scores)

            feat_x = fmask * win_node[0] + (1 - fmask) * feat_x
            feat_y = fmask * win_node[1] + (1 - fmask) * feat_y

        feat_cx=(feat_x+feat_y)/2
        feat_w=feat_y-feat_x

        feat_tx=(feat_cx-xref)/wref
        feat_tw=np.log(feat_w/wref)

        temp_labels.append(feat_labels)
        temp_localizations.append(np.stack([feat_tx,feat_tw],axis=-1))
        temp_scores.append(feat_scores)

    a=np.array(temp_labels)
    b=np.array(temp_localizations)
    c=np.array(temp_scores)

    return np.array(temp_labels),np.array(temp_localizations),np.array(temp_scores)

def get_encode_box(anchors,windows, label):
    target_labels = []
    target_localizations = []
    target_scores = []

    for i, anchor_layer in enumerate(anchors):
        glable, glocation, gscore = encode_box(anchor_layer, windows, label)

        target_labels.append(glable)
        target_localizations.append(glocation)
        target_scores.append(gscore)

    return target_labels,target_localizations,target_scores





