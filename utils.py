'''
@File name: common.py
@Author:
@Version: 1.0
@Creation time: 2019/6/7 - 20:30
@Describetion:
'''

import shutil
import logging, os, re, math
import numpy as np
import random
import datetime
import tensorflow as tf
from define_parameter import *
import codecs
import numpy as np
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
import pickle


#模型日志
def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file,encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def get_time():
    now_time=datetime.datetime.now()
    year=now_time.year
    month=now_time.month
    day=now_time.day
    hour=now_time.hour
    minute=now_time.minute
    return str(year)+"_"+str(month)+"_"+str(day)+"_"+str(hour)+"_"+str(minute)

def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")

def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


@add_arg_scope
def l2_normalization(
        inputs,
        scaling=False,
        scale_initializer=init_ops.ones_initializer(),
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        data_format='NHWC',
        trainable=True,
        scope=None):
    """Implement L2 normalization on every feature (i.e. spatial normalization).

    Should be extended in some near future to other dimensions, providing a more
    flexible normalization framework.

    Args:
      inputs: a 4-D tensor with dimensions [batch_size, height, width, channels].
      scaling: whether or not to add a post scaling operation along the dimensions
        which have been normalized.
      scale_initializer: An initializer for the weights.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: optional list of collections for all the variables or
        a dictionary containing a different list of collection per variable.
      outputs_collections: collection to add the outputs.
      data_format:  NHWC or NCHW data format.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      scope: Optional scope for `variable_scope`.
    Returns:
      A `Tensor` representing the output of the operation.
    """

    with variable_scope.variable_scope(
            scope, 'L2Normalization', [inputs], reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        dtype = inputs.dtype.base_dtype
        if data_format == 'NHWC':
            # norm_dim = tf.range(1, inputs_rank-1)
            norm_dim = tf.range(inputs_rank-1, inputs_rank)
            params_shape = inputs_shape[-1:]
        elif data_format == 'NCHW':
            # norm_dim = tf.range(2, inputs_rank)
            norm_dim = tf.range(1, 2)
            params_shape = (inputs_shape[1])

        # Normalize along spatial dimensions.
        outputs = nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)
        # Additional scaling.
        if scaling:
            scale_collections = utils.get_variable_collections(
                variables_collections, 'scale')
            scale = variables.model_variable('gamma',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=scale_initializer,
                                             collections=scale_collections,
                                             trainable=trainable)
            if data_format == 'NHWC':
                outputs = tf.multiply(outputs, scale)
            elif data_format == 'NCHW':
                scale = tf.expand_dims(scale, axis=-1)
                scale = tf.expand_dims(scale, axis=-1)
                outputs = tf.multiply(outputs, scale)
                # outputs = tf.transpose(outputs, perm=(0, 2, 3, 1))

        return utils.collect_named_outputs(outputs_collections,sc.original_name_scope, outputs)


def clean(cpkt_path):
    if os.path.isdir(cpkt_path):
        shutil.rmtree(cpkt_path)

def print_configuration_op(FLAGS,logger) -> object:
    logger.info('\n------------------------------实验超参数日志------------------------------')
    for name, value in FLAGS.__flags.items():
        type_=str(type(value.value))
        value_ = str(value.value)
        help=value.help
        logger.info('{:<25} \t\t {:<25} \t\t {:<20} \t\t {:<25}'.format(name, value_,type_,help))
    logger.info('------------------------------超参数输出完毕------------------------------\n')


def get_config() -> object:
    net_config={
      "img_shape": FLAGS.seq_length,
      "num_classes": FLAGS.box_num,
      "feat_layers": ["block1","block2"],
      "feat_shapes": [50,50],
      "anchor_size_bounds": [0.02, 0.9],
      "anchor_sizes": [1,1],
      "anchor_ratios": [1,1],
      "anchor_steps": [1,1],
      "anchor_offset": 0.5,
      "normalizations": [20, -1, -1, -1, -1, -1, -1],
      "prior_scaling": [0.1, 0.1, 0.2, 0.2],
    }
    return net_config


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[re.sub('\d', '0', word.lower())]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with ''pretrained embeddings.' % (c_found + c_lower + c_zeros, n_words,100. * (c_found + c_lower + c_zeros) / n_words))
    print('%i found directly, %i after lowercasing, ''%i after lowercasing + zero.' % (c_found, c_lower, c_zeros))
    return new_weights


def create_model(session, Model_class, path, load_vec, config, logger):
    '''
    :param session:
    :param Model_class:
    :param path:
    :param load_vec:
    :param config:
    :param logger:
    :return:
    '''
    # create model, reuse parameters if exists
    model = Model_class(config)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

class BatchManager(object):

    def __init__(self, sentence, target_labels,target_localizations,target_scores,batch_size):
        self.batch_size=batch_size
        self.batch_num= int(math.floor(len(sentence) /batch_size))

        self.sentence=self.data_batch(sentence,self.batch_num,self.batch_size)
        self.target_labels=self.layer_batch(target_labels)
        self.target_localizations=self.layer_batch(target_localizations)
        self.target_scores=self.layer_batch(target_scores)

    def data_batch(self, data,num_batch, batch_size):
        data=np.array(data)
        batch_data = list()
        for i in range(num_batch):
            temp=data[i*batch_size : (i+1)*batch_size]
            batch_data.append(temp)
        return np.array(batch_data)

    def layer_batch(self,data):
        layer_feature=[]
        for i,data_node in enumerate(data):
            layer_feature.append(self.data_batch(data_node,self.batch_num, self.batch_size))
        return layer_feature

def binary_label(target_labels):
    binary_class = {'0': [1, 0], '1': [0, 1]}
    result_list = []
    for i, node in enumerate(target_labels):
        shape = node.shape
        new_ndoe = node.ravel()
        a = new_ndoe.tolist()
        result = []
        for node in a:
            if (node in [1, 2, 3, 4, 5, 6, 7]):
                result.append([0, 1])  # 正例
            else:
                result.append([1, 0])  # 负例
        new_result = np.asanyarray(result)
        final_result = np.reshape(new_result, (shape[0], shape[1], shape[2], 2))
        result_list.append(final_result)
    return result_list


def sparse_binary_label(target_labels):
    binary_class = {'0': [1, 0], '1': [0, 1]}
    result_list = []
    for i, node in enumerate(target_labels):
        node[node == 0] = 0
        node[node == 1] = 1
        node[node == 2] = 1
        node[node == 3] = 1
        node[node == 4] = 1
        node[node == 5] = 1
        node[node == 6] = 1
        node[node == 7] = 1
        node[node == 8] = 1
        node[node == 9] = 1
        node[node == 10] = 1
        result_list.append(node)
    return result_list

def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("fianl model saved")


def ssd_bboxes_decode(feat_localizations,anchor_bboxes):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Return:
      numpy array Nx4: ymin, xmin, ymax, xmax
    """
    # Reshape for easier broadcasting.
    l_shape = feat_localizations.shape
    feat_localizations = np.reshape(feat_localizations,(-1, l_shape[-2], l_shape[-1]))
    xref, wref = anchor_bboxes
    wref=wref/FLAGS.seq_length
    xref = np.reshape(xref, [-1, 1])

    # Compute center, height and width
    cdc=feat_localizations[:, :, 0]
    cx = feat_localizations[:, :, 0] * wref  + xref
    aa=feat_localizations[:,:,1]
    w = wref * np.exp(feat_localizations[:, :, 1])
    # bboxes: x, y.
    bboxes = np.zeros_like(feat_localizations)
    bboxes[:, :, 0] = cx - w / 2.
    bboxes[:, :, 1] = cx + w / 2.
    # Back to original shape.
    bboxes = np.reshape(bboxes, l_shape)
    return bboxes


def ssd_bboxes_select_layer(predictions_layer,
                            localizations_layer,
                            anchors_layer,
                            select_threshold=FLAGS.select_threshold,
                            img_shape=(300, 300),
                            num_classes=21,
                            decode=True):
    """Extract classes, scores and bounding boxes from features in one layer.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    # First decode localizations features if necessary.
    if decode:
        localizations_layer = ssd_bboxes_decode(localizations_layer, anchors_layer)

    # Reshape features to: Batches x N x N_labels | 4.
    p_shape = predictions_layer.shape
    batch_size = p_shape[0] if len(p_shape) == 4 else 1
    predictions_layer = np.reshape(predictions_layer,
                                   (batch_size, -1, p_shape[-1]))
    l_shape = localizations_layer.shape
    localizations_layer = np.reshape(localizations_layer,
                                     (batch_size, -1, l_shape[-1]))

    # Boxes selection: use threshold or score > no-label criteria.
    if select_threshold is None or select_threshold == 0:
        # Class prediction and scores: assign 0. to 0-class
        classes = np.argmax(predictions_layer, axis=2)
        scores = np.amax(predictions_layer, axis=2)
        mask = (classes > 0)
        classes = classes[mask]
        scores = scores[mask]
        bboxes = localizations_layer[mask]
    else:
        sub_predictions = predictions_layer[:, :, 1:]
        idxes = np.where(sub_predictions > select_threshold)
        classes = idxes[-1]+1
        scores = sub_predictions[idxes]
        bboxes = localizations_layer[idxes[:-1]]

    return classes, scores, bboxes



def ssd_bboxes_select(predictions_net,
                      localizations_net,
                      anchors_net,
                      select_threshold=FLAGS.select_threshold,
                      img_shape=(300, 300),
                      num_classes=21,
                      decode=True):
    """

    :param predictions_net:
    :param localizations_net:
    :param anchors_net:
    :param select_threshold:
    :param img_shape:
    :param num_classes:
    :param decode:
    :return:
    """
    l_classes = []
    l_scores = []
    l_bboxes = []
    for i in range(len(predictions_net)):
        classes, scores, bboxes = ssd_bboxes_select_layer(
            predictions_net[i], localizations_net[i], anchors_net[i],
            select_threshold, img_shape, num_classes, decode)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)

    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    return classes, scores, bboxes




def bboxes_clip(bbox_ref, bboxes):
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.minimum(bboxes[1], bbox_ref[1])
    bboxes = np.transpose(bboxes)
    return bboxes

def bboxes_sort(classes, scores, bboxes, top_k=400):

    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes


def bboxes_jaccard(bboxes1, bboxes2):
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_x = np.maximum(bboxes1[0], bboxes2[0])
    int_y = np.minimum(bboxes1[1], bboxes2[1])

    inter_vol = np.maximum(int_y - int_x, 0)
    union_vol = bboxes1[1] - bboxes1[0] - inter_vol + bboxes2[1] - bboxes2[0]
    jaccard = inter_vol / union_vol
    return jaccard



def bboxes_nms(classes, scores, bboxes, nms_threshold=FLAGS.nms_threshold):
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            keep_overlap = np.logical_not(overlap >= nms_threshold)
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]

def eval_on_test(predictions_v, localisations_v,anchors):

    rclasses, rscores, rbboxes=ssd_bboxes_select(predictions_v, localisations_v,anchors)
    sen_box=np.array([0,1.])
    rbboxes = bboxes_clip(sen_box, rbboxes)
    rclasses, rscores, rbboxes=bboxes_sort(rclasses, rscores,rbboxes, top_k=400)
    rclasses, rscores, rbboxes = bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=FLAGS.nms_threshold)

    return rclasses, rscores, rbboxes


def eval_main(predictions_v, localisations_v,anchors):

    rclasses_total = []
    rscores_total = []
    rbboxes_total = []

    for predictions_0,predictions_1,  localisations_0,localisations_1 in zip(predictions_v[0],predictions_v[1], localisations_v[0],localisations_v[1]):
        predictions_s = [predictions_0,predictions_1]
        localisations_s = [localisations_0,localisations_1]

        rclasses, rscores, rbboxes = eval_on_test(predictions_s, localisations_s, anchors)
        rclasses_total.append(rclasses)
        rscores_total.append(rscores)
        rbboxes_total.append(rbboxes)
    return rclasses_total,rscores_total,rbboxes_total

def eval_main_directly(predictions_v, localisations_v,anchors):

    pr_predictions = []
    pr_localizations = []

    for predictions_0,predictions_1,  localisations_0,localisations_1 in zip(predictions_v[0],predictions_v[1], localisations_v[0],localisations_v[1]):
        predictions_s = [predictions_0,predictions_1]
        localisations_s = [localisations_0,localisations_1]

        predictions, localizations = eval_on_test_derectly(predictions_s, localisations_s, anchors)
        pr_predictions.append(predictions)
        pr_localizations.append(localizations)


    return pr_predictions,pr_localizations

def evaluate(sess, model, data, anchors,logger,test_sentence, test_windows,test_label,mode):
    '''
    :param sess:
    :param model:
    :param data:
    :param anchors:
    :param logger:
    :param test_sentence:
    :param test_windows:
    :return:
    '''

    localisations_result,classification_result = model.evaluate(sess,data,anchors)


    location = []
    classification = []

    for node in localisations_result:
        location.extend(node)

    for cla_node in classification_result:
        classification.extend(cla_node)

    length = len(location)
    test_bbox = test_windows[:length]
    test_array = [np.array(node) for node in test_bbox]

    length = len(location)
    test_label = test_label[:length]
    label_array = [np.array(node) for node in test_label]

    found_num = 0
    real_num = 0
    correct_num = 0

    correct_pickl=[]

    a=0
    b=0

    for label_node, real_node in zip(label_array, test_array):
        real = real_node * FLAGS.seq_length
        real = real.astype(np.int32)
        real = real.tolist()
        a += len(real)

        real_label=label_node.astype(np.int32)
        real_label = real_label.tolist()
        b += len(real_label)


    for loca_node, real_node in zip(location, test_array):
        loca = np.around(loca_node * FLAGS.seq_length)
        loca = loca.astype(np.int32)
        loca=loca.tolist()
        found_num += len(loca)


        real = real_node * FLAGS.seq_length
        real = real.astype(np.int32)
        real = real.tolist()
        real_num += len(real)


        correct = [x for x in real if x in loca]
        correct_pickl.append(correct)
        correct_num += len(correct)


    P = correct_num / found_num
    R = correct_num / real_num
    F = (2 * P * R ) / (P + R)

    logger.info("********  模型在"+mode+"集上进行evaluate")
    logger.info("********  测试集中的正例：{}".format(real_num))
    logger.info("********  预测的正例：{}".format(found_num))
    logger.info("********  正确预测的正例：{}".format(correct_num))

    logger.info("********  Precision：{:>.4f}".format(P))
    logger.info("********  Recall：{:>.4f}".format(R))
    logger.info("********  F-1 value：{:>.4f}".format(F))


    return [P,R,F]

def evaluate_mul(sess, model, data, anchors, logger, test_sentence, test_windows, test_label,mode):
    '''
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
    '''


    localisations_result, classification_result = model.evaluate(sess, data, anchors)

    location = []
    classification = []

    for node in localisations_result:
        location.extend(node)

    for cla_node in classification_result:
        classification.extend(cla_node)

    length = len(location)
    test_bbox = test_windows[:length]
    test_array = [np.array(node) for node in test_bbox]

    test_bb_label = test_label[:length]
    test_label_array = [node for node in test_bb_label]

    true_index_list = [1, 2, 3, 4, 5, 6, 7]
    type_list = ["00","VEH", "LOC", "WEA", "GPE", "PER", "ORG", "FAC"]

    found_num = {"VEH": 0, "LOC": 0, "WEA": 0, "GPE": 0, "PER": 0, "ORG": 0, "FAC": 0}
    real_num = {"VEH": 0, "LOC": 0, "WEA": 0, "GPE": 0, "PER": 0, "ORG": 0, "FAC": 0}
    correct_num = {"VEH": 0, "LOC": 0, "WEA": 0, "GPE": 0, "PER": 0, "ORG": 0, "FAC": 0}

    correct_pickl = []

    for loca_node, real_loca_node, cla_node, real_loca_label in zip(location, test_array, classification, test_label_array):

        loca = np.around(loca_node * FLAGS.seq_length)
        loca = loca.astype(np.int32)
        loca = loca.tolist()

        def delete_repeat(loca,cla_node):
            new_loca=[]
            new_cla_node=[]
            for i,de_loca_node in enumerate(loca):
                if(de_loca_node not in new_loca):
                    new_loca.append(de_loca_node)
                    new_cla_node.append(cla_node[i])
            return new_loca,new_cla_node

        loca,cla_node=delete_repeat(loca,cla_node)

        real = real_loca_node * FLAGS.seq_length
        real = real.astype(np.int32)
        real = real.tolist()

        for l_node in cla_node:
            found_num[type_list[l_node]] += 1

        for r_node in real_loca_label:
            aa=type_list[r_node]
            real_num[aa] += 1

        for ll_node, cc_node in zip(loca, cla_node):
            if (ll_node in real):
                aaa=real.index(ll_node)
                bb=real_loca_label[aaa]
                if (bb == cc_node):
                    correct_num[type_list[cc_node]] += 1

    total_found_num = {"VEH": 0, "LOC": 0, "WEA": 0, "GPE": 0, "PER": 0, "ORG": 0, "FAC": 0}
    total_real_num = {"VEH": 0, "LOC": 0, "WEA": 0, "GPE": 0, "PER": 0, "ORG": 0, "FAC": 0}
    total_correct_num = {"VEH": 0, "LOC": 0, "WEA": 0, "GPE": 0, "PER": 0, "ORG": 0, "FAC": 0}

    for found_num,real_num,correct_num in zip(found_num.items(),real_num.items(),correct_num.items()):

        Type=found_num[0]

        true_test_num=real_num[-1]
        pred_num=found_num[-1]
        true_pre_num=correct_num[-1]

        total_type=found_num[0]
        total_found_num[total_type]+=pred_num
        total_real_num[total_type]+=true_test_num
        total_correct_num[total_type]+=true_pre_num


        logger.info("-------------------------*{}*---------------------------------".format(Type))
        logger.info("|测试集中的正例: {}".format(true_test_num))
        logger.info("|预测结果中的正例:{}".format(pred_num))
        logger.info("|预测为正例的结果中真正的正例:{}".format(true_pre_num))
        P = 0 if pred_num == 0 else 100. * float(true_pre_num) / float(pred_num)
        R = 0 if true_pre_num == 0 else 100. * float(true_pre_num) / float(true_test_num)
        F = 0 if P + R == 0 else 2 * P * R / (P + R)
        logger.info("|Precision: {:>.4f} %".format(P))
        logger.info("|Recall: {:>.4f} %".format(R))
        logger.info("|F1: {:>.4f} %".format(F))
        logger.info("-------------------------*{}*---------------------------------\n".format(Type))


    total_true_test_num = sum(total_real_num.values())
    total_pred_num = sum(total_found_num.values())
    total_true_pre_num = sum(total_correct_num.values())

    logger.info("-------------------------------------------在{}集上的Total，Micro-average计算-------------------------------------------".format(mode))
    logger.info("-------------------------*Total Performance*---------------------------------")
    logger.info("|测试集中各类的正例总数: {}".format(total_true_test_num))
    logger.info("|预测结果中各类的正例总数:{}".format(total_pred_num))
    logger.info("|预测为正例的结果中各类真正的正例总数:{}".format(total_true_pre_num))
    P = 0 if total_pred_num == 0 else 100. * total_true_pre_num / total_pred_num
    R = 0 if total_true_pre_num == 0 else 100. * total_true_pre_num / total_true_test_num
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    logger.info("|Precision: {:>.4f} %".format(P))
    logger.info("|Recall: {:>.4f} %".format(R))
    logger.info("|F1: {:>.4f} %".format(F))


def backup_log():

    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S')
    source = "./log/train_log"
    target = "./log_backups/log_backup-" + str(now_time)

    shutil.copyfile(source, target)

