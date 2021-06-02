'''
@File name: model.py
@Author: Yuefei Wu
@Version: 1.0
@Creation time: 2019/6/7 - 20:33
@Describetion:
'''


import pickle
import tensorflow as tf
import numpy as np
import utils
conv1d=tf.nn.conv1d
slim = tf.contrib.slim
from define_parameter import *
import tf_common
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.rnn import CoupledInputForgetGateLSTMCell as LSTMCell
from tensorflow.contrib.layers import xavier_initializer
initializer=xavier_initializer()

class Model(object):
    def __init__(self,net_config):

        self.char_dim = FLAGS.emb_dim
        self.config=net_config
        self.initializer = initializers.xavier_initializer()
        self.global_step = tf.Variable(0, trainable=False)

        self.sentence = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.seq_length,768*2], name="EmbInputs")

        self.glabel_1 = tf.placeholder(dtype=tf.int64, shape=[FLAGS.batch_size, 50, FLAGS.box_num], name="glabel_1")
        self.glabel_2 = tf.placeholder(dtype=tf.int64, shape=[FLAGS.batch_size, 50, FLAGS.box_num], name="glabel_2")
        self.blabel=tf_common.gtensor_list([self.glabel_1,self.glabel_2])

        self.glocation_1 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 50, FLAGS.box_num, 2], name="glocaiton_1")
        self.glocation_2 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size,50, FLAGS.box_num, 2], name="glocaiton_2")
        self.blocation = tf_common.gtensor_list([self.glocation_1,self.glocation_2])

        self.gscore_1 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 50, FLAGS.box_num], name="gscore_1")
        self.gscore_2 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 50, FLAGS.box_num], name="gscore_2")
        self.bscore = tf_common.gtensor_list([self.gscore_1,self.gscore_2])

        self.embedding = self.sentence

        self.predictions, self.localisations, self.logits, end_points=self.build_net(self.embedding)

        self.total_loss=self.loss_layer(self.logits,self.localisations,self.blabel,self.blocation,self.bscore)

        with tf.variable_scope("optimizer"):
            optimizer = FLAGS.optimizer
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(FLAGS.learning_rate)
            else:
                raise KeyError

            grads_vars = self.opt.compute_gradients(self.total_loss)
            self.train_op = self.opt.apply_gradients(grads_vars, self.global_step,name="s_second")

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def ssd_multibox_layer_dense(self,inputs,
                           num_classes):
        """
        :param inputs:
        :param num_classes:
        :return:
        """
        net = inputs
        num_anchors = FLAGS.box_num  #
        num_loc_pred = num_anchors * 2
        loc_dense_input = tf.reshape(net, shape=[-1, 100 * 2])
        loc_output_1 = tf.layers.dense(loc_dense_input, 100)
        loc_output_2 = tf.layers.dense(loc_output_1, num_loc_pred)
        loc_pred = tf.reshape(loc_output_2, shape=[FLAGS.batch_size, FLAGS.seq_length,num_anchors, -1])


        num_cls_pred = num_anchors * num_classes
        cla_dense_input = tf.reshape(net, shape=[-1, 100 * 2])
        cla_output_1 = tf.layers.dense(cla_dense_input, 100)
        cla_output_2 = tf.layers.dense(cla_output_1, num_cls_pred)
        cls_pred = tf.reshape(cla_output_2, shape=[FLAGS.batch_size, FLAGS.seq_length, num_anchors, -1])

        return cls_pred, loc_pred

    def embedding_layer(self, char_inputs,name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """

        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(name="char_embedding",shape=[self.num_chars, self.char_dim],initializer=self.initializer)#初始化emd张量，随机赋值
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed



    def build_net(self,input,reuse=None):
        '''
        :param input: [batch_size,seq_length,emb_dim]
        :param reuse:
        :return:
        '''

        end_points = {}
        end_point = 'block1'
        with tf.variable_scope(end_point):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = LSTMCell(100, use_peepholes=True, initializer=initializer,state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"], lstm_cell["backward"],input, dtype=tf.float32,sequence_length=None)  # final_states表示的是该循环单元最后时刻的输出，outputs表示循环单元所有时刻的输出
            net = tf.concat(outputs, axis=2)
        end_points[end_point] = net


        end_point = 'block2'
        with tf.variable_scope(end_point):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = LSTMCell(100, use_peepholes=True, initializer=initializer,state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"], lstm_cell["backward"],input,dtype=tf.float32,sequence_length=None)  # final_states表示的是该循环单元最后时刻的输出，outputs表示循环单元所有时刻的输出
            net = tf.concat(outputs, axis=2)
        end_points[end_point] = net


        class_num = FLAGS.class_num
        predictions = []
        logits = []
        localisations = []

        for i, layer in enumerate(self.config["feat_layers"]):
            with tf.variable_scope(layer + '_box'):
                cls, loc = self.ssd_multibox_layer_dense(end_points[layer], class_num)
                predictions.append(tf.nn.softmax(cls))
                logits.append(cls)
                localisations.append(loc)

        return predictions, localisations, logits, end_points

    def loss_layer(self,logits, localisations,
                   gclasses, glocalisations, gscores,
                   match_threshold=FLAGS.match_threshold,
                   negative_ratio=FLAGS.negative_ratio,
                   alpha=1.,
                   label_smoothing=0.,
                   device='/cpu:0',
                   scope=None):
        with tf.name_scope(scope, 'ssd_losses'):
            lshape = tf_common.get_shape(logits[0], 5)
            num_classes = lshape[-1]
            batch_size = lshape[0]

            # Flatten out all vectors!
            flogits = []
            fgclasses = []
            fgscores = []
            flocalisations = []
            fglocalisations = []
            for i in range(len(logits)):
                flogits.append(tf.reshape(logits[i], [-1, num_classes]))
                fgclasses.append(tf.reshape(gclasses[i], [-1]))
                fgscores.append(tf.reshape(gscores[i], [-1]))
                flocalisations.append(tf.reshape(localisations[i], [-1, 2]))
                fglocalisations.append(tf.reshape(glocalisations[i], [-1, 2]))
            logits = tf.concat(flogits, axis=0)
            gclasses = tf.concat(fgclasses, axis=0)
            gscores = tf.concat(fgscores, axis=0)
            localisations = tf.concat(flocalisations, axis=0)
            glocalisations = tf.concat(fglocalisations, axis=0)
            dtype = logits.dtype

            pmask = gscores >= match_threshold
            fpmask = tf.cast(pmask, dtype)
            n_positives = tf.reduce_sum(fpmask)
            no_classes = tf.cast(pmask, tf.int32)
            predictions = slim.softmax(logits)
            nmask = tf.logical_and(tf.logical_not(pmask),gscores > -0.5)
            fnmask = tf.cast(nmask, dtype)
            nvalues = tf.where(nmask, predictions[:, 0],1. - fnmask)
            nvalues_flat = tf.reshape(nvalues, [-1])

            max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
            n_neg = tf.minimum(n_neg, max_neg_entries)

            val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
            max_hard_pred = -val[-1]
            nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
            fnmask = tf.cast(nmask, dtype)

            #cross-entropy loss.
            with tf.name_scope('cross_entropy_pos'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=gclasses)
                cross_entropy_pos_loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size,name='value')

            with tf.name_scope('cross_entropy_neg'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=no_classes)
                cross_entropy_neg_loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size,name='value')

            with tf.name_scope('localization'):
                weights = tf.expand_dims(alpha * fpmask, axis=-1)
                loss = tf_common.abs_smooth(localisations - glocalisations)
                localization_loss = tf.div(tf.reduce_sum(loss * weights), batch_size,name='value')

            sum_loss=(cross_entropy_pos_loss+cross_entropy_neg_loss+localization_loss)/3

            return sum_loss

    def create_feed_dict(self, is_train, batch):
        '''
        :param is_train: train or not
        :param batch: a list for batch_data
        :return:
        '''
        if is_train:
            sentence, labels_1, labels_2, localizations_1,localizations_2, scores_1,scores_2= batch
            feed_dict_train = {
                self.sentence: sentence,

                self.glabel_1: labels_1,
                self.glabel_2: labels_2,

                self.glocation_1:localizations_1,
                self.glocation_2:localizations_2,

                self.gscore_1:scores_1,
                self.gscore_2:scores_2,
            }
            return feed_dict_train
        else:
            sentence= batch
            feed_dict_test = {self.sentence: sentence}
            return feed_dict_test



    def run_step(self, sess, batch,is_train):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, total_loss, _ = sess.run([self.global_step, self.total_loss, self.train_op],feed_dict)
            return global_step, total_loss
        else:
            predictions_t, localisations_t, logits_t= sess.run([self.predictions, self.localisations, self.logits],feed_dict)
            return predictions_t, localisations_t, logits_t

    def evaluate(self,sess,data,anchors):
        localisations_result=[]
        logits_result=[]

        test_batch_sentence = data.sentence

        predictions_v_saved=[]
        localisations_v_saved=[]

        for sentence_node in test_batch_sentence:
            predictions_v, localisations_v, logits_v=self.run_step(sess,sentence_node,False)

            predictions_v_saved.append(predictions_v)
            localisations_v_saved.append(localisations_v)

            rclasses, rscores, rbboxes=utils.eval_main(predictions_v, localisations_v,anchors)
            localisations_result.append(rbboxes)
            logits_result.append(rclasses)

        return localisations_result,logits_result

    def evaluate_directly(self,sess,data,anchors):
        localisations_result=[]
        logits_result=[]

        test_batch_sentence = data.sentence

        for sentence_node in test_batch_sentence:
            predictions_v, localisations_v, logits_v=self.run_step(sess,sentence_node,False)
            pr_predictions,pr_localizations=utils.eval_main_directly(predictions_v, localisations_v,anchors)
            localisations_result.append(pr_localizations)
            logits_result.append(pr_predictions)

        return localisations_result,logits_result
