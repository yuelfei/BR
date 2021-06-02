'''
@File name: dafine_parameter.py
@Author: Yuefei Wu
@Version: 1.0
@Creation time: 2019/6/8 - 13:14
@Describetion: 
'''
import tensorflow as tf
import os


flags = tf.app.flags
flags.DEFINE_boolean("clean",       True,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Wither train the model")
flags.DEFINE_boolean("test",       True,      "Wither test with saved model")

flags.DEFINE_boolean('binary_model', False,'binary or multiple')
flags.DEFINE_integer("class_num",   8,      "max_entity_length")

flags.DEFINE_string("ckpt_path",    "save_model",      "Path to save model")

flags.DEFINE_string("pad_emb",    "./data_emb/pad_emb",      "asd")
flags.DEFINE_integer("bert_emb_dim",    768,         "embedding dim")

flags.DEFINE_integer("batch_size",   20,         "batch size")
flags.DEFINE_integer("epoch",    40,         "epoch")
flags.DEFINE_integer( 'num_preprocessing_threads', 4,'The number of threads used to create the batches.')
flags.DEFINE_integer("seq_length",    50,         "seq_length per sentence")
flags.DEFINE_integer("emb_dim",    100,         "embedding dim")
flags.DEFINE_string("optimizer",    "adam",      "Optimizer for training ")
flags.DEFINE_integer("clip",    5,      "Clip for training")
flags.DEFINE_float("dropout",    0.2,         "dropout")
flags.DEFINE_float('momentum', 0.9,'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

flags.DEFINE_float('learning_rate', 0.001, 'Learning_rate.')
flags.DEFINE_float('select_threshold', 0.50, 'Alpha pdarameter in the loss function.')
flags.DEFINE_float('loss_alpha', 1., 'Alpha parameter in the loss function.')
flags.DEFINE_float('negative_ratio', 5., 'Negative ratio in the loss function.')
flags.DEFINE_float('match_threshold', 0.7, 'Matching threshold in the loss function.')
flags.DEFINE_float('nms_threshold',0.65, 'Matching threshold in the loss function.')
flags.DEFINE_integer("box_num",    6,      "box_num")
flags.DEFINE_integer("max_entity_length",    6,      "max_entity_length")

flags.DEFINE_string("train_log",    os.path.join("log", "train_log"),   "Path for train_log file")

flags.DEFINE_string("train_file", os.path.join("data", "Train_json"), "Path for train data")
flags.DEFINE_string("dev_file", os.path.join("data", "Dev_json"), "Path for dev data")
flags.DEFINE_string("test_file", os.path.join("data", "Test_json"), "Path for test data")

flags.DEFINE_string("Debug_file",    os.path.join("data", "Debug_json"),   "Path for debug data")

flags.DEFINE_boolean('pre_emb', True,'Use CPUs to deploy clones.')
flags.DEFINE_string("emb_file",    os.path.join("wiki_100.utf8"),   "Path for test data")



FLAGS = tf.app.flags.FLAGS



 

