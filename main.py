'''
@File name: main.py
@Author: Yuefei Wu
@Version: 2.0
@Creation time: 2020/8/8 - 19:33
@Describetion:
'''

import tensorflow as tf
from define_parameter import *
from model import Model
from pre_factory import *
from utils import *
import tf_common
import datetime
slim = tf.contrib.slim
from deployment import model_deploy
input_shape=(FLAGS.seq_length,FLAGS.emb_dim)

def main():

    logger=get_logger(FLAGS.train_log)

    if FLAGS.clean:
        clean(FLAGS.ckpt_path)

    config=get_config()
    logger.info("******")
    logger.info("ACE中文,按照622划分数据")
    logger.info("******")
    print_configuration_op(FLAGS,logger)

    #从json file中提取数据,数据目录为
    tr_sentence, tr_windows, tr_label = load_data(FLAGS.train_file)
    new_dev_sentence, new_dev_windows, new_dev_label = load_data(FLAGS.dev_file)
    new_test_sentence, new_test_windows, new_test_label = load_data(FLAGS.test_file)

    '''
    ACE2005中文数据集
    '''
    logger.info("加载Bert-emb······")
    train_emb_out_base = convert_emb("data_emb/pre_trained_emb_out/train")
    dev_emb_out_base= convert_emb("data_emb/pre_trained_emb_out/dev")
    test_emb_out_base= convert_emb("data_emb/pre_trained_emb_out/test")

    train_emb_in_base = convert_emb("data_emb/pre_trained_emb_in/train")
    dev_emb_in_base = convert_emb("data_emb/pre_trained_emb_in/train")
    test_emb_in_base = convert_emb("data_emb/pre_trained_emb_in/train")

    logger.info("Bert-emb 拼接中······")
    train_emb = np.concatenate((train_emb_out_base,train_emb_in_base), axis=2)
    dev_emb = np.concatenate((dev_emb_out_base,dev_emb_in_base), axis=2)
    test_emb = np.concatenate((test_emb_out_base,test_emb_in_base), axis=2)

    logger.info("根据多个特征图，生成先验窗口······")

    anchors = ssd_anchors_all_layers(img_shape=FLAGS.seq_length, layers_shape=[50,50], anchor_sizes=[[1,2,3,4,5,6],[1,2,3,4,5,6]],offset=[0,0.5],anchor_ratios=1,anchor_steps=1)
    logger.info("根据多个特征图，进行数据编码······")
    tr_target_labels, tr_target_localizations, tr_target_scores = get_encode_box(anchors, tr_windows, tr_label)
    dev_target_labels, dev_target_localizations, dev_target_scores = get_encode_box(anchors, new_dev_windows, new_dev_label)
    test_target_labels, test_target_localizations, test_target_scores = get_encode_box(anchors, new_test_windows, new_test_label)

    if(FLAGS.binary_model):
        tr_target_labels = sparse_binary_label(tr_target_labels)
        dev_target_labels = sparse_binary_label(dev_target_labels)
        test_target_labels = sparse_binary_label(test_target_labels)

    tr_manager_data = BatchManager(train_emb, tr_target_labels, tr_target_localizations, tr_target_scores,batch_size=FLAGS.batch_size)
    tr_batch_sentence = tr_manager_data.sentence
    tr_batch_labels = tr_manager_data.target_labels
    tr_batch_localizations = tr_manager_data.target_localizations
    tr_batch_scores = tr_manager_data.target_scores

    dev_manager_data = BatchManager(dev_emb, dev_target_labels, dev_target_localizations,dev_target_scores, batch_size=FLAGS.batch_size)
    test_manager_data = BatchManager(test_emb, test_target_labels, test_target_localizations,test_target_scores, batch_size=FLAGS.batch_size)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    F1_value=0
    with tf.Session() as sess:
        # 传入当前会话，存储model的目录，装载预训练文件的函数，模型参数
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, logger)
        if FLAGS.train:
            logger.info("Start training")
            for i in range(FLAGS.epoch):
                loss = []
                for sentence_node, labels_1, labels_2, localizations_1,localizations_2,scores_1,scores_2 in zip(tr_batch_sentence, tr_batch_labels[0],tr_batch_labels[1],tr_batch_localizations[0],tr_batch_localizations[1],tr_batch_scores[0],tr_batch_scores[1]):
                    step, batch_loss=model.run_step(sess,[sentence_node, labels_1, labels_2, localizations_1,localizations_2,scores_1,scores_2],True)
                    loss.append(batch_loss)
                logger.info("epoch {} loss  {:>9.6f}".format(i,np.mean(loss)))
                P,R,F=evaluate(sess, model, dev_manager_data, anchors, logger, new_dev_sentence, new_dev_windows,new_dev_label,mode="dev")
                if(F>F1_value):
                    F1_value=F
                    logger.info("模型当前最好性能为：{}".format(F))
                    save_model(sess,model,FLAGS.ckpt_path,logger)
        if FLAGS.test:
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                model.saver.restore(sess, ckpt.model_checkpoint_path)

            if (not FLAGS.binary_model):
                logger.info("Mul-eval on Test-set:")
                evaluate_mul(sess, model, test_manager_data, anchors, logger, new_test_sentence, new_test_windows,new_test_label,mode="test")
            else:
                logger.info("Eval on Test-set:")
                evaluate(sess, model, test_manager_data, anchors, logger, new_test_sentence, new_test_windows,new_test_label,mode="test")

    backup_log()
    logger.info("-------------------------实验已完成--------------------------\n")

if __name__ == '__main__':
    main()


