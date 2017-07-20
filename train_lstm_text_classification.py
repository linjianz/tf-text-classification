#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Linjian Zhang
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import shutil
from datetime import datetime
import re
from tensorflow.contrib import learn


dir0 = '20170720_2'    # change it every time when training
lr_base = 1e-3          # 初始学习率
epoch_max = 200         # 最大epoch次数
epoch_save = 20         # 每#epoch保存一次模型
max_to_keep = 3         # 最多保存模型数目
batch_size = 64         # batch size
embedding_size = 128    # 词向量维度
hidden_size = 128       # lstm隐层单元个数
num_layer = 2           # lstm层数
########################################
data_file1 = 'data/rt-polaritydata/rt-polarity.pos'
data_file2 = 'data/rt-polaritydata/rt-polarity.neg'
# dir_restore = 'model/cnn/20170705_1/model-30200'
net_name = 'lstm/'
dir_models = 'model/' + net_name
dir_logs = 'log/' + net_name
dir_model = dir_models + dir0
dir_log_train = dir_logs + dir0 + '_train'
dir_log_val = dir_logs + dir0 + '_val'
if not os.path.exists(dir_models):
    os.mkdir(dir_models)
if not os.path.exists(dir_logs):
    os.mkdir(dir_logs)
if os.path.exists(dir_model):
    shutil.rmtree(dir_model)
if os.path.exists(dir_log_train):
    shutil.rmtree(dir_log_train)
if os.path.exists(dir_log_val):
    shutil.rmtree(dir_log_val)

os.mkdir(dir_model)
os.mkdir(dir_log_train)
os.mkdir(dir_log_val)
########################################


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(file1, file2):
    # Load data from files
    data1 = list(open(file1, "rb").readlines())
    data1 = [s.strip() for s in data1]
    number_pos = len(data1)
    data2 = list(open(file2, "rb").readlines())
    data2 = [s.strip() for s in data2]
    # Split by words
    data_total = data1 + data2
    data_total = [clean_str(str(sent)) for sent in data_total]

    # Generate labels
    label1 = [[0, 1] for _ in data1]
    label2 = [[1, 0] for _ in data2]

    # Generate vocabulary
    sequence_length = max([len(x.split(" ")) for x in data_total])
    vocab_processor = learn.preprocessing.VocabularyProcessor(sequence_length)
    data = np.array(list(vocab_processor.fit_transform(data_total)))
    # 将text转化为index，并padding成相同长度，这里发现了一个小bug（同一个单词会有不同的index？）
    # vocab_processor.save(os.path.join(dir_model, 'vocab'))  # 保存vocabulary

    data_pos = data[0:number_pos]
    data_neg = data[number_pos:]
    number_train_pos = int(0.9 * number_pos)
    data_t = np.concatenate([data_pos[:number_train_pos], data_neg[:number_train_pos]], 0)
    label_t = np.concatenate([label1[:number_train_pos], label2[:number_train_pos]], 0)
    data_v = np.concatenate([data_pos[number_train_pos:], data_neg[number_train_pos:]], 0)
    label_v = np.concatenate([label1[number_train_pos:], label2[number_train_pos:]], 0)

    # Generate mask  [57, sample_number]
    mask_t = (np.array(data_t) > 0).astype(int).transpose([1, 0])
    mask_v = (np.array(data_v) > 0).astype(int).transpose([1, 0])
    return data_t, label_t, mask_t, data_v, label_v, mask_v, vocab_processor


class Data(object):
    def __init__(self, data, label, mask, bs=batch_size, shuffle=True):
        self.data = data
        self.label = label
        self.mask = mask
        self.bs = bs
        self.shuffle = shuffle
        self.index = 0  # point at total_index
        self.number = len(label)
        self.total_index = range(self.number)
        if self.shuffle:
            self.total_index = np.random.permutation(self.total_index)

    def next_batch(self):
        start = self.index
        self.index += self.bs
        if self.index > self.number:
            if self.shuffle:
                self.total_index = np.random.permutation(self.total_index)
            self.index = 0
            start = self.index
            self.index += self.bs
        end = self.index
        return_x1 = self.data[self.total_index[start:end]]
        return_x2 = self.label[self.total_index[start:end]]
        return_x3 = self.mask[:, self.total_index[start:end]]
        return return_x1, return_x2, return_x3


class Net(object):
    def __init__(self, sequence_length, num_class, vocabulary_size, hidden_size, num_layer):
        self.x1 = tf.placeholder(tf.int32, [None, sequence_length], name='x1')  # sentence
        self.x2 = tf.placeholder(tf.int32, [None, num_class], name='x2')  # label
        self.x3 = tf.placeholder(tf.float32, [sequence_length, None], name='x3')  # mask
        self.x4 = tf.placeholder(tf.float32, [], name='x4')  # lr
        self.x5 = tf.placeholder(tf.float32, [], name='x5')  # dropout
        self.x6 = tf.placeholder(tf.int32, [], name='x6')  # batch size
        with tf.variable_scope('embedding'):
            self.w_embed = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='w_embed')
            embed = tf.nn.embedding_lookup(self.w_embed, self.x1)  # [bs, 57, 128]
            self.inputs = tf.nn.dropout(embed, self.x5)
        # lstm
        with tf.variable_scope('lstm_cell'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.x5)  # dropout
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layer, state_is_tuple=True)
            self._initial_state = cell.zero_state(self.x6, dtype=tf.float32)
        output = []
        state = self._initial_state
        with tf.variable_scope('lstm_layer'):
            for time_step in range(sequence_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs[:, time_step, :], state)
                output.append(cell_output)  # [57, 64, 128]
        output = output * self.x3[:, :, None]  # [57, 64, 128]
        # 输出要用mask屏蔽掉padding部分，也就是直接将原句中padding为零的的地方同样作用于输出
        with tf.name_scope("mean_pooling_layer"):
            output = tf.reduce_sum(output, 0)/(tf.reduce_sum(self.x3, 0)[:, None])  # [64, 128] 沿时间轴做mean-pooling
        with tf.variable_scope('softmax'):
            self.fc = slim.fully_connected(output, num_class, activation_fn=None, scope='fc')

        # loss & accuracy
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc, labels=self.x2, name='loss')
        self.loss = tf.reduce_mean(losses)  # 不能少！取均值
        optimizer = tf.train.AdamOptimizer(self.x4)
        self.train_op = slim.learning.create_train_op(self.loss, optimizer)
        self.prediction = tf.argmax(self.fc, 1, name='prediction')
        correct_prediction = tf.equal(self.prediction, tf.argmax(self.x2, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')

        # tensor board
        loss_summary = tf.summary.scalar('loss', self.loss)
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        self.summary_merge = tf.summary.merge([loss_summary, accuracy_summary])

        # init & save configuration
        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        # self.t_vars = tf.trainable_variables()
        # self.variables_names = [v.name for v in self.t_vars]  #  turn on if you want to check the variables

        # gpu configuration
        self.tf_config = tf.ConfigProto()
        # self.tf_config.gpu_options.allow_growth = True
        # if use_gpu_1:
        #     self.tf_config.gpu_options.visible_device_list = '1'


def main(_):
    # 1. 载入数据。将数据处理成网络需要的格式
    data_t, label_t, mask_t, data_v, label_v, mask_v, vocab_processor = load_data(data_file1, data_file2)
    # 2. 数据初始化及产生mini-batch数据
    model_data_t = Data(data_t, label_t, mask_t)
    # 3. 定义graph
    model = Net(sequence_length=data_t.shape[1],
                num_class=label_t.shape[1],
                vocabulary_size=len(vocab_processor.vocabulary_),
                hidden_size=hidden_size,
                num_layer=num_layer)

    with tf.Session(config=model.tf_config) as sess:
        writer_train = tf.summary.FileWriter(dir_log_train, sess.graph)
        writer_val = tf.summary.FileWriter(dir_log_val, sess.graph)
        sess.run(model.init)
        for epoch in range(epoch_max):
            lr = lr_base
            iter_per_epoch = len(label_t) // batch_size
            for iteration in range(iter_per_epoch):
                global_iter = epoch * iter_per_epoch + iteration
                x1_t, x2_t, x3_t = model_data_t.next_batch()
                feed_dict_t = dict()
                feed_dict_t[model.x1] = x1_t
                feed_dict_t[model.x2] = x2_t
                feed_dict_t[model.x3] = x3_t
                feed_dict_t[model.x4] = lr
                feed_dict_t[model.x5] = 0.5
                feed_dict_t[model.x6] = batch_size
                sess.run(model.train_op, feed_dict_t)
                # display
                if not (iteration + 1) % 10:
                    summary_out_t, loss_out_t, acc_out_t = sess.run([model.summary_merge, model.loss, model.accuracy],
                                                                    feed_dict_t)
                    writer_train.add_summary(summary_out_t, global_iter + 1)
                    print('%s, epoch %03d/%03d, iter %04d/%04d, lr %.5f, loss: %.5f, accuracy: %.5f' %
                          (datetime.now(), epoch + 1, epoch_max, iteration + 1, iter_per_epoch, lr, loss_out_t,
                           acc_out_t))
                if not (iteration + 1) % 100:
                    feed_dict_v = dict()
                    feed_dict_v[model.x1] = data_v
                    feed_dict_v[model.x2] = label_v
                    feed_dict_v[model.x3] = mask_v
                    feed_dict_v[model.x5] = 1.0
                    feed_dict_v[model.x6] = data_v.shape[0]
                    summary_out_v, loss_out_v, acc_out_v = sess.run([model.summary_merge, model.loss, model.accuracy],
                                                                    feed_dict_v)
                    writer_val.add_summary(summary_out_v, global_iter + 1)
                    print('****val loss: %.5f, accuracy: %.5f****' % (loss_out_v, acc_out_v))
                # save
                if not (global_iter + 1) % (epoch_save * iter_per_epoch):
                    model.saver.save(sess, (dir_model + '/model'), global_step=global_iter + 1)


if __name__ == '__main__':
    tf.app.run()
