import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
from Settings import Config
from module import ff, multihead_attention, ln, mask, SigmoidAtt
import sys
from tensorflow.python.keras.utils import losses_utils


class MM:
    def __init__(self, is_training):
        self.config = Config()
        self.att_dim = self.config.att_dim
        self.visual = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, self.config.max_visual_len, 709],
                                     name='visual')
        self.audio = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, self.config.max_audio_len, 33],
                                    name='audio')
        self.text = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, self.config.max_text_len, 768],
                                   name='text')
        self.label = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size], name='label')
        self.flag = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size], name='flag')
        #self.pretrained_output = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, 275, 300], name='pre')#全模态时需要注释[self.config.batch_size, 275, 300]

        visual = tf.layers.dense(self.visual, self.config.att_dim, use_bias=False)
        audio = tf.layers.dense(self.audio, self.config.att_dim, use_bias=False)
        text = tf.layers.dense(self.text, self.config.att_dim, use_bias=False)

        with tf.variable_scope('vv', reuse=tf.AUTO_REUSE):
            enc_vv = multihead_attention(queries=visual,
                                         keys=visual,
                                         values=visual,
                                         num_heads=4,
                                         dropout_rate=0.2,
                                         training=True,
                                         causality=False)
            enc_vv = ff(enc_vv, num_units=[4 * self.config.att_dim, self.config.att_dim])
            enc_vv_new = enc_vv

        with tf.variable_scope('aa', reuse=tf.AUTO_REUSE):
            enc_aa = multihead_attention(queries=audio,
                                         keys=audio,
                                         values=audio,
                                         num_heads=4,
                                         dropout_rate=0.2,
                                         training=True,
                                         causality=False)
            enc_aa = ff(enc_aa, num_units=[4 * self.config.att_dim, self.config.att_dim])
            # 32 150 300

        with tf.variable_scope('tt', reuse=tf.AUTO_REUSE):
            enc_tt = multihead_attention(queries=text,
                                         keys=text,
                                         values=text,
                                         num_heads=4,
                                         dropout_rate=0.2,
                                         training=True,
                                         causality=False)
            enc_tt = ff(enc_tt, num_units=[4 * self.config.att_dim, self.config.att_dim])
            # 32 25 300

        with tf.variable_scope('all_weights', reuse=tf.AUTO_REUSE):
            Wr_wq = tf.get_variable('Wr_wq', [300, 1])
            Wm_wq = tf.get_variable('Wm_wq', [300, 300])
            Wu_wq = tf.get_variable('Wu_wq', [300, 300])

            wei_va = tf.get_variable('wei_va', [self.att_dim, 150])
            wei_vt = tf.get_variable('wei_vt', [self.att_dim, 150])
            wei_ta = tf.get_variable('wei_ta', [self.att_dim, 150])

            wei_dvat = tf.get_variable('wei_vat', [self.att_dim, 150])
            wei_datv = tf.get_variable('wei_atv', [self.att_dim, 150])
            wei_dvta = tf.get_variable('wei_vta', [self.att_dim, 150])

            W_l = tf.get_variable('W_l', [self.att_dim, self.config.class_num])
            b_l = tf.get_variable('b_l', [1, self.config.class_num])

        common_new_1 = tf.reshape(tf.concat([tf.matmul(tf.reshape(enc_vv, [-1, self.att_dim]), wei_va),
                                             tf.matmul(tf.reshape(enc_vv, [-1, self.att_dim]), wei_vt)], -1),
                                  [self.config.batch_size, -1, self.att_dim])

        common_new_2 = tf.reshape(tf.concat([tf.matmul(tf.reshape(enc_aa, [-1, self.att_dim]), wei_va),
                                             tf.matmul(tf.reshape(enc_aa, [-1, self.att_dim]), wei_ta)], -1),
                                  [self.config.batch_size, -1, self.att_dim])

        common_new_3 = tf.reshape(tf.concat([tf.matmul(tf.reshape(enc_tt, [-1, self.att_dim]), wei_vt),
                                             tf.matmul(tf.reshape(enc_tt, [-1, self.att_dim]), wei_ta)], -1),
                                  [self.config.batch_size, -1, self.att_dim])

        enc_all_new = tf.concat([common_new_1, common_new_2, common_new_3], 1)
        enc_new = tf.convert_to_tensor(enc_all_new)


        with tf.variable_scope('dec_vt', reuse=tf.AUTO_REUSE):
            dec_vt = multihead_attention(queries=enc_tt,
                                          keys=enc_vv,
                                          values=enc_vv,
                                          num_heads=4,
                                          dropout_rate=0.2,
                                          training=True,
                                          causality=False)
            dec_vt = ff(dec_vt, num_units=[4 * self.config.att_dim, self.config.att_dim])

        with tf.variable_scope('enc_dec_vt', reuse=tf.AUTO_REUSE):
            enc_dec_vt = multihead_attention(queries=dec_vt,
                                          keys=dec_vt,
                                          values=dec_vt,
                                          num_heads=4,
                                          dropout_rate=0.2,
                                          training=True,
                                          causality=False)
            enc_dec_vt = ff(enc_dec_vt, num_units=[4 * self.config.att_dim, self.config.att_dim])


        with tf.variable_scope('dec_vta', reuse=tf.AUTO_REUSE):
            dec_vta = multihead_attention(queries=enc_aa,
                                          keys=enc_dec_vt,
                                          values=enc_dec_vt,
                                          num_heads=4,
                                          dropout_rate=0.2,
                                          training=True,
                                          causality=False)
            dec_vta = ff(dec_vta, num_units=[4 * self.config.att_dim, self.config.att_dim])



        with tf.variable_scope('dec_at', reuse=tf.AUTO_REUSE):
            dec_at = multihead_attention(queries=enc_tt,
                                          keys=enc_aa,
                                          values=enc_aa,
                                          num_heads=4,
                                          dropout_rate=0.2,
                                          training=True,
                                          causality=False)
            dec_at = ff(dec_at, num_units=[4 * self.config.att_dim, self.config.att_dim])


        with tf.variable_scope('enc_dec_at', reuse=tf.AUTO_REUSE):
            enc_dec_at = multihead_attention(queries=dec_at,
                                          keys=dec_at,
                                          values=dec_at,
                                          num_heads=4,
                                          dropout_rate=0.2,
                                          training=True,
                                          causality=False)
            enc_dec_at = ff(enc_dec_at, num_units=[4 * self.config.att_dim, self.config.att_dim])


        with tf.variable_scope('dec_atv', reuse=tf.AUTO_REUSE):
            dec_atv = multihead_attention(queries=enc_vv_new,
                                          keys=enc_dec_at,
                                          values=enc_dec_at,
                                          num_heads=4,
                                          dropout_rate=0.2,
                                          training=True,
                                          causality=False)
            dec_atv = ff(dec_atv, num_units=[4 * self.config.att_dim, self.config.att_dim])


        common_4 = tf.reshape(tf.concat([tf.matmul(tf.reshape(dec_atv, [-1, self.att_dim]), wei_dvat),
                                         tf.matmul(tf.reshape(dec_atv, [-1, self.att_dim]), wei_dvta)], -1),
                              [self.config.batch_size, -1, self.att_dim])
        common_5 = tf.reshape(tf.concat([tf.matmul(tf.reshape(dec_vta, [-1, self.att_dim]), wei_dvat),
                                         tf.matmul(tf.reshape(dec_vta, [-1, self.att_dim]), wei_datv)], -1),
                              [self.config.batch_size, -1, self.att_dim])
        common_6 = tf.reshape(tf.concat([tf.matmul(tf.reshape(dec_at, [-1, self.att_dim]), wei_dvta),
                                         tf.matmul(tf.reshape(dec_vt, [-1, self.att_dim]), wei_datv)], -1),
                              [self.config.batch_size, -1, self.att_dim])

        enc_all_1 = tf.concat([common_4, common_5, common_6], 1)
        enc_new_new_1 = tf.convert_to_tensor(enc_all_1)

        with tf.variable_scope('nn', reuse=tf.AUTO_REUSE):
            enc_en = multihead_attention(queries=enc_new,
                                         keys=enc_new_new_1,
                                         values=enc_new_new_1,
                                         num_heads=4,
                                         dropout_rate=0.2,
                                         training=True,
                                         causality=False)
            enc_en = ff(enc_en, num_units=[4 * 300, 300])

        # encode kl loss
        kl = tf.keras.losses.KLDivergence(reduction=losses_utils.ReductionV2.NONE, name='kl')

        #pre-trained loss
        # kl_loss1 = kl(tf.nn.softmax(enc_en, -1), tf.nn.softmax(self.pretrained_output, -1))#
        # kl_loss2 = kl(tf.nn.softmax(self.pretrained_output, -1), tf.nn.softmax(enc_en, -1))#
        # self.kl_loss = tf.reduce_sum(tf.reduce_mean(kl_loss1, -1), -1) + tf.reduce_sum(tf.reduce_mean(kl_loss2, -1), -1)#

        enc_en = tf.multiply(enc_en, 1, name='encode_outputs')

        # decode
        with tf.variable_scope('de', reuse=tf.AUTO_REUSE):
            enc_de = multihead_attention(queries=enc_en,
                                         keys=enc_en,
                                         values=enc_en,
                                         num_heads=4,
                                         dropout_rate=0.2,
                                         training=True,
                                         causality=False)
            enc_de = ff(enc_de, num_units=[4 * 300, 300])

        # decoder loss
        de_loss1 = kl(tf.nn.softmax(enc_de, -1), tf.nn.softmax(enc_new, -1))
        de_loss2 = kl(tf.nn.softmax(enc_new, -1), tf.nn.softmax(enc_de, -1))
        self.de_loss = tf.reduce_sum(tf.reduce_mean(de_loss1, -1), -1) + tf.reduce_sum(tf.reduce_mean(de_loss2, -1), -1)

        outputs_en = SigmoidAtt(enc_en, Wr_wq, Wm_wq, Wu_wq)
        self.output = outputs_en

        vt_loss1 = kl(tf.nn.softmax(dec_vt, -1), tf.nn.softmax(enc_tt, -1))
        vt_loss2 = kl(tf.nn.softmax(enc_tt, -1), tf.nn.softmax(dec_vt, -1))
        self.vt_loss = tf.reduce_sum(tf.reduce_mean(vt_loss1, -1), -1) + tf.reduce_sum(tf.reduce_mean(vt_loss2, -1), -1)

        vta_loss1 = kl(tf.nn.softmax(dec_vta, -1), tf.nn.softmax(enc_aa, -1))
        vta_loss2 = kl(tf.nn.softmax(enc_aa, -1), tf.nn.softmax(dec_vta, -1))
        self.vta_loss = tf.reduce_sum(tf.reduce_mean(vta_loss1, -1), -1) + tf.reduce_sum(tf.reduce_mean(vta_loss2, -1),
                                                                                         -1)

        at_loss1 = kl(tf.nn.softmax(dec_at, -1), tf.nn.softmax(enc_tt, -1))
        at_loss2 = kl(tf.nn.softmax(enc_tt, -1), tf.nn.softmax(dec_at, -1))
        self.at_loss = tf.reduce_sum(tf.reduce_mean(at_loss1, -1), -1) + tf.reduce_sum(tf.reduce_mean(at_loss2, -1), -1)

        atv_loss1 = kl(tf.nn.softmax(dec_atv, -1), tf.nn.softmax(enc_vv_new, -1))
        atv_loss2 = kl(tf.nn.softmax(enc_vv_new, -1), tf.nn.softmax(dec_atv, -1))
        self.atv_loss = tf.reduce_sum(tf.reduce_mean(atv_loss1, -1), -1) + tf.reduce_sum(tf.reduce_mean(atv_loss2, -1), -1)


        temp_new = outputs_en
        temp_new = tf.layers.dense(temp_new, self.config.att_dim, use_bias=False)

        output_res = tf.add(tf.matmul(temp_new, W_l), b_l)
        ouput_label = tf.one_hot(self.label, self.config.class_num)
        self.prob = tf.nn.softmax(output_res)

        with tf.name_scope('loss'):
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output_res, labels=ouput_label))
            self.loss = loss
            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                                  weights_list=[W_l, b_l])


            self.total_loss = self.loss + 0.1 * self.l2_loss + 0.1 * self.de_loss + 0.1 * self.vt_loss + 0.1 * self.vta_loss + 0.1 * self.at_loss + 0.1 * self.atv_loss   #
            #self.total_loss = self.loss + 0.1 * self.l2_loss + 0.1 * self.de_loss + 0.1 * self.kl_loss + 0.1 * self.vt_loss + 0.1 * self.vta_loss + 0.1 * self.at_loss + 0.1 * self.atv_loss + 0.1 * self.de_loss + 0.1  * self.kl_loss + 0.1 * self.vt_loss + 0.1 * self.vta_loss + 0.1 * self.at_loss + 0.1 * self.atv_loss#
