# -*- coding: UTF-8 -*-

from Settings import Config
import re
import os
import sys
import numpy as np
import pickle
import random

class Dataset:
    def __init__(self):
        self.config = Config()
        self.iter_num = 0
        self.label_set = ['Negative', 'Neutral', 'Positive']
        self.visual = pickle.load(open('./data/mosi/processed_visual_dict.pkl', 'rb'))
        self.audio = pickle.load(open('./data/mosi/audio_dict.pkl', 'rb'))
        self.text = pickle.load(open('./data/mosi/text_emb.pkl', 'rb'))
        self.label = pickle.load(open('./data/mosi/label_dict.pkl', 'rb'))
        self.max_visual_len = self.config.max_visual_len
        self.max_audio_len = self.config.max_audio_len
        self.max_text_len = self.config.max_text_len
        self.keys = []
        # for key in self.label:
        #     self.keys.append(key)
        # np.random.shuffle(self.keys)
        self.flag = {}
        for item in self.keys:
            self.flag[item] = -1



    def padding(self, seq, max_len):
        shape_1 = np.shape(seq)[0]
        shape_2 = np.shape(seq)[1]
        emb_matrix = list(np.zeros([max_len, shape_2]))
        if shape_1 >= max_len:
            shape_1 = max_len
        for i in range(shape_1):
            emb_matrix[i] = seq[i]
        return emb_matrix


    def setdata(self, train_size):
        traindata = {'ID':[], 'V':[], 'A':[], 'T':[], 'L':[], 'F':[]}
        testdata = {'ID':[], 'V':[], 'A':[], 'T':[], 'L':[], 'F':[]}
        temp_order = list(range(len(self.label)))
#       np.random.shuffle(temp_order)
        for i in range(len(self.keys)):
            cur_id = self.keys[i]
            if i < train_size:
                traindata['ID'].append(cur_id)
                traindata['V'].append(self.padding(self.visual[cur_id], self.max_visual_len))
                traindata['A'].append(self.padding(self.audio[cur_id], self.max_audio_len))
                traindata['T'].append(self.padding(self.text[cur_id], self.max_text_len))
                traindata['L'].append(self.label_set.index(self.label[cur_id]))
                traindata['F'].append(self.flag[cur_id])
            else:
                testdata['ID'].append(cur_id)
                testdata['V'].append(self.padding(self.visual[cur_id], self.max_visual_len))
                testdata['A'].append(self.padding(self.audio[cur_id], self.max_audio_len))
                testdata['T'].append(self.padding(self.text[cur_id], self.max_text_len))
                testdata['L'].append(self.label_set.index(self.label[cur_id]))
                testdata['F'].append(self.flag[cur_id])
        return traindata, testdata

    def nextBatch(self, traindata, testdata, is_training = True):
        nextIDBatch = []
        nextVisualBatch = []
        nextAudioBatch = []
        nextTextBatch = []
        nextLabelBatch = []
        nextFlagBatch = []

        if is_training:
            if (self.iter_num+1)*self.config.batch_size > len(traindata['ID']):
                self.iter_num = 0
            if self.iter_num == 0:
                self.temp_order =  list(range(len(traindata['ID'])))
                np.random.shuffle(self.temp_order)
            temp_order = self.temp_order[self.iter_num*self.config.batch_size:(self.iter_num+1)*self.config.batch_size]
        else:
            if (self.iter_num+1)*self.config.batch_size > len(testdata['ID']):
                self.iter_num = 0
            if self.iter_num == 0:
                self.temp_order =  list(range(len(testdata['ID'])))
            temp_order = self.temp_order[self.iter_num*self.config.batch_size:(self.iter_num+1)*self.config.batch_size]


        ID = []
        visual = []
        audio = []
        text = []
        label = []
        flag = []

        for it in temp_order:
            if is_training:
                ID.append(traindata['ID'][it])
                visual.append(traindata['V'][it])
                audio.append(traindata['A'][it])
                text.append(traindata['T'][it])
                label.append(traindata['L'][it])
                flag.append(traindata['F'][it])
            else:
                ID.append(testdata['ID'][it])
                visual.append(testdata['V'][it])
                audio.append(testdata['A'][it])
                text.append(testdata['T'][it])
                label.append(testdata['L'][it])
                flag.append(testdata['F'][it])

        self.iter_num += 1
        nextIDBatch = np.array(ID)
        nextVisualBatch = np.array(visual)
        nextAudioBatch = np.array(audio)
        nextTextBatch = np.array(text)
        nextLableBatch = np.array(label)
        nextFlagBatch = np.array(flag)

        cur_batch = {'ID':nextIDBatch, 'V':nextVisualBatch, 'A':nextAudioBatch, 'T':nextTextBatch, 'L':nextLableBatch, 'F':nextFlagBatch}
        return cur_batch

if __name__ == '__main__':
    data = Dataset()
    traindata, testdata = data.setdata(2000)    #训练集的大小为2000，测试集大约是100左右

#    print (traindata['A'][:2])

    pickle.dump(traindata, open('D:\STMSA_data\mosi/train.pkl', 'wb'))
    pickle.dump(testdata, open('D:\STMSA_data\mosi/test.pkl', 'wb'))


#    print (traindata['L'][:10])
    cur_batch = data.nextBatch(traindata, testdata, True)
#    print (cur_batch['L'])
    print (cur_batch['ID'])
