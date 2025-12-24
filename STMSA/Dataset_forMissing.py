import pickle
import random
import numpy as np
from Settings import Config
config = Config()
max_visual_len = config.max_visual_len
max_audio_len = config.max_audio_len
max_text_len = config.max_text_len
missing_rate = 0.1
missing_type = 'm11'
a = 0  #3
b = 2  #5
traindata = pickle.load(open('D:\STMSA_data\mosi/train.pkl', 'rb'))
testdata_ = pickle.load(open('D:\STMSA_data\mosi/test.pkl', 'rb'))



def setdata():
    testdata = traindata
    missing_num = int(len(testdata['ID']) * missing_rate)
    testdata_for_m00fill_passing = {'ID': [], 'V': [], 'A': [], 'T': [], 'L': [], 'F': []}
    testdata_for_m00fill_passing_shuffle = {'ID': [], 'V': [], 'A': [], 'T': [], 'L': [], 'F': []}
    miss_visual = list(np.zeros([max_visual_len, 709]))
    miss_audio = list(np.zeros([max_audio_len, 33]))
    miss_text = list(np.zeros([max_text_len, 768]))
    print(len(testdata['ID']))
    for i in range(0, len(testdata['ID'])):
        print(i)
        if i < missing_num:
            rnd = random.randint(a,b)  # 0: visual  1:audio  2:text
            if rnd == 0:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(miss_visual)
                testdata_for_m00fill_passing['A'].append(testdata['A'][i])
                testdata_for_m00fill_passing['T'].append(testdata['T'][i])
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
            elif rnd == 1:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(testdata['V'][i])
                testdata_for_m00fill_passing['A'].append(miss_audio)
                testdata_for_m00fill_passing['T'].append(testdata['T'][i])
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
            elif rnd == 2:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(testdata['V'][i])
                testdata_for_m00fill_passing['A'].append(testdata['A'][i])
                testdata_for_m00fill_passing['T'].append(miss_text)
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
            elif rnd == 3:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(testdata['V'][i])
                testdata_for_m00fill_passing['A'].append(miss_audio)
                testdata_for_m00fill_passing['T'].append(miss_text)
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
            elif rnd == 4:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(miss_visual)
                testdata_for_m00fill_passing['A'].append(testdata['A'][i])
                testdata_for_m00fill_passing['T'].append(miss_text)
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
            else:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(miss_visual)
                testdata_for_m00fill_passing['A'].append(miss_audio)
                testdata_for_m00fill_passing['T'].append(testdata['T'][i])
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
        else:
            testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
            testdata_for_m00fill_passing['V'].append(testdata['V'][i])
            testdata_for_m00fill_passing['A'].append(testdata['A'][i])
            testdata_for_m00fill_passing['T'].append(testdata['T'][i])
            testdata_for_m00fill_passing['L'].append(testdata['L'][i])
            testdata_for_m00fill_passing['F'].append(testdata['F'][i])


    my_list = testdata_for_m00fill_passing['ID']
    id_shuffle = random.sample(my_list, len(my_list))
    for id in  id_shuffle:
        index = testdata_for_m00fill_passing['ID'].index(id)
        testdata_for_m00fill_passing_shuffle['ID'].append(testdata_for_m00fill_passing['ID'][index])
        testdata_for_m00fill_passing_shuffle['V'].append(testdata_for_m00fill_passing['V'][index])
        testdata_for_m00fill_passing_shuffle['A'].append(testdata_for_m00fill_passing['A'][index])
        testdata_for_m00fill_passing_shuffle['T'].append(testdata_for_m00fill_passing['T'][index])
        testdata_for_m00fill_passing_shuffle['L'].append(testdata_for_m00fill_passing['L'][index])
        testdata_for_m00fill_passing_shuffle['F'].append(testdata_for_m00fill_passing['F'][index])

    pickle.dump(testdata_for_m00fill_passing, open('D:\STMSA_data\mosi/' + missing_type + 'train3.pkl', 'wb'))

def setdata_fortest():

    testdata = testdata_
    missing_num = int(len(testdata['ID']) * missing_rate)
    testdata_for_m00fill_passing = {'ID': [], 'V': [], 'A': [], 'T': [], 'L': [], 'F': []}
    testdata_for_m00fill_passing_shuffle = {'ID': [], 'V': [], 'A': [], 'T': [], 'L': [], 'F': []}
    miss_visual = list(np.zeros([max_visual_len, 709]))
    miss_audio = list(np.zeros([max_audio_len, 33]))
    miss_text = list(np.zeros([max_text_len, 768]))
    print(len(testdata['ID']))
    for i in range(0, len(testdata['ID'])):
        print(i)
        if i < missing_num:
            rnd = random.randint(a,b)  # 0: visual  1:audio  2:text
            if rnd == 0:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(miss_visual)
                testdata_for_m00fill_passing['A'].append(testdata['A'][i])
                testdata_for_m00fill_passing['T'].append(testdata['T'][i])
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
            elif rnd == 1:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(testdata['V'][i])
                testdata_for_m00fill_passing['A'].append(miss_audio)
                testdata_for_m00fill_passing['T'].append(testdata['T'][i])
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
            elif rnd == 2:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(testdata['V'][i])
                testdata_for_m00fill_passing['A'].append(testdata['A'][i])
                testdata_for_m00fill_passing['T'].append(miss_text)
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
            elif rnd == 3:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(testdata['V'][i])
                testdata_for_m00fill_passing['A'].append(miss_audio)
                testdata_for_m00fill_passing['T'].append(miss_text)
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
            elif rnd == 4:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(miss_visual)
                testdata_for_m00fill_passing['A'].append(testdata['A'][i])
                testdata_for_m00fill_passing['T'].append(miss_text)
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
            else:
                testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
                testdata_for_m00fill_passing['V'].append(miss_visual)
                testdata_for_m00fill_passing['A'].append(miss_audio)
                testdata_for_m00fill_passing['T'].append(testdata['T'][i])
                testdata_for_m00fill_passing['L'].append(testdata['L'][i])
                testdata_for_m00fill_passing['F'].append(rnd)
        else:
            testdata_for_m00fill_passing['ID'].append(testdata['ID'][i])
            testdata_for_m00fill_passing['V'].append(testdata['V'][i])
            testdata_for_m00fill_passing['A'].append(testdata['A'][i])
            testdata_for_m00fill_passing['T'].append(testdata['T'][i])
            testdata_for_m00fill_passing['L'].append(testdata['L'][i])
            testdata_for_m00fill_passing['F'].append(testdata['F'][i])


    my_list = testdata_for_m00fill_passing['ID']
    id_shuffle = random.sample(my_list, len(my_list))
    for id in  id_shuffle:
        index = testdata_for_m00fill_passing['ID'].index(id)
        testdata_for_m00fill_passing_shuffle['ID'].append(testdata_for_m00fill_passing['ID'][index])
        testdata_for_m00fill_passing_shuffle['V'].append(testdata_for_m00fill_passing['V'][index])
        testdata_for_m00fill_passing_shuffle['A'].append(testdata_for_m00fill_passing['A'][index])
        testdata_for_m00fill_passing_shuffle['T'].append(testdata_for_m00fill_passing['T'][index])
        testdata_for_m00fill_passing_shuffle['L'].append(testdata_for_m00fill_passing['L'][index])
        testdata_for_m00fill_passing_shuffle['F'].append(testdata_for_m00fill_passing['F'][index])

    pickle.dump(testdata_for_m00fill_passing, open('D:\STMSA_data\mosi/' + missing_type + 'test3.pkl', 'wb'))

if __name__ == '__main__':
    setdata()
    setdata_fortest()