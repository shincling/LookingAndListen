# coding=utf8
import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import time
import config as config
from predata import prepare_data
import myNet
# from test_multi_labels_speech import multi_label_vector
import os
import shutil
import librosa
import soundfile as sf
import myNet

import scipy.interpolate as inter


# import matlab
# import matlab.engine
# from separation import bss_eval_sources
# import bss_test

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


global_id=random.random()
config.EPOCH_SIZE = 300
np.random.seed(1)  # 设定种子
torch.manual_seed(1)
random.seed(1)
# stout=sys.stdout
# log_file=open(config.LOG_FILE_PRE,'w')
# sys.stdout=log_file
# logfile=config.LOG_FILE_PRE
test_all_outputchannel = 0

def interpolate(var,size,axis=1):
    # 要求输入是二维图像，插值的是后面的这一个维度！
    # cc=Variable(torch.random(6*256,75),requires_grad=True)
    # var=cc
    shape=var.size()
    assert len(shape)==2
    out_var=Variable(torch.zeros(shape[0],size),requires_grad=0).cuda()
    for i in range(size):
        out_var[:,i]=var[:,i*shape[1]/size]
    return out_var.view(shape[0],size)

def bss_eval_fromGenMap(multi_mask, x_input, top_k_mask_mixspeech, dict_idx2spk, data, sort_idx):
    if config.Out_Sep_Result:
        dst = 'batch_output'+str(global_id)+''
        if os.path.exists(dst):
            print " cleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)

    sample_idx = 0
    for each_pre, mask, s_idx in zip(multi_mask, top_k_mask_mixspeech, sort_idx):
        _mix_spec = data['mix_phase'][sample_idx]
        xxx = x_input[sample_idx].data.cpu().numpy()
        phase_mix = np.angle(_mix_spec)
        for idx, each_spk in enumerate(each_pre):
            this_spk = idx
            y_pre_map = each_pre[idx].data.cpu().numpy()
            # 如果第二个通道概率比较大
            # if idx==0 and s_idx[0].data.cpu().numpy()>s_idx[1].data.cpu().numpy():
            #      y_pre_map=1-each_pre[1].data.cpu().numpy()
            # if idx==1 and s_idx[0].data.cpu().numpy()<s_idx[1].data.cpu().numpy():
            #      y_pre_map=1-each_pre[0].data.cpu().numpy()
            y_pre_map = y_pre_map * xxx
            _pred_spec = y_pre_map * np.exp(1j * phase_mix)
            wav_pre = librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT)
            min_len = len(wav_pre)
            if test_all_outputchannel:
                min_len = len(wav_pre)
            sf.write('batch_output'+str(global_id)+'/{}_testspk{}_pre.wav'.format(sample_idx, this_spk), wav_pre[:min_len],
                     config.FRAME_RATE, )
        sf.write('batch_output'+str(global_id)+'/{}_True_mix.wav'.format(sample_idx), data['mix_wav'][sample_idx][:min_len],
                 config.FRAME_RATE, )
        sample_idx += 1

    for sample_idx, each_sample in enumerate(data['multi_spk_wav_list']):
        for each_spk in each_sample.keys():
            this_spk = each_spk
            wav_genTrue = each_sample[this_spk]
            min_len = 39936
            sf.write('batch_output'+str(global_id)+'/{}_{}_realTrue.wav'.format(sample_idx, this_spk), wav_genTrue[:min_len],
                     config.FRAME_RATE, )

    for sample_idx, each_sample in enumerate(data['multi_spk_fea_list']):
        _mix_spec = data['mix_phase'][sample_idx]
        phase_mix = np.angle(_mix_spec)
        for each_spk in each_sample.keys():
            this_spk = each_spk
            y_true_map = each_sample[this_spk]
            _genture_spec = y_true_map * np.exp(1j * phase_mix)
            wav_genTrue = librosa.core.spectrum.istft(np.transpose(_genture_spec), config.FRAME_SHIFT, )
            min_len = len(wav_pre)
            sf.write('batch_output'+str(global_id)+'/{}_{}_genTrue.wav'.format(sample_idx, this_spk), wav_genTrue[:min_len],
                     config.FRAME_RATE, )

def bss_eval_cRM(predict_map_real,predict_map_fake,y_multi_map,y_map_gtruth,dict_idx2spk,train_data):
    #评测和结果输出部分
    if config.Out_Sep_Result:
        dst='batch_output'
        if os.path.exists(dst):
            print " \ncleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)

    for sample_idx,each_sample in enumerate(train_data['multi_spk_wav_list']):
        for each_spk in each_sample.keys():
            this_spk=each_spk
            wav_genTrue=each_sample[this_spk]
            min_len = 39936
            sf.write('batch_output/{}_{}_realTrue.wav'.format(sample_idx,this_spk),wav_genTrue[:min_len],config.FRAME_RATE,)

    # 对于每个sample
    sample_idx=0 #代表一个batch里的依次第几个
    for each_y,each_pre_real,each_pre_fake,each_trueVector,spk_name in zip(y_multi_map,predict_map_real,predict_map_fake,y_map_gtruth,train_data['aim_spkname']):
        _mix_spec=train_data['mix_phase'][sample_idx]
        phase_mix = np.angle(_mix_spec)
        for idx,one_cha in enumerate(each_trueVector):
            if one_cha: #　如果此刻这个候选人通道是开启的
                this_spk=dict_idx2spk[one_cha]
                y_true_map=each_y[idx].data.cpu().numpy()
                y_pre_map_real=each_pre_real[idx].data.cpu().numpy()
                y_pre_map_fake=each_pre_fake[idx].data.cpu().numpy()
                _pred_spec = y_pre_map_real + (1j * y_pre_map_fake)
                _genture_spec = y_true_map[:,:,0] + (1j * y_true_map[:,:,1])
                wav_pre=librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT)
                wav_genTrue=librosa.core.spectrum.istft(np.transpose(_genture_spec), config.FRAME_SHIFT,)
                min_len = np.min((len(train_data['multi_spk_wav_list'][sample_idx][this_spk]), len(wav_pre)))
                if test_all_outputchannel:
                    min_len =  len(wav_pre)
                sf.write('batch_output/{}_{}_pre.wav'.format(sample_idx,this_spk),wav_pre[:min_len],config.FRAME_RATE,)
                sf.write('batch_output/{}_{}_genTrue.wav'.format(sample_idx,this_spk),wav_genTrue[:min_len],config.FRAME_RATE,)
        sf.write('batch_output/{}_True_mix.wav'.format(sample_idx),train_data['mix_wav'][sample_idx][:min_len],config.FRAME_RATE,)
        sample_idx+=1

def bss_eval(predict_multi_map, y_multi_map, y_map_gtruth, dict_idx2spk, train_data):
    # 评测和结果输出部分
    if config.Out_Sep_Result:
        dst = 'batch_output'+str(global_id)+''
        if os.path.exists(dst):
            print " \ncleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)

    # 对于每个sample
    sample_idx = 0  # 代表一个batch里的依次第几个
    for each_y, each_pre, each_trueVector, spk_name in zip(y_multi_map, predict_multi_map, y_map_gtruth,
                                                           train_data['aim_spkname']):
        _mix_spec = train_data['mix_phase'][sample_idx]
        phase_mix = np.angle(_mix_spec)
        for idx, one_cha in enumerate(each_trueVector):
            if one_cha:  # 如果此刻这个候选人通道是开启的
                this_spk = dict_idx2spk[one_cha]
                y_true_map = each_y[idx].data.cpu().numpy()
                y_pre_map = each_pre[idx].data.cpu().numpy()
                _pred_spec = y_pre_map * np.exp(1j * phase_mix)
                _genture_spec = y_true_map * np.exp(1j * phase_mix)
                wav_pre = librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT)
                wav_genTrue = librosa.core.spectrum.istft(np.transpose(_genture_spec), config.FRAME_SHIFT, )
                min_len = np.min((len(train_data['multi_spk_wav_list'][sample_idx][this_spk]), len(wav_pre)))
                if test_all_outputchannel:
                    min_len = len(wav_pre)
                sf.write('batch_output'+str(global_id)+'/{}_{}_pre.wav'.format(sample_idx, this_spk), wav_pre[:min_len],
                         config.FRAME_RATE, )
                sf.write('batch_output'+str(global_id)+'/{}_{}_genTrue.wav'.format(sample_idx, this_spk), wav_genTrue[:min_len],
                         config.FRAME_RATE, )
        sf.write('batch_output'+str(global_id)+'/{}_True_mix.wav'.format(sample_idx), train_data['mix_wav'][sample_idx][:min_len],
                 config.FRAME_RATE, )
        sample_idx += 1


def print_memory_state(memory):
    print '\n memory states:'
    for one in memory:
        print '\nspk:{},video:{},age:{}'.format(one[0], one[1].cpu().numpy()[
                                                        2 * config.EMBEDDING_SIZE - 3:2 * config.EMBEDDING_SIZE + 5],
                                                one[2])

def convert2numpy(data_list,top_k):
    output_size=(config.BATCH_SIZE,top_k)+ np.array(data_list[0].values()[0]).shape
    # print output_size
    output_array=np.zeros(output_size,dtype=np.float32)
    for idx,dict_sample in enumerate(data_list):#对于这个batch里的每一个sample(是一个dict)
        spk_all=sorted(dict_sample.keys()) #将它们排序，确保统一
        for jdx,spk in enumerate(spk_all):
            output_array[idx,jdx]=np.array(data_list[idx][spk])
    return output_array

class ATTENTION(nn.Module):
    def __init__(self,speech_fre):
        self.fre=speech_fre#应该是301
        super(ATTENTION, self).__init__()
        self.lstm_layer = nn.LSTM(
            input_size=(8*257+256),
            hidden_size=config.HIDDEN_UNITS,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )
        self.fc_layers1=nn.Linear(2*config.HIDDEN_UNITS,config.FC_UNITS)
        self.fc_layers2=nn.Linear(config.FC_UNITS,config.FC_UNITS)
        self.fc_layers3=nn.Linear(config.FC_UNITS,config.FC_UNITS)
        self.final_layer=nn.Linear(config.FC_UNITS,257*2)

    def forward(self, mix_hidden, query):
        # todo:这个要弄好，其实也可以直接抛弃memory来进行attention | DONE
        mix_shape=mix_hidden.size()#应该是bs*301*(8*257)的东西
        query_shape=query.size()#应该是bs*topk*301*256的东西
        top_k=query_shape[1]
        BATCH_SIZE = mix_hidden.size()[0]
        # assert query.size()==(BATCH_SIZE,self.hidden_size)
        # assert mix_hidden.size()[-1]==self.hidden_size
        # mix_hidden：bs,max_len,fre,hidden_size  query:bs,hidden_size

        mix_hidden = mix_hidden.view(BATCH_SIZE, 1,  mix_shape[1], mix_shape[2]).expand(BATCH_SIZE,top_k,mix_shape[1],mix_shape[2])
        mix_hidden = mix_hidden.contiguous().view(BATCH_SIZE*top_k, mix_shape[1], mix_shape[2]) #现在mix_hidden变成了(bs*topk)*301*(8*257)
        query=query.view(BATCH_SIZE*top_k,query_shape[2],query_shape[3]) #现在query变成了(bs*topk)*301*256
        # print mix_hidden
        # print '\n',query
        multi_moda=torch.cat((mix_hidden,query),dim=2) #得到了拼接好的特征矩阵

        multi_moda=self.lstm_layer(multi_moda)[0]
        print 'after the lstm size:',multi_moda.size()
        multi_moda=F.relu(self.fc_layers1(multi_moda))
        multi_moda=F.relu(self.fc_layers2(multi_moda))
        multi_moda=F.relu(self.fc_layers3(multi_moda))

        print 'The size of last embedding:',multi_moda.size() #应该是(bs*topk),301,600
        results=self.final_layer(multi_moda).view(BATCH_SIZE,top_k,mix_shape[1],257,2)
        print 'The size of output:',results.size() #应该是(bs*topk),301,600
        results=F.sigmoid(results)
        return results

class FACE_HIDDEN_simple(nn.Module):
    #这个是定制的那个预训练的抽脸部特征的1024的图像层次，后面替换，目前先随便用了一层全链接
    def __init__(self):
        super(FACE_HIDDEN_simple, self).__init__()
        self.layer=nn.Linear(3*299*299,1024)
    def forward(self, x):
        # x是bs,topk,75,3,299,299的
        topk=x.size()[1]
        x=x.contiguous()
        x=x.view(config.BATCH_SIZE,topk,75,-1)
        x=self.layer(x) # bs,topk,75,1024
        # x=x.view(config.BATCH_SIZE,topk,75,1024)
        x=torch.transpose(x,2,3).contiguous().view(config.BATCH_SIZE,topk,1024,75,1)

        return x

class FACE_HIDDEN(nn.Module):
    #这个是定制的那个预训练的抽脸部特征的1024的图像层次，后面替换，目前先随便用了一层全链接
    def __init__(self):
        super(FACE_HIDDEN, self).__init__()
        self.layer=nn.Linear(3*299*299,1024)
        self.image_net=myNet.inception_v3(pretrained=True)

    def forward(self, x):
        # x是bs,topk,75,3,299,299的
        topk=x.size()[1]
        x=x.view(-1,3,299,299)
        x=x.contiguous()
        x=self.image_net(x)[2] # bs,topk,75,2048
        print 'shape after image_net',x.size()
        x=x.view(config.BATCH_SIZE,topk,75,2048)
        x=torch.transpose(x,2,3).contiguous().view(config.BATCH_SIZE,topk,2048,75,1)

        return x


class FACE_EMB(nn.Module):
    def __init__(self,fre=301):
        self.fre=fre
        super(FACE_EMB, self).__init__()
        self.cnn1=nn.Conv2d(1024,256,(7,1),stride=1,padding=(3,0),dilation=(1,1))
        # self.cnn1=nn.Conv2d(2048,256,(7,1),stride=1,padding=(3,0),dilation=(1,1))
        self.cnn2=nn.Conv2d(256,256,(5,1),stride=1,padding=(2,0),dilation=(1,1))
        self.cnn3=nn.Conv2d(256,256,(5,1),stride=1,padding=(4,0),dilation=(2,1))
        self.cnn4=nn.Conv2d(256,256,(5,1),stride=1,padding=(8,0),dilation=(4,1))
        self.cnn5=nn.Conv2d(256,256,(5,1),stride=1,padding=(16,0),dilation=(8,1))
        self.cnn6=nn.Conv2d(256,256,(5,1),stride=1,padding=(32,0),dilation=(16,1))
        self.num_cnns=6
        self.bn1=nn.BatchNorm2d(256)
        self.bn2=nn.BatchNorm2d(256)
        self.bn3=nn.BatchNorm2d(256)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)

    def forward(self, x):
        print '\n Face layer log:'
        #　这个时候的输入应该是　bs,top-k,1024个通道,75帧,１
        shape=x.size()
        x = x.contiguous()
        x = x.view(-1,x.size(2),x.size(3),1)
        for idx in range(self.num_cnns):
            cnn_layer=eval('self.cnn{}'.format(idx+1))
            bn_layer=eval('self.bn{}'.format(idx+1))
            x=F.relu(cnn_layer(x))
            x=bn_layer(x)
            print 'Face shape after CNNs:',idx,'', x.size()

        #　到这里是(bs*topk, 256L, 75L, 1L)
        x=interpolate(x.view(-1,config.MAX_LEN_VIDEO),size=self.fre,axis=1)# 给进去一个二维，最后一个维度是要插值的
        #　到这里插值过后是（bs*topk*256,fre)
        x=torch.transpose(x.view(config.BATCH_SIZE,shape[1],256,self.fre),2,3).contiguous().cuda()
        return x.view(config.BATCH_SIZE,shape[1],self.fre,256)

class MIX_SPEECH(nn.Module):
    def __init__(self):
        super(MIX_SPEECH, self).__init__()

        self.cnn1=nn.Conv2d(2,96,(1,7),stride=1,padding=(0,3),dilation=(1,1))
        self.cnn2=nn.Conv2d(96,96,(7,1),stride=1,padding=(3,0),dilation=(1,1))
        self.cnn3=nn.Conv2d(96,96,(5,5),stride=1,padding=(2,2),dilation=(1,1))
        self.cnn4=nn.Conv2d(96,96,(5,5),stride=1,padding=(4,2),dilation=(2,1))
        self.cnn5=nn.Conv2d(96,96,(5,5),stride=1,padding=(8,2),dilation=(4,1))

        self.cnn6=nn.Conv2d(96,96,(5,5),stride=1,padding=(16,2),dilation=(8,1))
        self.cnn7=nn.Conv2d(96,96,(5,5),stride=1,padding=(32,2),dilation=(16,1))
        self.cnn8=nn.Conv2d(96,96,(5,5),stride=1,padding=(64,2),dilation=(32,1))
        self.cnn9=nn.Conv2d(96,96,(5,5),stride=1,padding=(2,2),dilation=(1,1))
        self.cnn10=nn.Conv2d(96,96,(5,5),stride=1,padding=(4,4),dilation=(2,2))

        self.cnn11=nn.Conv2d(96,96,(5,5),stride=1,padding=(8,8),dilation=(4,4))
        self.cnn12=nn.Conv2d(96,96,(5,5),stride=1,padding=(16,16),dilation=(8,8))
        self.cnn13=nn.Conv2d(96,96,(5,5),stride=1,padding=(32,32),dilation=(16,16))
        self.cnn14=nn.Conv2d(96,96,(5,5),stride=1,padding=(64,64),dilation=(32,32))
        self.cnn15=nn.Conv2d(96,8,(1,1),stride=1,padding=(0,0),dilation=(1,1))
        self.num_cnns=15
        self.bn1=nn.BatchNorm2d(96)
        self.bn2=nn.BatchNorm2d(96)
        self.bn3=nn.BatchNorm2d(96)
        self.bn4=nn.BatchNorm2d(96)
        self.bn5=nn.BatchNorm2d(96)
        self.bn6=nn.BatchNorm2d(96)
        self.bn7=nn.BatchNorm2d(96)
        self.bn8=nn.BatchNorm2d(96)
        self.bn9=nn.BatchNorm2d(96)
        self.bn10=nn.BatchNorm2d(96)
        self.bn11=nn.BatchNorm2d(96)
        self.bn12=nn.BatchNorm2d(96)
        self.bn13=nn.BatchNorm2d(96)
        self.bn14=nn.BatchNorm2d(96)
        self.bn15=nn.BatchNorm2d(8)

    def forward(self, x):
        print '\nSpeech layer log:'
        x = x.contiguous()
        for idx in range(self.num_cnns):
            cnn_layer=eval('self.cnn{}'.format(idx+1))
            bn_layer=eval('self.bn{}'.format(idx+1))
            x=F.relu(cnn_layer(x))
            x=bn_layer(x)
            print 'speech shape after CNNs:',idx,'', x.size()
        return x


class MULTI_MODAL(nn.Module):
    def __init__(self, speech_fre):
        super(MULTI_MODAL, self).__init__()
        self.fre=speech_fre
        print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'
        self.images_layer =FACE_EMB().cuda() #初始化处理各个任务的层
        self.mix_speech_layer = MIX_SPEECH().cuda()#初始化处理混合语音的层
        self.att_layer=ATTENTION(speech_fre).cuda() #做后端的融合和输出的层

    def forward(self, mix_speech,querys):
        mix_speech_hidden=self.mix_speech_layer(mix_speech)#bs,8,301,257
        mix_speech_hidden=torch.transpose(mix_speech_hidden,1,2).contiguous().view(config.BATCH_SIZE,self.fre,-1)
        # Todo:这里要经过一个变化，把８×２５７弄成一个维的
        querys_hidden=self.images_layer(querys)
        out=self.att_layer(mix_speech_hidden,querys_hidden)
        return out

def top_k_mask(batch_pro, alpha, top_k):
    'batch_pro是 bs*n的概率分布，例如2×3的，每一行是一个概率分布\
    alpha是阈值，大于它的才可以取，可以跟Multi-label语音分离的ACC的alpha对应;\
    top_k是最多输出几个候选目标\
    输出是与bs*n的一个mask，float型的'
    size = batch_pro.size()
    final = torch.zeros(size)
    sort_result, sort_index = torch.sort(batch_pro, 1, True)  # 先排个序
    sort_index = sort_index[:, :top_k]  # 选出每行的top_k的id
    sort_result = torch.sum(sort_result > alpha, 1)
    for line_idx in range(size[0]):
        line_top_k = sort_index[line_idx][:int(sort_result[line_idx].data.cpu().numpy())]
        line_top_k = line_top_k.data.cpu().numpy()
        for i in line_top_k:
            final[line_idx, i] = 1
    return final, sort_index


def main():
    print('go to model')
    print '*' * 80

    spk_global_gen = prepare_data(mode='global', train_or_test='train')
    global_para = spk_global_gen.next()
    print global_para
    #TODO:exchange speech_fea and mix_speech_len~! 没改之前　fre是３０１，len是２５７
    spk_all_list, dict_spk2idx, dict_idx2spk,speech_fre, mix_speech_len,\
    total_frames, spk_num_total, batch_total = global_para
    del spk_global_gen
    num_labels = len(spk_all_list)

    # print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'
    # images_layer =FACE_EMB() #初始化处理各个任务的层
    # images_layer(Variable(torch.rand(2,3,1024,75,1),requires_grad=True))
    # print images_layer.state_dict()
    # print images_layer.parameters().next()
    # print images_layer.state_dict().keys()
    # print images_layer.parameters().next()
    # mix_speech_layer = MIX_SPEECH().cuda()#初始化处理混合语音的层
    # mix_speech_layer(Variable(torch.rand(3,2,301,257)).cuda())
    # 1/0.
    # att_layer=ATTENTION(speech_fre) #做后端的融合和输出的层
    #
    # print images_layer
    # print mix_speech_layer
    # print att_layer

    print 'hhh'
    # speech_fre=257
    face_layer=FACE_HIDDEN_simple().cuda()

    model=MULTI_MODAL(speech_fre).cuda()
    print model
    print model.state_dict().keys()

    init_lr=0.0008
    optimizer = torch.optim.Adam([{'params':model.parameters()},
                                  {'params':face_layer.parameters()}], lr=init_lr)
    if 0 and config.Load_param:
        params_path='sss'
        model.load_state_dict(torch.load(params_path))
        print 'Params:',params_path, 'loaded successfully~!\n'

    loss_func = torch.nn.MSELoss().cuda()  # the target label is NOT an one-hotted
    loss_multi_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted

    print '''Begin to calculate.'''
    for epoch_idx in range(config.MAX_EPOCH):
        if epoch_idx > 0:
            print 'SDR_SUM (len:{}) for epoch {} : {}'.format(SDR_SUM.shape, epoch_idx - 1, SDR_SUM.mean())
        SDR_SUM = np.array([])
        batch_idx=0
        train_data_gen = prepare_data('once', 'train')
        # train_data_gen=prepare_data('once','test')
        # train_data_gen=prepare_data('once','eval_test')
        # for batch_idx in range(config.EPOCH_SIZE):
        while True:
            print '*' * 40, epoch_idx, batch_idx, '*' * 40
            train_data = train_data_gen.next()
            if train_data==False:
                break #如果这个epoch的生成器没有数据了，直接进入下一个epoch
            top_k_num=train_data['top_k'] #对于这个batch的top-k
            print 'top-k this batch:',top_k_num

            mix_speech_orignial=Variable(torch.from_numpy(train_data['mix_feas'])).cuda()
            mix_speech=torch.transpose(mix_speech_orignial,1,3)
            mix_speech=torch.transpose(mix_speech,2,3)
            # (2L, 301L, 257L, 2L) >>> (2L, 2L, 301L, 257L)
            print 'mix_speech_shape:',mix_speech.size()

            images_query=Variable(torch.from_numpy(convert2numpy(train_data['multi_video_list'],top_k_num))).cuda() #大小bs,topk,75,3,299,299
            print 'images_query_shape:',images_query.size()
            images_query=face_layer(images_query)
            print 'images_query_embeddings_shape:',images_query.size()
            y_map=convert2numpy(train_data['multi_spk_fea_list'],top_k_num) #最终的map
            print 'final map shape:',y_map.shape
            predict_multi_masks=model(mix_speech,images_query)
            print 'predict results shape:',predict_multi_masks.size() #(2L, topk, 301L, 257L, 2L)

            mix_speech_multi=mix_speech_orignial.view(config.BATCH_SIZE,1,speech_fre,mix_speech_len,2)\
                .expand(config.BATCH_SIZE,top_k_num,speech_fre,mix_speech_len,2)
            # (2L, 301L, 257L, 2L) >> (2L, topk,301L, 257L, 2L)

            predict_multi_masks_real=predict_multi_masks[:,:,:,:,0]
            predict_multi_masks_fake=predict_multi_masks[:,:,:,:,1]
            mix_speech_real=mix_speech_multi[:,:,:,:,0]
            mix_speech_fake=mix_speech_multi[:,:,:,:,1]
            y_map_real=Variable(torch.from_numpy(y_map[:,:,:,:,0])).cuda()
            y_map_fake=Variable(torch.from_numpy(y_map[:,:,:,:,1])).cuda()

            predict_real=predict_multi_masks_real*mix_speech_real-predict_multi_masks_fake*mix_speech_fake
            predict_fake=predict_multi_masks_real*mix_speech_fake+predict_multi_masks_fake*mix_speech_real
            print 'predict real/fake size:',predict_real.size()

            loss_real=loss_func(predict_real,y_map_real)
            loss_fake=loss_func(predict_fake,y_map_fake)
            loss_all=loss_real+loss_fake
            print 'loss:',loss_real.data[0],loss_fake.data[0]


            optimizer.zero_grad()   # clear gradients for next train
            loss_all.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            batch_idx+=1
            continue

            '''
            multi_mask = att_multi_speech
            # top_k_mask_mixspeech_multi=top_k_mask_mixspeech.view(config.BATCH_SIZE,top_k_num,1,1).expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre)
            # multi_mask=multi_mask*Variable(top_k_mask_mixspeech_multi).cuda()

            x_input_map = Variable(torch.from_numpy(train_data['mix_feas'])).cuda()
            # print x_input_map.size()
            x_input_map_multi = x_input_map.view(config.BATCH_SIZE, 1, mix_speech_len, speech_fre).expand(
                config.BATCH_SIZE, top_k_num, mix_speech_len, speech_fre)
            # predict_multi_map=multi_mask*x_input_map_multi
            predict_multi_map = multi_mask * x_input_map_multi

            bss_eval_fromGenMap(multi_mask, x_input_map, top_k_mask_mixspeech, dict_idx2spk, train_data,
                                top_k_sort_index)
            SDR_SUM = np.append(SDR_SUM, bss_test.cal('batch_output'+str(global_id)+'/', 2))
            print 'SDR_SUM (len:{}) for epoch {} : {}'.format(SDR_SUM.shape, epoch_idx, SDR_SUM.mean())
            '''
        if 1 and epoch_idx >= 10 and epoch_idx % 5 == 0:
            torch.save(model.state_dict(),'params/modelparams_{}_{}'.format(global_id,epoch_idx))
            # torch.save(face_layer.state_dict(),'params/faceparams_{}_{}'.format(global_id,epoch_idx))

        if 1 and epoch_idx % 3 == 0:
            eval_bss(mix_hidden_layer_3d,adjust_layer, mix_speech_classifier, mix_speech_multiEmbedding, att_speech_layer,
                     loss_multi_func, dict_spk2idx, dict_idx2spk, num_labels, mix_speech_len, speech_fre)
if __name__ == "__main__":
    main()
