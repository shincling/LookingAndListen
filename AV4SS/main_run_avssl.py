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

# import matlab
# import matlab.engine
# from separation import bss_eval_sources
# import bss_test


torch.cuda.set_device(0)
config.EPOCH_SIZE = 300
np.random.seed(1)  # 设定种子
torch.manual_seed(1)
random.seed(1)
# stout=sys.stdout
# log_file=open(config.LOG_FILE_PRE,'w')
# sys.stdout=log_file
# logfile=config.LOG_FILE_PRE
test_all_outputchannel = 0


def bss_eval_fromGenMap(multi_mask, x_input, top_k_mask_mixspeech, dict_idx2spk, data, sort_idx):
    if config.Out_Sep_Result:
        dst = 'batch_output'
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
            sf.write('batch_output/{}_testspk{}_pre.wav'.format(sample_idx, this_spk), wav_pre[:min_len],
                     config.FRAME_RATE, )
        sf.write('batch_output/{}_True_mix.wav'.format(sample_idx), data['mix_wav'][sample_idx][:min_len],
                 config.FRAME_RATE, )
        sample_idx += 1

    for sample_idx, each_sample in enumerate(data['multi_spk_wav_list']):
        for each_spk in each_sample.keys():
            this_spk = each_spk
            wav_genTrue = each_sample[this_spk]
            min_len = 39936
            sf.write('batch_output/{}_{}_realTrue.wav'.format(sample_idx, this_spk), wav_genTrue[:min_len],
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
            sf.write('batch_output/{}_{}_genTrue.wav'.format(sample_idx, this_spk), wav_genTrue[:min_len],
                     config.FRAME_RATE, )


def bss_eval(predict_multi_map, y_multi_map, y_map_gtruth, dict_idx2spk, train_data):
    # 评测和结果输出部分
    if config.Out_Sep_Result:
        dst = 'batch_output'
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
                sf.write('batch_output/{}_{}_pre.wav'.format(sample_idx, this_spk), wav_pre[:min_len],
                         config.FRAME_RATE, )
                sf.write('batch_output/{}_{}_genTrue.wav'.format(sample_idx, this_spk), wav_genTrue[:min_len],
                         config.FRAME_RATE, )
        sf.write('batch_output/{}_True_mix.wav'.format(sample_idx), train_data['mix_wav'][sample_idx][:min_len],
                 config.FRAME_RATE, )
        sample_idx += 1


def print_memory_state(memory):
    print '\n memory states:'
    for one in memory:
        print '\nspk:{},video:{},age:{}'.format(one[0], one[1].cpu().numpy()[
                                                        2 * config.EMBEDDING_SIZE - 3:2 * config.EMBEDDING_SIZE + 5],
                                                one[2])

def convert2numpy(data_list,top_k):
    output_size=(config.BATCH_SIZE,top_k)+ np.array(data_list[0].values[0]).shape
    output_array=np.zeros(output_size)
    for idx,dict_sample in enumerate(data_list):#对于这个batch里的每一个sample(是一个dict)
        spk_all=sorted(dict_sample.keys()) #将它们排序，确保统一
        for jdx,spk in enumerate(spk_all):
            output_array[idx,jdx]=np.array(data_list[idx][spk])
    return output_array

class ATTENTION(nn.Module):
    def __init__(self,speech_fre):
        self.fre=speech_fre#应该是257
        super(ATTENTION, self).__init__()
        self.lstm_layer = nn.LSTM(
            input_size=(8*257+256),
            hidden_size=config.HIDDEN_UNITS,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )
        self.fc_layers=[nn.Linear(2*config.HIDDEN_UNITS,config.FC_UNITS) for i in range(3)]
        self.final_layer=nn.Linear(config.FC_UNITS,speech_fre*2)

    def forward(self, mix_hidden, query):
        # todo:这个要弄好，其实也可以直接抛弃memory来进行attention | DONE
        mix_shape=mix_hidden.size()#应该是bs*1*298*(8*257)的东西
        query_shape=query.size()#应该是bs*topk*1*298*256的东西
        top_k=query_shape[1]
        BATCH_SIZE = mix_hidden.size()[0]
        # assert query.size()==(BATCH_SIZE,self.hidden_size)
        # assert mix_hidden.size()[-1]==self.hidden_size
        # mix_hidden：bs,max_len,fre,hidden_size  query:bs,hidden_size

        mix_hidden = mix_hidden.view(BATCH_SIZE, 1,  mix_shape[2], mix_shape[3]).expand(-1,top_k,-1,-1)
        mix_hidden = mix_hidden.view(BATCH_SIZE*top_k, mix_shape[2], mix_shape[3]) #现在mix_hidden变成了(bs*topk)*298*(8*257)
        query=query.view(BATCH_SIZE*top_k, mix_shape[2], mix_shape[3]) #现在query变成了(bs*topk)*298*256
        multi_moda=torch.cat((mix_hidden,query),2) #得到了拼接好的特征矩阵

        multi_moda=self.lstm_layer(multi_moda)
        for la in self.fc_layers:
            multi_moda=F.relu(la(multi_moda))
        print 'The size of last embedding:',multi_moda.size() #应该是(bs*topk),298,600
        results=self.final_layer(multi_moda).view(BATCH_SIZE,top_k,mix_shape[2],2,self.fre)
        results=F.sigmoid(results)
        return results


class FACE_EMB(nn.Module):
    def __init__(self):
        super(FACE_EMB, self).__init__()
        self.cnn_list=[
            nn.Conv2d(1024,256,(7,1),stride=1,padding=(7,0),dilation=(1,1)),
            nn.Conv2d(256,256,(5,1),stride=1,padding=(9,0),dilation=(1,1)),
            nn.Conv2d(256,256,(5,1),stride=1,padding=(13,0),dilation=(2,1)),
            nn.Conv2d(256,256,(5,1),stride=1,padding=(21,0),dilation=(4,1)),
            nn.Conv2d(256,256,(5,1),stride=1,padding=(37,0),dilation=(8,1)),
            # nn.Conv2d(256,256,(5,1),stride=1,padding=(69,0),dilation=(16,1)),
        ]
        self.cnn_list1=[
            nn.ConvTranspose2d(1024,256,(7,1),stride=1, dilation=(1,1)),
            nn.ConvTranspose2d(256,256,(5,1),stride=1, dilation=(1,1)),
            nn.ConvTranspose2d(256,256,(5,1),stride=1, dilation=(2,1)),
            nn.ConvTranspose2d(256,256,(5,1),stride=1, dilation=(4,1)),
            nn.ConvTranspose2d(256,256,(5,1),stride=1, dilation=(8,1)),
            nn.ConvTranspose2d(256,256,(5,1),stride=1, dilation=(16,1)),
        ]

        self.cnn1=nn.Conv2d(1024,256,(7,1),stride=1,padding=(7,0),dilation=(1,1))
        self.cnn2=nn.Conv2d(256,256,(5,1),stride=1,padding=(9,0),dilation=(1,1))
        self.cnn3=nn.Conv2d(256,256,(5,1),stride=1,padding=(13,0),dilation=(2,1))
        self.cnn4=nn.Conv2d(256,256,(5,1),stride=1,padding=(21,0),dilation=(4,1))
        self.cnn5=nn.Conv2d(256,256,(5,1),stride=1,padding=(37,0),dilation=(8,1))
        self.cnn6=nn.Conv2d(256,256,(5,1),stride=1,padding=(69,0),dilation=(16,1))
        self.num_cnns=6

    def forward(self, x):
        #　这个时候的输入应该是　bs*top-k*1024个通道*75帧×１
        shape=x.size()
        x = x.contiguous()
        x = x.view(-1,x.size(2),x.size(3),1)
        for idx in range(self.num_cnns):
            cnn_layer=eval('self.cnn{}'.format(idx+1))
            x=F.relu(cnn_layer(x))
            # x=F.batch_norm(x,0,1)
            print 'speech shape after CNNs:',idx,'', x.size()

        # return x.view(shape)
        return x

class MIX_SPEECH(nn.Module):
    def __init__(self):
        super(MIX_SPEECH, self).__init__()

        self.cnn1=nn.Conv2d(2,96,(1,7),stride=1,padding=(1,7),dilation=(1,1))
        self.cnn2=nn.Conv2d(96,96,(7,1),stride=1,padding=(7,1),dilation=(1,1))
        self.cnn3=nn.Conv2d(96,96,(5,5),stride=1,padding=(9,9),dilation=(1,1))
        self.cnn4=nn.Conv2d(96,96,(5,5),stride=1,padding=(13,11),dilation=(2,1))
        self.cnn5=nn.Conv2d(96,96,(5,5),stride=1,padding=(21,13),dilation=(4,1))

        self.cnn6=nn.Conv2d(96,96,(5,5),stride=1,padding=(37,15),dilation=(8,1))
        self.cnn7=nn.Conv2d(96,96,(5,5),stride=1,padding=(69,17),dilation=(16,1))
        self.cnn8=nn.Conv2d(96,96,(5,5),stride=1,padding=(133,19),dilation=(32,1))
        self.cnn9=nn.Conv2d(96,96,(5,5),stride=1,padding=(135,21),dilation=(1,1))
        self.cnn10=nn.Conv2d(96,96,(5,5),stride=1,padding=(139,25),dilation=(2,2))

        self.cnn11=nn.Conv2d(96,96,(5,5),stride=1,padding=(147,33),dilation=(4,4))
        self.cnn12=nn.Conv2d(96,96,(5,5),stride=1,padding=(163,49),dilation=(8,8))
        self.cnn13=nn.Conv2d(96,96,(5,5),stride=1,padding=(195,81),dilation=(16,16))
        self.cnn14=nn.Conv2d(96,96,(5,5),stride=1,padding=(259,145),dilation=(32,32))
        self.cnn15=nn.Conv2d(96,8,(1,1),stride=1,padding=(259,145),dilation=(1,1))
        self.num_cnns=15

    def forward(self, x):
        x = x.contiguous()
        for idx in range(self.num_cnns):
            cnn_layer=eval('self.cnn{}'.format(idx+1))
            x=F.relu(cnn_layer(x))
            # x=F.batch_norm(x,0,1)
            print 'speech shape after CNNs:',idx,'', x.size()
        return x


class MULTI_MODAL(nn.Module):
    def __init__(self, speech_fre):
        super(MULTI_MODAL, self).__init__()
        print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'
        self.images_layer =FACE_EMB().cuda() #初始化处理各个任务的层
        self.mix_speech_layer = MIX_SPEECH().cuda()#初始化处理混合语音的层
        self.att_layer=ATTENTION(speech_fre).cuda() #做后端的融合和输出的层

    def forward(self, mix_speech,querys):
        mix_speech_hidden=self.mix_speech_layer(mix_speech)
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
    spk_all_list, dict_spk2idx, dict_idx2spk, mix_speech_len, speech_fre,\
    total_frames, spk_num_total, batch_total = global_para
    del spk_global_gen
    num_labels = len(spk_all_list)

    # print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'
    # images_layer =FACE_EMB() #初始化处理各个任务的层
    # print images_layer.state_dict()
    # print images_layer.parameters().next()
    # images_layer(Variable(torch.rand(3,2,1024,75,1)))
    # print images_layer.state_dict().keys()
    # print images_layer.parameters().next()
    # 1/0.
    # mix_speech_layer = MIX_SPEECH().cuda()#初始化处理混合语音的层
    # att_layer=ATTENTION(speech_fre) #做后端的融合和输出的层
    #
    # print images_layer
    # print mix_speech_layer
    # print att_layer

    print 'hhh'
    speech_fre=257
    model=MULTI_MODAL(speech_fre).cuda()
    print model
    print model.state_dict().keys()
    1/0

    init_lr=0.0008
    optimizer = torch.optim.Adam([{'params':model.parameters()}], lr=init_lr)
    if 0 and config.Load_param:
        params_path='sss'
        model.load_state_dict(torch.load(params_path))
        print 'Params:',params_path, 'loaded successfully~!\n'

    loss_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted
    loss_multi_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted

    print '''Begin to calculate.'''
    for epoch_idx in range(config.MAX_EPOCH):
        if epoch_idx > 0:
            print 'SDR_SUM (len:{}) for epoch {} : {}'.format(SDR_SUM.shape, epoch_idx - 1, SDR_SUM.mean())
        SDR_SUM = np.array([])
        for batch_idx in range(config.EPOCH_SIZE):
            print '*' * 40, epoch_idx, batch_idx, '*' * 40
            train_data_gen = prepare_data('once', 'train')
            # train_data_gen=prepare_data('once','test')
            # train_data_gen=prepare_data('once','eval_test')
            train_data = train_data_gen.next()

            mix_speech=Variable(torch.from_numpy(train_data['mix_feas'])).cuda()
            images_query=Variable(torch.from_numpy(convert2numpy(train_data['multi_video_list']))).cuda() #大小bs,topk,75,3,299,299
            y_map=convert2numpy(train_data['multI_spk_fea_list']) #最终的map

            predict_multi_masks=model(mix_speech,images_query)








            '''混合语音len,fre,Emb 3D表示层'''
            mix_speech_hidden = mix_hidden_layer_3d(Variable(torch.from_numpy(train_data['mix_feas'])).cuda())
            # 暂时关掉video部分,因为s2 s3 s4 的视频数据不全暂时

            '''Speech self Sepration　语音自分离部分'''
            mix_speech_output = mix_speech_classifier(Variable(torch.from_numpy(train_data['mix_feas'])).cuda())
            # 从数据里得到ground truth的说话人名字和vector
            y_spk_list = [one.keys() for one in train_data['multi_spk_fea_list']]
            y_spk_list = train_data['multi_spk_fea_list']
            y_spk_gtruth, y_map_gtruth = multi_label_vector(y_spk_list, dict_spk2idx)
            # 如果训练阶段使用Ground truth的分离结果作为判别
            if 1 and config.Ground_truth:
                mix_speech_output = Variable(torch.from_numpy(y_map_gtruth)).cuda()
                if 0 and test_all_outputchannel:  # 把输入的mask改成全１，可以用来测试输出所有的channel
                    mix_speech_output = Variable(torch.ones(config.BATCH_SIZE, num_labels, ))
                    y_map_gtruth = np.ones([config.BATCH_SIZE, num_labels])

            max_num_labels = 2
            top_k_mask_mixspeech, top_k_sort_index = top_k_mask(mix_speech_output, alpha=-0.5,
                                                                top_k=max_num_labels)  # torch.Float型的
            top_k_mask_idx = [np.where(line == 1)[0] for line in top_k_mask_mixspeech.numpy()]
            mix_speech_multiEmbs = mix_speech_multiEmbedding(top_k_mask_mixspeech,
                                                             top_k_mask_idx)  # bs*num_labels（最多混合人个数）×Embedding的大小

            assert len(top_k_mask_idx[0]) == len(top_k_mask_idx[-1])
            top_k_num = len(top_k_mask_idx[0])

            # 需要计算：mix_speech_hidden[bs,len,fre,emb]和mix_mulEmbedding[bs,num_labels,EMB]的Ａttention
            # 把　前者扩充为bs*num_labels,XXXXXXXXX的，后者也是，然后用ＡＴＴ函数计算它们再转回来就好了　
            mix_speech_hidden_5d = mix_speech_hidden.view(config.BATCH_SIZE, 1, mix_speech_len, speech_fre,
                                                          config.EMBEDDING_SIZE)
            mix_speech_hidden_5d = mix_speech_hidden_5d.expand(config.BATCH_SIZE, top_k_num, mix_speech_len, speech_fre,
                                                               config.EMBEDDING_SIZE).contiguous()
            mix_speech_hidden_5d_last = mix_speech_hidden_5d.view(-1, mix_speech_len, speech_fre, config.EMBEDDING_SIZE)
            # att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'align').cuda()
            att_speech_layer = ATTENTION(config.EMBEDDING_SIZE, 'dot').cuda()
            att_multi_speech = att_speech_layer(mix_speech_hidden_5d_last,
                                                mix_speech_multiEmbs.view(-1, config.EMBEDDING_SIZE))
            # print att_multi_speech.size()
            att_multi_speech = att_multi_speech.view(config.BATCH_SIZE, top_k_num, mix_speech_len,
                                                     speech_fre)  # bs,num_labels,len,fre这个东西
            # print att_multi_speech.size()
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
            SDR_SUM = np.append(SDR_SUM, bss_test.cal('batch_output/', 2))
            print 'SDR_SUM (len:{}) for epoch {} : {}'.format(SDR_SUM.shape, epoch_idx, SDR_SUM.mean())


if __name__ == "__main__":
    main()
