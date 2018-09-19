#coding=utf8
import sys
import os
import numpy as np
import time
import random
import config as config
import re
import soundfile as sf
import resampy
import librosa
import shutil
import subprocess
# import Image
from PIL import Image

channel_first=config.channel_first
# np.random.seed(1)#设定种子
# random.seed(1)

def extract_frames(video, dst):
    with open('video_log', "w") as ffmpeg_log:
        video_id = video.split("/")[-1].split(".")[0]
        if os.path.exists(dst):
            print " cleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   '-y',  # (optional) overwrite output file if it exists
                                   '-i', video,  # input file
                                   '-vf', "scale={}:{}".format(config.VideoSize[0],config.VideoSize[1]),  # input file
                                   '-r', str(config.VIDEO_RATE),  # samplling rate of the Video
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%03d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command, stdout=ffmpeg_log, stderr=ffmpeg_log)

def split_forTrainDevTest(spk_list,train_or_test):
    '''为了保证一个统一的训练和测试的划分标准，不得不用通用的一些方法来限定一下,
    这里采用的是用sorted先固定方法的排序，那么不论方法或者seed怎么设置，训练测试的划分标准维持不变，
    也就是数据集会维持一直'''
    length=len(spk_list)
    # spk_list=sorted(spk_list,key=lambda x:(x[1]))#这个意思是按照文件名的第二个字符排序
    # spk_list=sorted(spk_list)#这个意思是按照文件名的第1个字符排序,暂时采用这种
    spk_list=sorted(spk_list,key=lambda x:(x[-1]))#这个意思是按照文件名的最后一个字符排序
    #TODO:暂时用最后一个字符排序，这个容易造成问题，可能第一个比较不一样的，这个需要注意一下
    if train_or_test=='train':
        return spk_list[:int(round(0.7*length))]
    elif train_or_test=='valid':
        return spk_list[(int(round(0.7*length))+1):int(round(0.8*length))]
    elif train_or_test=='test':
        return spk_list[(int(round(0.8*length))+1):]
    else:
        raise ValueError('Wrong input of train_or_test.')

def prepare_datasize(gen):
    data=gen.next()
    #此处顺序是 mix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkid.shape,query.shape
    #一个例子：(5, 17040) (5, 134, 129) (5, 134, 129) (5,) (5, 32, 400, 300, 3)
    #暂时输出的是：语音长度、语音频率数量、视频截断之后的长度
    print 'datasize:',data[1].shape[1],data[1].shape[2],data[4].shape[1],data[-1],(data[4].shape[2],data[4].shape[3])
    return data[1].shape[1],data[1].shape[2],data[4].shape[1],data[-1],(data[4].shape[2],data[4].shape[3])

def create_mix_list(train_or_test,mix_k,data_path,all_spk,Num_samples_per_batch):
    list_path=data_path+'/list_mixtures/'
    file_name=open(list_path+'faceemb_mix_{}_spk_{}.txt'.format(mix_k,train_or_test),'w')

    for i_line in range(Num_samples_per_batch):
        aim_spk_k=random.sample(all_spk,mix_k)#本次混合的候选人
        line=''
        for spk in aim_spk_k:
            sample_name=random.sample(os.listdir('{}/face_emb/s2-s35/{}_imgnpy/'.format(data_path,spk)),1)[0]
            sample_name=sample_name.replace('npy','wav')
            line+='GRID/data/face_emb/voice/{}/{} 0.000 '.format(spk,sample_name)
        line+='\n'
        file_name.write(line)

def convert2(array):
    shape=array.shape
    o=array.real.reshape(shape[0],shape[1],1).repeat(2,2)
    o[:,:,1]=array.imag
    return o

def prepare_data(mode,train_or_test,min=None,max=None):
    '''
    :param
    mode: type str, 'global' or 'once' ， global用来获取全局的spk_to_idx的字典，所有说话人的列表等等
    train_or_test:type str, 'train','valid' or 'test'
     其中把每个文件夹每个人的按文件名的排序的前70%作为训练，70-80%作为valid，最后20%作为测试
    :return:
    '''
    # 如错有预订的min和max，主要是为了主程序做valid的时候统一某个固定的说话人的个数上
    if min:
        config.MIN_MIX=min
    if max:
        config.MAX_MIX=max

    mix_speechs=np.zeros((config.BATCH_SIZE,config.MAX_LEN_SPEECH))
    mix_feas=[]#应该是bs,n_frames,n_fre这么多
    mix_phase=[]#应该是bs,n_frames,n_fre这么多
    aim_fea=[]#应该是bs,n_frames,n_fre这么多
    aim_spkid=[] #np.zeros(config.BATCH_SIZE)
    aim_spkname=[] #np.zeros(config.BATCH_SIZE)
    query=[]#应该是BATCH_SIZE，shape(query)的形式，用list再转换把
    multi_spk_fea_list=[] #应该是bs个dict，每个dict里是说话人name为key，clean_fea为value的字典
    multi_spk_wav_list=[] #应该是bs个dict，每个dict里是说话人name为key，clean_fea为value的字典
    multi_video_list=[] #应该是bs个dict，每个dict里是说话人name为key，clean_fea为value的字典
    multi_video_fea_list=[] #应该是bs个dict，每个dict里是说话人name为key，clean_fea为value的字典

    #目标数据集的总data，底下应该存放分目录的文件夹，每个文件夹应该名字是sX
    data_path=config.aim_path+'/data'
    #语音刺激
    if config.MODE==3:
        if config.DATASET=='GRID': #开始构建数据集
            all_spk_train=os.listdir(data_path+'/train')
            all_spk_eval=os.listdir(data_path+'/valid')
            all_spk_test=os.listdir(data_path+'/test')
            if train_or_test=='train':
                all_spk_type=all_spk_train
            elif train_or_test=='valid':
                all_spk_type=all_spk_eval
            elif train_or_test=='test':
                all_spk_type=all_spk_test
            all_spk = set(all_spk_train+all_spk_eval+all_spk_test).union()

            all_spk = os.listdir((data_path+'/face_emb/s2-s35/'))
            all_spk = [spk.replace('_imgnpy','') for spk in all_spk]

            batch_idx=0
            list_path=data_path+'/list_mixtures/'
            all_samples_list={}
            sample_idx={}
            number_samples={}
            batch_mix={}
            mix_number_list=range(config.MIN_MIX,config.MAX_MIX+1)
            number_samples_all=0
            for mix_k in mix_number_list:
                aim_list_path=None
                if config.TRAIN_LIST and train_or_test=='train':
                    aim_list_path=list_path+'faceemb_mix_{}_spk_train.txt'.format(mix_k)
                if config.VALID_LIST and train_or_test=='valid':
                    aim_list_path=list_path+'faceemb_mix_{}_spk_valid.txt'.format(mix_k)
                if config.TEST_LIST and train_or_test=='test':
                    aim_list_path=list_path+'faceemb_mix_{}_spk_test.txt'.format(mix_k)
                if not aim_list_path: #如果没有List就随机创建一个
                    create_mix_list(train_or_test,mix_k,data_path,all_spk,config.Num_samples_per_epoch)
                    if  train_or_test=='train':
                        aim_list_path=list_path+'faceemb_mix_{}_spk_train.txt'.format(mix_k)
                    if  train_or_test=='valid':
                        aim_list_path=list_path+'faceemb_mix_{}_spk_valid.txt'.format(mix_k)
                    if train_or_test=='test':
                        aim_list_path=list_path+'faceemb_mix_{}_spk_test.txt'.format(mix_k)

                all_samples_list[mix_k]=open(aim_list_path).readlines()#[:10]
                number_samples[mix_k]=len(all_samples_list[mix_k])
                batch_mix[mix_k]=len(all_samples_list[mix_k])/config.BATCH_SIZE
                number_samples_all+=len(all_samples_list[mix_k])

                sample_idx[mix_k]=0#每个通道从0开始计数

                if config.SHUFFLE_BATCH:
                    random.shuffle(all_samples_list[mix_k])
                    print '\nshuffle success!',all_samples_list[mix_k][0]

            if number_samples_all==0:
                print '*'*10,'There is no lists setted. Begin to sample randomly.'
                number_samples_all=config.Num_samples_per_epoch

            batch_total=number_samples_all/config.BATCH_SIZE
            print 'batch_total_num:',batch_total

            mix_k=random.sample(mix_number_list,1)[0] #第一个batch里的混合个数
            for ___ in range(number_samples_all):
                if ___==number_samples_all-1:
                    print 'ends here.___'
                    yield False
                mix_len=0
                print mix_k,'mixed sample_idx[mix_k]:',sample_idx[mix_k],batch_idx
                if sample_idx[mix_k]>=batch_mix[mix_k]*config.BATCH_SIZE:
                    print mix_k,'mixed data is over~trun to the others number.'
                    mix_number_list.remove(mix_k)
                    try:
                        mix_k=random.sample(mix_number_list,1)[0]
                    except ValueError:
                        print 'seems there gets all over.'
                        if len(mix_number_list)==0:
                            print 'all mix number is over~!'
                        yield False
                    # mix_k=random.sample(mix_number_list,1)[0]
                    mix_k=random.sample(mix_number_list,1)[0]
                    batch_idx=0
                    mix_speechs=np.zeros((config.BATCH_SIZE,config.MAX_LEN_SPEECH))
                    mix_feas=[]#应该是bs,n_frames,n_fre这么多
                    mix_phase=[]
                    aim_fea=[]#应该是bs,n_frames,n_fre这么多
                    aim_spkid=[] #np.zeros(config.BATCH_SIZE)
                    aim_spkname=[]
                    query=[]#应该是BATCH_SIZE，shape(query)的形式，用list再转换把
                    multi_spk_fea_list=[]
                    multi_spk_wav_list=[]
                    multi_video_list=[]
                    multi_video_fea_list=[]
                    continue

                all_over=1 #用来判断所有的是不是都结束了
                for kkkkk in mix_number_list:
                    if not sample_idx[kkkkk]>=batch_mix[mix_k]*config.BATCH_SIZE:
                        print kkkkk,'mixed data is not over'
                        all_over=0
                        break
                    if all_over:
                        print 'all mix number is over~!'
                        yield False

                if not aim_list_path: #如果没有候选list的话，就每次自己随机sample
                    if train_or_test=='train':
                        aim_spk_k=random.sample(all_spk_train,mix_k)#本次混合的候选人
                    elif train_or_test=='eval':
                        aim_spk_k=random.sample(all_spk_eval,mix_k)#本次混合的候选人
                    elif train_or_test=='test':
                        aim_spk_k=random.sample(all_spk_test,mix_k)#本次混合的候选人
                    aim_spk_db_k=[0]*mix_k

                else:
                    aim_spk_k=re.findall('/(.{2,4})/.{6}\.wav ',all_samples_list[mix_k][sample_idx[mix_k]])
                    aim_spk_db_k=map(float,re.findall(' (.*?) ',all_samples_list[mix_k][sample_idx[mix_k]]))
                    aim_spk_samplename_k=re.findall('/(.{6})\.wav ',all_samples_list[mix_k][sample_idx[mix_k]])
                    assert len(aim_spk_k)==mix_k==len(aim_spk_db_k)==len(aim_spk_samplename_k)

                multi_fea_dict_this_sample={}
                multi_wav_dict_this_sample={}
                multi_video_dict_this_sample={}
                multi_video_fea_dict_this_sample={}
                multi_db_dict_this_sample={}

                for k,spk in enumerate(aim_spk_k):
                    #选择dB的通道～！
                    sample_name=aim_spk_samplename_k[k]
                    # if train_or_test!='test':
                    #     spk_speech_path=data_path+'/'+'train'+'/'+spk+'/'+spk+'_speech/'+sample_name+'.wav'
                    # else:
                    #     spk_speech_path=data_path+'/'+'eval_test'+'/'+spk+'/'+spk+'_speech/'+sample_name+'.wav'

                    spk_speech_path=data_path+'/'+'face_emb/voice'+'/'+spk+'/'+sample_name+'.wav'

                    signal, rate = sf.read(spk_speech_path)  # signal 是采样值，rate 是采样频率
                    if len(signal.shape) > 1:
                        signal = signal[:, 0]
                    if rate != config.FRAME_RATE:
                        # 如果频率不是设定的频率则需要进行转换
                        signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
                    if signal.shape[0] > config.MAX_LEN_SPEECH:  # 根据最大长度裁剪
                        signal = signal[:config.MAX_LEN_SPEECH]
                    # 更新混叠语音长度
                    if signal.shape[0] > mix_len:
                        mix_len = signal.shape[0]

                    signal -= np.mean(signal)  # 语音信号预处理，先减去均值
                    signal /= np.max(np.abs(signal))  # 波形幅值预处理，幅值归一化

                    # 如果需要augment数据的话，先进行随机shift, 以后考虑固定shift
                    if config.AUGMENT_DATA and train_or_test=='train':
                        random_shift = random.sample(range(len(signal)), 1)[0]
                        signal = np.append(signal[random_shift:], signal[:random_shift])

                    if signal.shape[0] < config.MAX_LEN_SPEECH:  # 根据最大长度用 0 补齐,
                        signal=np.append(signal,np.zeros(config.MAX_LEN_SPEECH - signal.shape[0]))

                    if k==0:#第一个作为目标
                        ratio=10**(aim_spk_db_k[k]/20.0)
                        signal=ratio*signal
                        aim_spkname.append(aim_spk_k[0])
                        # aim_spk=eval(re.findall('\d+',aim_spk_k[0])[0])-1 #选定第一个作为目标说话人
                        #TODO:这里有个问题是spk是从１开始的貌似，这个后面要统一一下　-->　已经解决，构建了spk和idx的双向索引
                        aim_spk_speech=signal
                        aim_spkid.append(aim_spkname)
                        wav_mix=signal
                        # print signal.shape
                        aim_fea_clean = np.transpose((librosa.core.spectrum.stft(signal, config.FFT_SIZE, config.HOP_LEN,
                                                                                    config.WIN_LEN)))
                        aim_fea_clean=convert2(aim_fea_clean)
                        # print aim_fea_clean.shape
                        #TODO:这个实现出来跟原文不太一样啊，是２５７×３０１（原文是２９８）
                        aim_fea.append(aim_fea_clean)
                        # 把第一个人顺便也注册进去混合dict里
                        multi_fea_dict_this_sample[spk]=aim_fea_clean
                        multi_wav_dict_this_sample[spk]=signal

                        #视频处理部分，为了得到query
                        '''
                        aim_spk_video_path=data_path+'/'+train_or_test+'/'+spk+'/'+spk+'_video/'+sample_name+'.mpg'
                        sample_name='video_output/'+sample_name
                        extract_frames(aim_spk_video_path,sample_name) #抽取frames从第一个目标人的视频里,在本目录下生成一个临时的文件夹
                        aim_video_imagename_list = sorted(os.listdir(sample_name)) #得到这个文件夹里的所有图像的名字
                        aim_video_image_list=[]#用来存放这些抽出来帧的images的列表，后面转化为array

                        if len(aim_video_imagename_list)>config.MAX_LEN_VIDEO:
                            aim_video_imagename_list=aim_video_imagename_list[:config.MAX_LEN_VIDEO]
                        if len(aim_video_imagename_list)<config.MAX_LEN_VIDEO:#视频短了，用最后一张补齐。
                            aim_video_imagename_list.extend([aim_video_imagename_list[-1] for jj in range(config.MAX_LEN_VIDEO-len(aim_video_imagename_list))])
                            
                        for img in aim_video_imagename_list:
                            im=Image.open(sample_name+'/'+img)
                            pix=im.load()
                            width,height=im.size
                            #此处用来决定三个通道维度上的先后顺序是x,y,3还是3,x,y
                            if not channel_first:
                                im_array=np.zeros([width,height,3],dtype=np.float32)
                                for x in range(width):
                                    for y in range(height):
                                        im_array[x,y]=pix[x,y]
                            else:
                                im_array=np.zeros([3,width,height],dtype=np.float32)
                                for x in range(width):
                                    for y in range(height):
                                        im_array[:,x,y]=pix[x,y]
                            aim_video_image_list.append(im_array)
                        multi_video_dict_this_sample[spk]=aim_video_image_list
                        multi_video_fea_dict_this_sample[spk]=aim_video_image_list
                        shutil.rmtree(sample_name)#删除临时文件夹
                        '''
                        aim_spk_fea_video_path=data_path+'/face_emb/s2-s35/'+spk+'_imgnpy/'+sample_name+'.npy'
                        multi_video_fea_dict_this_sample[spk]=np.load(aim_spk_fea_video_path)

                    else:
                        ratio=10**(aim_spk_db_k[k]/20.0)
                        signal=ratio*signal
                        wav_mix = wav_mix + signal  # 混叠后的语音
                        #　这个说话人的语音
                        some_fea_clean = np.transpose((librosa.core.spectrum.stft(signal, config.FFT_SIZE, config.HOP_LEN,
                                                                                       config.WIN_LEN)))
                        some_fea_clean=convert2(some_fea_clean)
                        multi_fea_dict_this_sample[spk]=some_fea_clean
                        multi_wav_dict_this_sample[spk]=signal


                        #视频处理部分，为了得到query
                        '''
                        aim_spk_video_path=data_path+'/'+train_or_test+'/'+spk+'/'+spk+'_video/'+sample_name+'.mpg'
                        dst='video_output/'+sample_name
                        sample_name=dst
                        extract_frames(aim_spk_video_path,dst) #抽取frames从第一个目标人的视频里,在本目录下生成一个临时的文件夹
                        aim_video_imagename_list = sorted(os.listdir(dst)) #得到这个文件夹里的所有图像的名字
                        aim_video_image_list=[]#用来存放这些抽出来帧的images的列表，后面转化为array

                        if len(aim_video_imagename_list)>config.MAX_LEN_VIDEO:
                            aim_video_imagename_list=aim_video_imagename_list[:config.MAX_LEN_VIDEO]
                        if len(aim_video_imagename_list)<config.MAX_LEN_VIDEO:#视频短了，用最后一张补齐。
                            aim_video_imagename_list.extend([aim_video_imagename_list[-1] for jj in range(config.MAX_LEN_VIDEO-len(aim_video_imagename_list))])

                        for img in aim_video_imagename_list:
                            im=Image.open(sample_name+'/'+img)
                            pix=im.load()
                            width,height=im.size
                            #此处用来决定三个通道维度上的先后顺序是x,y,3还是3,x,y
                            if not channel_first:
                                im_array=np.zeros([width,height,3],dtype=np.float32)
                                for x in range(width):
                                    for y in range(height):
                                        im_array[x,y]=pix[x,y]
                            else:
                                im_array=np.zeros([3,width,height],dtype=np.float32)
                                for x in range(width):
                                    for y in range(height):
                                        im_array[:,x,y]=pix[x,y]
                            aim_video_image_list.append(im_array)
                        multi_video_dict_this_sample[spk]=aim_video_image_list
                        shutil.rmtree(sample_name)#删除临时文件夹
                        '''

                        aim_spk_fea_video_path=data_path+'/face_emb/s2-s35/'+spk+'_imgnpy/'+sample_name+'.npy'
                        multi_video_fea_dict_this_sample[spk]=np.load(aim_spk_fea_video_path)


                multi_spk_fea_list.append(multi_fea_dict_this_sample) #把这个sample的dict传进去
                multi_spk_wav_list.append(multi_wav_dict_this_sample) #把这个sample的dict传进去
                multi_video_list.append(multi_video_dict_this_sample) #把这个sample的dict传进去
                multi_video_fea_list.append(multi_video_fea_dict_this_sample) #把这个sample的dict传进去

                # 这里采用log 以后可以考虑采用MFCC或GFCC特征做为输入
                if config.IS_LOG_SPECTRAL:
                    feature_mix = np.log(np.transpose((librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                        config.FRAME_SHIFT,
                                                                                        window=config.WINDOWS)))
                                         + np.spacing(1))
                else:
                    feature_mix = np.transpose((librosa.core.spectrum.stft(wav_mix, config.FFT_SIZE, config.HOP_LEN,
                                                                                    config.WIN_LEN,)))
                    feature_mix=convert2(feature_mix)

                mix_speechs[batch_idx,:]=wav_mix
                mix_feas.append(feature_mix)
                # mix_phase.append(np.transpose(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                #                                                                      config.FRAME_SHIFT,)))
                mix_phase.append(np.transpose(librosa.core.spectrum.stft(wav_mix, config.FFT_SIZE, config.HOP_LEN,
                                                                         config.WIN_LEN,)))
                batch_idx+=1
                # print 'batch_dix:{}/{},'.format(batch_idx,config.BATCH_SIZE),
                if batch_idx==config.BATCH_SIZE: #填满了一个batch
                    mix_feas=np.array(mix_feas)
                    mix_phase=np.array(mix_phase)
                    aim_fea=np.array(aim_fea)
                    # aim_spkid=np.array(aim_spkid)
                    query=np.array(query)
                    print 'spk_list_from_this_gen:{}'.format(aim_spkname)
                    print 'aim spk list:', [one.keys() for one in multi_spk_fea_list]
                    # print '\nmix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkname.shape,query.shape,all_spk_num:'
                    # print mix_speechs.shape,mix_feas.shape,aim_fea.shape,len(aim_spkname),query.shape,len(all_spk)
                    if mode=='global':
                        all_spk=sorted(all_spk)
                        all_spk=sorted(all_spk_train)
                        all_spk_eval=sorted(all_spk_eval)
                        all_spk_test=sorted(all_spk_test)
                        dict_spk_to_idx={spk:idx for idx,spk in enumerate(all_spk)}
                        dict_idx_to_spk={idx:spk for idx,spk in enumerate(all_spk)}
                        yield all_spk,dict_spk_to_idx,dict_idx_to_spk,\
                              aim_fea.shape[1],aim_fea.shape[2],config.MAX_LEN_VIDEO,len(all_spk),batch_total
                              #上面的是：语音长度、语音频率、视频分割多少帧 TODO:后面把这个替换了query.shape[1]
                    elif mode=='once':
                        yield {'mix_wav':mix_speechs,
                               'mix_feas':mix_feas,
                               'mix_phase':mix_phase,
                               'aim_fea':aim_fea,
                               'aim_spkname':aim_spkname,
                               'query':query,
                               'num_all_spk':len(all_spk),
                               'multi_spk_fea_list':multi_spk_fea_list,
                               'multi_spk_wav_list':multi_spk_wav_list,
                               'multi_video_list':multi_video_list,
                               'multi_video_fea_list':multi_video_fea_list,
                               'batch_total':batch_total,
                               'top_k':mix_k
                               }

                    #下一个batch的混合说话人个数， 先调整一下
                    mix_k=random.sample(mix_number_list,1)[0]
                    batch_idx=0
                    mix_speechs=np.zeros((config.BATCH_SIZE,config.MAX_LEN_SPEECH))
                    mix_feas=[]#应该是bs,n_frames,n_fre这么多
                    mix_phase=[]
                    aim_fea=[]#应该是bs,n_frames,n_fre这么多
                    aim_spkid=[] #np.zeros(config.BATCH_SIZE)
                    aim_spkname=[]
                    query=[]#应该是BATCH_SIZE，shape(query)的形式，用list再转换把
                    multi_spk_fea_list=[]
                    multi_spk_wav_list=[]
                    multi_video_list=[]
                    multi_video_fea_list=[]
                sample_idx[mix_k]+=1

        else:
            raise ValueError('No such dataset:{} for Video.'.format(config.DATASET))
        pass

    #图像刺激
    elif config.MODE==2:
        pass

    #视频刺激
    elif config.MODE==1:
        raise ValueError('No such dataset:{} for Speech'.format(config.DATASET))
    #概念刺激
    elif config.MODE==4:
        pass

    else:
        raise ValueError('No such Model:{}'.format(config.MODE))

# cc=prepare_data('once','train')
# cc.next()
# print cc
# time.sleep(10)