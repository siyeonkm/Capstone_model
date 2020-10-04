import sys
import numpy as np
import copy
import torch
from torch import nn, optim
from torch.nn import functional as F
# from torch.autograd import Variable
import pickle as pkl

# device 설정(gpu 없을 시 cpu로 돌아가도록)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# GRU기반 인코더 디코더 먼저 정의해놓음
# 1. 인코더
class EncoderGRU(nn.Module):
    def __init__(self, args):
        super(EncoderGRU, self).__init__()
        self.args = args
        self.enc = nn.GRUCell(input_size=self.args.input_embed_size,
                    hidden_size=self.args.enc_hidden_size)

    def forward(self, embedded_input, h_init):
        '''
        인코딩 과정
        Params:
            x: input feature, (batch_size, time, feature dims)
            h_init: initial hidden state, (batch_size, enc_hidden_size)
        Returns:
            h: updated hidden state of the next time step, (batch.size, enc_hiddden_size)
        '''
        h = self.enc(embedded_input, h_init)
        return h

# 2. 디코더
class DecoderGRU(nn.Module):
    def __init__(self, args):
        super(DecoderGRU, self).__init__()
        self.args = args
        # PREDICTOR INPUT FC
        self.hidden_to_pred_input = nn.Sequential(nn.Linear(self.args.dec_hidden_size,
                                                            self.args.predictor_input_size),
                                                  nn.ReLU())

        # PREDICTOR DECODER
        self.dec = nn.GRUCell(input_size=self.args.predictor_input_size,
                                        hidden_size=self.args.dec_hidden_size)

        # PREDICTOR OUTPUT
        if self.args.non_linear_output:
            self.hidden_to_pred = nn.Sequential(nn.Linear(self.args.dec_hidden_size,
                                                            self.args.pred_dim),
                                                nn.Tanh())
        else:
            self.hidden_to_pred = nn.Linear(self.args.dec_hidden_size,
                                                            self.args.pred_dim)

    def forward(self, h, embedded_ego_pred=None):
        '''
        미래 관측치 예측에 대한 RNN 예측 모델
        Params:
            h: hidden state tensor from the encoder, (batch_size, enc_hidden_size)
            embedded_ego_pred: (batch_size, pred_timesteps, input_embed_size)
        '''
        output = torch.zeros(h.shape[0], self.args.pred_timesteps, self.args.pred_dim).to(device)

        all_pred_h = torch.zeros([h.shape[0], self.args.pred_timesteps, self.args.dec_hidden_size]).to(device)
        all_pred_inputs = torch.zeros([h.shape[0], self.args.pred_timesteps, self.args.predictor_input_size]).to(device)

        # predict input 값 0으로 초기화
        pred_inputs = torch.zeros(h.shape[0], self.args.predictor_input_size).to(device) #self.hidden_to_pred_input(h)
        for i in range(self.args.pred_timesteps):
            if self.args.with_ego:
                pred_inputs = (embedded_ego_pred[:, i, :] + pred_inputs)/2  # 미래의 ego motion이랑 prediction의 평균
            all_pred_inputs[:, i, :] = pred_inputs
            h = self.dec(pred_inputs, h)

            pred_inputs = self.hidden_to_pred_input(h)

            all_pred_h[:,i,:] = h

            output[:,i,:] = self.hidden_to_pred(h)

        return output, all_pred_h, all_pred_inputs


# FOL : Future Object Localization
class FolRNNED(nn.Module):
    def __init__(self, args):
        super(FolRNNED, self).__init__()

        # get args and process
        self.args = copy.deepcopy(args)

        # 각 객체의 위치 정보를 추출하는 location Encoder
        # 1) 객체의 현재 bounding box에 대한 정보인 X_t를 입력받음
        # 2) 시간 정보와 공간 정보를 합친 Spatiotemporal feature를 입력받음
        # -> Spatiotemporal : Optical flow를 RolPool 과정을 통해 추출한 벡터로, 이미지 내의 움직임에 대한 추가정보 포함
        # 이 둘을 기반으로 특징을 추출함
        self.box_enc_args = copy.deepcopy(args) # 1)
        self.flow_enc_args = copy.deepcopy(args) # 2)

        # 1),2)를 concat
        if self.args.enc_concat_type == 'cat':
            self.args.dec_hidden_size = self.args.box_enc_size + self.args.flow_enc_size
        else:
            if self.args.box_enc_size != self.args.flow_enc_size:
                raise ValueError('Box encoder size %d != flow encoder size %d'
                                    %(self.args.box_enc_size,self.args.flow_enc_size))
            else:
                self.args.dec_hidden_size = self.args.box_enc_size

        self.box_enc_args.enc_hidden_size = self.args.box_enc_size
        self.flow_enc_args.enc_hidden_size = self.args.flow_enc_size

        # 인코더 디코더 초기화.
        self.box_encoder = EncoderGRU(self.box_enc_args)
        self.flow_encoder = EncoderGRU(self.flow_enc_args)
        self.args.non_linear_output = True
        self.predictor = DecoderGRU(self.args)

        # 다른 layer들 초기화.
        self.box_embed = nn.Sequential(nn.Linear(4, self.args.input_embed_size), # size of box input is 4
                                        nn.ReLU())
        self.flow_embed = nn.Sequential(nn.Linear(50, self.args.input_embed_size), # size of flow input is 50=5*5*2
                                        nn.ReLU())
        self.ego_pred_embed = nn.Sequential(nn.Linear(3, self.args.input_embed_size), # size of ego input is 3
                                        nn.ReLU())

        # Pretrained 모델을 사용시
        # if args.with_ego:
        #     # initialize ego motion predictor and load the pretrained model
        #     print("Initializing pre-trained ego motion predictor!")
        #     self.ego_predictor = EgoRNNED(self.args)
        #     self.ego_predictor.load_state_dict(torch.load(self.args.best_ego_pred_model))
        #     print("Pre-trained ego_motion predictor done!")

    def forward(self, box, flow, ego_pred):
        '''
        RNN 기반 인코더 디코더 모델
        Params:
            box: (batch_size, segment_len, 4)
            flow: (batch_size, segment_len, 5, 5, 2)
            ego_pred: (batch_size, segment_len, pred_timesteps, 3) or None

            for training and validation, segment_len is large, e.g. 10
            for online testing, segment_len=1
        return:
            fol_predictions: predicted with shape (batch_size, segment_len, pred_timesteps, pred_dim)
        '''
        self.args.batch_size = box.shape[0]
        if len(flow.shape) > 3:
            flow = flow.view(self.args.batch_size, self.args.segment_len, -1)
        embedded_box_input= self.box_embed(box)
        embedded_flow_input= self.flow_embed(flow)

        embedded_ego_input = self.ego_pred_embed(ego_pred) # (batch_size, segment_len, pred_timesteps, input_embed_size)

        # initialize hidden states as zeros
        box_h = torch.zeros(self.args.batch_size, self.args.box_enc_size).to(device)
        flow_h = torch.zeros(self.args.batch_size, self.args.flow_enc_size).to(device)

        # a zero tensor used to save fol prediction
        fol_predictions = torch.zeros(self.args.batch_size,
                                    self.args.segment_len,
                                    self.args.pred_timesteps,
                                    self.args.pred_dim).to(device)

        # 모델을 반복적으로 실행시켜서 각 시간에 대한 미래의 프레임 T 예측
        for i in range(self.args.segment_len):
            # 인코더
            box_h = self.box_encoder(embedded_box_input[:,i,:], box_h)
            flow_h = self.flow_encoder(embedded_flow_input[:,i,:], flow_h)

            # Concat
            if self.args.enc_concat_type == 'cat':
                hidden_state = torch.cat((box_h, flow_h), dims=1)
            elif self.args.enc_concat_type in ['sum', 'avg', 'average']:
                hidden_state = (box_h + flow_h) / 2
            else:
                raise NameError(self.args.enc_concat_type, ' is unknown!!')

            # 디코더
            if self.args.with_ego:
                output, _, _ = self.predictor(hidden_state, embedded_ego_input[:,i,:,:])
            else:
                # 미래의 ego motion 없이 예측
                output, _, _ = self.predictor(hidden_state, None)

            fol_predictions[:,i,:,:] = output
        return fol_predictions

    def predict(self, box, flow, box_h, flow_h, ego_pred):
        '''
        미래의 bbox 예측하기 위해 forward inference run
        Params:
            box: (1, 4)
            flow: (1, 1, 5, 5, 2)
            ego_pred: (1, pred_timesteps, 3)
        return:
            box_changes:()
            box_h,
            flow_h
        '''
        # self.args.batch_size = box.shape[0]
        if len(flow.shape) > 3:
            flow = flow.view(1, -1)
        embedded_box_input= self.box_embed(box)
        embedded_flow_input= self.flow_embed(flow)
        embedded_ego_input = None
        if self.args.with_ego:
            embedded_ego_input = self.ego_pred_embed(ego_pred)

        # 모델 반복적으로 돌려서 각 시간마다 5개의 future frame 예측
        box_h = self.box_encoder(embedded_box_input, box_h)
        flow_h = self.flow_encoder(embedded_flow_input, flow_h)

        if self.args.enc_concat_type == 'cat':
            hidden_state = torch.cat((box_h, flow_h), dims=1)
        elif self.args.enc_concat_type in ['sum', 'avg', 'average']:
            hidden_state = (box_h + flow_h) / 2
        else:
            raise NameError(self.args.enc_concat_type, ' is unknown!!')

        box_changes, _, _ = self.predictor(hidden_state, embedded_ego_input)


        return box_changes, box_h, flow_h

# Ego RNN
class EgoRNNED(nn.Module):
    def __init__(self, args):
        super(EgoRNNED, self).__init__()

        self.args = copy.deepcopy(args)

        # 모델 돌리기 위한 argumnets 설정
        self.args.input_embed_size = self.args.ego_embed_size
        self.args.enc_hidden_size = self.args.ego_enc_size
        self.args.dec_hidden_size = self.args.ego_dec_size
        self.args.pred_dim = self.args.ego_dim
        self.args.predictor_input_size = self.args.ego_pred_input_size
        self.args.with_ego = False

        # 인코더 초기화
        self.ego_encoder = EncoderGRU(self.args)

        # 다른 layer들 초기화
        self.ego_embed = nn.Sequential(nn.Linear(3, self.args.ego_embed_size), # size of box input is 4
                                        nn.ReLU())#nn.LeakyReLU(0.1)

        self.args.non_linear_output = False # Future Ego motion 에측할 때는 non_linear_output을 사용하지 않음.
        self.predictor = DecoderGRU(self.args)


    def forward(self, ego_x, image=None):
        '''
        RNN 기반 인코더 디코더 모델
        Params:
            ego_x: (batch_size, segment_len, ego_dim)
            image: (batch_size, segment_len, feature_dim) e.g. feature_dim = 1024

            for training and validation, segment_len is large, e.g. 10
            for online testing, segment_len=1
        return:
            predictions: predicted ego motion with shape (batch_size, segment_len, pred_timesteps, ego_dim)
        '''
        self.args.batch_size = ego_x.shape[0]

        # 임베딩
        embedded_ego_input= self.ego_embed(ego_x)

        # initialize hidden states as zeros
        ego_h = torch.zeros(self.args.batch_size, self.args.enc_hidden_size).to(device)

        # a zero tensor used to save outputs
        predictions = torch.zeros(self.args.batch_size,
                                    self.args.segment_len,
                                    self.args.pred_timesteps,
                                    self.args.pred_dim).to(device)

        # run model iteratively, predict 5 future frames at each time
        for i in range(self.args.segment_len):
            ego_h = self.ego_encoder(embedded_ego_input[:,i,:], ego_h)
            output, _, _ = self.predictor(ego_h)
            predictions[:,i,:,:] = output
            # break
        return predictions

    def predict(self, ego_x, ego_h, image=None):
        '''
        Params:
            ego_x: (1, 3)
            ego_h: (1, 64)
            #image: (1, 1, 1024) e.g. feature_dim = 1024
        returns:
            ego_changes: (pred_timesteps, 3)
            ego_h: (1, ego_enc_size)
        '''
        # 임베딩
        embedded_ego_input= self.ego_embed(ego_x)

        ego_h = self.ego_encoder(embedded_ego_input, ego_h)
        ego_changes, _, _ = self.predictor(ego_h)

        return ego_changes, ego_h
