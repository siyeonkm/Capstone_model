'''
Data Loader
'''

import os
import numpy as np
import glob
import pickle as pkl

import torch
from torch.utils import data

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HEVIDataset(data.Dataset):
    def __init__(self, args, phase):
        '''
        HEV-I dataset
        1. Frame_id : 비디오에서 물체의 임시적인 위치
        2. bbox : 물체가 나타났을 때부터 사라질때까지의 bounding box
        3. flow : ROIPool로부터 얻은 optical flow feature들
        4. ego_motion : orbslam2로 부터 얻은 ego car odometry에 대한 [yaw, x, z] 값

        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.args = args
        self.data_root = os.path.join(self.args.data_root, phase)
        self.sessions = glob.glob(os.path.join(self.data_root,'*'))

        self.all_inputs = []
        for session in self.sessions:

            data = pkl.load(open(session, 'rb'))
            # frame_id = data['frame_id']
            bbox = data['bbox']
            flow = data['flow']
            ego_motion = data['ego_motion'] # [yaw, x, z]값


            # Forward
            seed = np.random.randint(self.args.seed_max)
            for start in range(seed, len(flow), int(self.args.segment_len/2)):

                end = start + self.args.segment_len
                if end + self.args.pred_timesteps <= len(bbox) and end <= len(flow):
                    input_bbox = bbox[start:end,:]
                    input_flow = flow[start:end,:,:,:]
                    input_ego_motion = self.get_input(ego_motion, start, end)

                    target_bbox = self.get_target(bbox, start, end)
                    target_ego_motion = self.get_target(ego_motion, start, end)

                    # target_ego_motion = self.get_target(ego_motion_session, ego_start, ego_end)
                    # if input_flow.shape[0] != 16:
                    #     print(flow.shape)
                    #     print(bbox.shape)
                    #     print(input_flow.shape)
                    #     print("start: {} end:{} length:{}".format(start, end, self.args.segment_len))

                    self.all_inputs.append([input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion])

            # Backward
            seed = np.random.randint(self.args.seed_max)
            for end in range(min([len(bbox)-self.args.pred_timesteps, len(flow)]),
                             seed,
                             -self.args.segment_len):

                start = end - self.args.segment_len
                if start >= 0:
                    input_bbox = bbox[start:end,:]
                    input_flow = flow[start:end,:,:,:]
                    input_ego_motion = self.get_input(ego_motion, start, end)
                    target_bbox = self.get_target(bbox, start, end)
                    target_ego_motion = self.get_target(ego_motion, start, end)
                    # if input_flow.shape[0] != 16:
                    #     print(flow.shape)
                    #     print(bbox.shape)
                    #     print(input_flow.shape)
                    #     print("start: {} end:{} length:{}".format(start, end, self.args.segment_len))
                    self.all_inputs.append([input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion])

    def get_input(self, ego_motion_session, start, end):
        if start == 0:
            return np.vstack([ego_motion_session[0:1, :] - ego_motion_session[0:1, :],
                              ego_motion_session[start+1:end] - ego_motion_session[start:end-1]])
        else:
            return ego_motion_session[start:end, :] - ego_motion_session[start-1:end-1, :]

    def get_target(self, session, start, end):
        '''
        input 클립(session, start, end)이 주어졌을때, target 예측(미래에 변화할 값)
        Params:
            session: the input time sequence of a car, can be bbox or ego_motion with shape (time, :)
            start: start frame id
            end: end frame id
        Returns:
            target: Target tensor with shape (self.args.segment_len, pred_timesteps, :)
                    The target is the change of the values. e.g. target of yaw is \delta{\theta}_{t0,tn}
        '''
        target = torch.zeros(self.args.segment_len, self.args.pred_timesteps, session.shape[-1])
        for i, target_start in enumerate(range(start, end)):
            '''the target of time t is the change of bbox/ego motion at times [t+1,...,t+5}'''
            target_start = target_start + 1
            try:
                target[i,:,:] = torch.as_tensor(session[target_start:target_start+self.args.pred_timesteps,:] -
                                            session[target_start-1:target_start,:])
            except:
                print("segment start: ", start)
                print("sample start: ", target_start)
                print("segment end: ", end)
                print(session.shape)
                raise ValueError()
        return target

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, index):
        input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion = self.all_inputs[index]
        input_bbox = torch.FloatTensor(input_bbox).to(device)
        input_flow = torch.FloatTensor(input_flow).to(device)
        input_ego_motion = torch.FloatTensor(input_ego_motion).to(device)

        target_bbox = torch.FloatTensor(target_bbox).to(device)
        target_ego_motion = torch.FloatTensor(target_ego_motion).to(device)

        return input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion


class HEVIEgoDataset(data.Dataset):
    def __init__(self, args, phase):
        '''
        HEVI ego car motion dataset 생성
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.args = args
        self.data_root = os.path.join(self.args.data_root, phase)
        self.sessions = glob.glob(os.path.join(self.data_root,'*'))

        self.all_inputs = []
        for session_file in self.sessions:
            # for each car in dataset, we split to several trainig samples
            ego_motion_session = np.load(session_file) # a (time, 3) numpy array of [yaw, x, z]
            # ego_motion_session = ego_motion_session[:, [0,3,5]] # [yaw, pitch, roll, x, y, z]
            ego_motion_session = torch.FloatTensor(ego_motion_session)
            # go farwad along the session to get data samples
            seed = np.random.randint(self.args.seed_max)
            for start in range(seed, len(ego_motion_session), int(self.args.segment_len/2)): # the interval is half segment length

                end = start + self.args.segment_len
                if end + self.args.pred_timesteps <= len(ego_motion_session):

                    input_ego_motion = self.get_input(ego_motion_session, start, end)

                    target_ego_motion = self.get_target(ego_motion_session, start, end)

                    self.all_inputs.append([input_ego_motion, target_ego_motion])


    def get_input(self, ego_motion_session, start, end):
        if start == 0:
            return torch.cat((ego_motion_session[0:1, :],
                              ego_motion_session[start+1:end] - ego_motion_session[start:end-1]),
                              dim=0)
        else:
            return ego_motion_session[start:end, :] - ego_motion_session[start-1:end-1, :]

    def get_target(self, ego_motion_session, start, end):
        target = torch.zeros(self.args.segment_len, self.args.pred_timesteps, ego_motion_session.shape[-1])
        for i, target_start in enumerate(range(start, end)):
            '''the target of time t is the chaneg of bbox/ego motion at times [t+1,...,t+5}'''
            target_start = target_start + 1
            target[i,:,:] = torch.as_tensor(ego_motion_session[target_start:target_start+self.args.pred_timesteps,:] -
                                            ego_motion_session[target_start-1:target_start,:])
        return target

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, index):
        input_ego_motion, target_ego_motion = self.all_inputs[index]
        input_ego_motion = torch.FloatTensor(input_ego_motion).to(device)

        target_ego_motion = torch.FloatTensor(target_ego_motion).to(device)

        return input_ego_motion, target_ego_motion
