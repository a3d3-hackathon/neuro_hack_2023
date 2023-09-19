import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def reverse_mask(pred_stack, target_stack):
    '''
    pred_stack, target_stack: numpy array, dim=2
    Remove elements where target==0
    '''
    if pred_stack.ndim != 2 or target_stack.ndim!=2:
        print('dim!=2')
        exit()

    idx = np.any(target_stack, axis=1)
    target_select = target_stack[idx, :]
    pred_select = pred_stack[idx, :]

    return pred_select, target_select

def collate_fn_custom(batch):
    data, target = [], []
    len_trg_list = []
    neural_idx = []
    coord_idx = [] 
    
    for item in batch:
        data.append(item[0])

        # check zero in target, to prevent being removed from reverse_mask
        trg = item[1]
        if 0 in trg:
            trg[(trg == 0.0).nonzero(as_tuple=True)] = 1e-6

        target.append(trg)
        len_trg_list.append(len(trg))

        neural_idx.append(item[2])
        coord_idx.append(item[3])

    data = torch.stack(data)
    if len(set(len_trg_list)) == 1:
        target = torch.stack(target)
    else:
        target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True)

        # # sequence heatmap
        # for b in range(target.shape[0]):
        #     trg = np.transpose(target.detach().cpu().numpy()[b])
        #     fig, ax = plt.subplots()
        #     plt.imshow(trg)
        #     plt.xlabel('sequence')
        #     plt.ylabel('limb-8')
        #     for i in range(trg.shape[0]):
        #         for j in range(trg.shape[1]):
        #             text = ax.text(j, i, f"{trg[i, j]:.4f}", ha="center", va="center", color="w")
        #     plt.title('target, Batch=' + str(b) + '/' + str(target.shape[0]))
        #     plt.show()

    return data, target, neural_idx, coord_idx

class PDatasetA3D3(Dataset):
    def __init__(self, dir_data, name_neural, name_coord, sequence_length, output_idx, kind, idx_list):
        '''
        dir_data: directory of data (neural data, coordinate data)
        name_neural: file name of neural data. ex) spks_z_sel
        name_coord: file name of coordinates. ex) behav_coord_likeli_norm
        sequence_length: length of the neural sequence to output. ex) 5
        output_idx = [None], [0], [0,1], ...
        kind = 'seq2seq_overlapO'-for training, 'seq2seq_overlapX'-for test. *If seq2seq, sequence lengh of x and y time are different. 
                overlapO: allows sequences to be overlapped. overlapX: sequences are not overlapped.
        idx_list = [start, end(include)] index of sequence of neural_data (neural_data 2nd dim), or [[start1, end1(include)], [start2, end2(include)], ...]. For train and test data. 
        '''
        self.kind = kind 
        self.idx_neural_batch_list = []
        self.idx_behav_batch_list = []

        # Check wrong kind
        if self.kind not in ['seq2seq_overlapO', 'seq2seq_overlapX']:
            print('Weird kind in dataset')
            exit()
        
        # Check wrong idx_list
        if isinstance(idx_list[0], list) is False:
            if idx_list[1] < idx_list[0]:
                print('idx_list is werid. ', idx_list[0], idx_list[1])
                exit()
        else:
            for j in range(len(idx_list)):
                if idx_list[j][1]<idx_list[j][0]:
                    print('idx_list[', j, '] is werid. ', idx_list[j])
                    exit()
            
        # Load data
        path_data_neural = os.path.join(dir_data, name_neural+'.npy')
        path_data_coord = os.path.join(dir_data, name_coord+'.npy')

        neural_data_load = np.load(path_data_neural) #(neuron, time)
        neural_data_load = np.transpose(neural_data_load) #(time, neuron) 
        self.neural_data_tensor = torch.tensor(neural_data_load, dtype=torch.float32)
        #print('neural_data_tensor: ', self.neural_data_tensor.shape)

        behav_coord_load = np.load(path_data_coord) #(behav, time)
        behav_coord_load = np.transpose(behav_coord_load) #(time, behav)
        if output_idx[0] is not None: behav_coord_load = behav_coord_load[:, output_idx] # Select output_idx
        self.behav_coord_tensor = torch.tensor(behav_coord_load, dtype=torch.float32)
        #print('behav_coord_tensor: ', self.behav_coord_tensor.shape)

        if 'seq2seq' in kind: # If seq2seq, load idx_coord_neural
            path_idx = os.path.join(dir_data, 'idx_coord_neural.npy')
            idx_coord_neural = np.load(path_idx).astype('int64')
            #print('idx_coord_neural: ', idx_coord_neural.shape, idx_coord_neural[:30], idx_coord_neural[-30:])

            if idx_coord_neural.shape[0] != behav_coord_load.shape[0]:
                print("idx_coord_neural.shape[0] != behav_coord.shape[0]", idx_coord_neural.shape, behav_coord_load.shape)
                exit()
        
        # Get idx_neural_batch_list and idx_behav_batch_list from idx_list 
        if self.kind == 'seq2seq_overlapO':
            if isinstance(idx_list[0], list) is False:
                for i in range(idx_list[0], idx_list[1]-sequence_length+2):
                    idx_neural_list = [i+s for s in range(sequence_length)]
                    self.idx_neural_batch_list.append(idx_neural_list)

                    idx_coord_list = []
                    for i in idx_neural_list:
                        idx_found = np.where(idx_coord_neural==i)[0]
                        idx_coord_list.extend(idx_found)
                    if len(idx_coord_list) != len(set(idx_coord_list)):
                        print('idx_coord_list has duplicates in dataset.')
                        exit()
                    self.idx_behav_batch_list.append(idx_coord_list)
            else:
                for idx_list_inside in idx_list:
                    for i in range(idx_list_inside[0], idx_list_inside[1]-sequence_length+2):
                        idx_neural_list = [i+s for s in range(sequence_length)]
                        self.idx_neural_batch_list.append(idx_neural_list)
                        
                        idx_coord_list = []
                        for i in idx_neural_list:
                            idx_found = np.where(idx_coord_neural==i)[0]
                            idx_coord_list.extend(idx_found)
                        if len(idx_coord_list) != len(set(idx_coord_list)):
                            print('idx_coord_list has duplicates in dataset.')
                            exit()
                        self.idx_behav_batch_list.append(idx_coord_list)

        elif self.kind == 'seq2seq_overlapX':
            if isinstance(idx_list[0], list) is False:
                for i in range((idx_list[1]-idx_list[0]+1)//sequence_length):
                    idx_neural_list = [idx_list[0]+i*sequence_length+s for s in range(sequence_length)]
                    self.idx_neural_batch_list.append(idx_neural_list)

                    idx_coord_list = []
                    for i in idx_neural_list:
                        idx_found = np.where(idx_coord_neural==i)[0]
                        idx_coord_list.extend(idx_found)
                    if len(idx_coord_list) != len(set(idx_coord_list)):
                        print('idx_coord_list has duplicates in dataset.')
                        exit()
                    self.idx_behav_batch_list.append(idx_coord_list)
            else:
                for idx_list_inside in idx_list:
                    for i in range((idx_list_inside[1]-idx_list_inside[0]+1)//sequence_length):
                        idx_neural_list = [idx_list_inside[0]+i*sequence_length+s for s in range(sequence_length)]
                        self.idx_neural_batch_list.append(idx_neural_list)

                        idx_coord_list = []
                        for i in idx_neural_list:
                            idx_found = np.where(idx_coord_neural==i)[0]
                            idx_coord_list.extend(idx_found)
                        if len(idx_coord_list) != len(set(idx_coord_list)):
                            print('idx_coord_list has duplicates in dataset.')
                            exit()
                        self.idx_behav_batch_list.append(idx_coord_list)

    def __len__(self):
        if self.kind in ['seq2seq_overlapO', 'seq2seq_overlapX']:
            return len(self.idx_neural_batch_list)
      
    def __getitem__(self, idx):
        if self.kind in ['seq2seq_overlapO', 'seq2seq_overlapX']:
            # Get neural_idx_list
            neural_idx_list = self.idx_neural_batch_list[idx]
           
            # Get coord_idx_list
            coord_idx_list = self.idx_behav_batch_list[idx]
            
            # Get data
            return self.neural_data_tensor[neural_idx_list, :], self.behav_coord_tensor[coord_idx_list, :], neural_idx_list, coord_idx_list

if __name__ == '__main__':
    dir_data = r"E:\Data_Working\A3D3_Hackathon_230919\Animal1-G8_53950_1L_Redo"
    name_neural = 'spks_z_sel'
    name_coord = 'behav_coord_likeli_norm'
    seq_len = 5
    output_idx = [None] #[None], [0,1], ... 
    dataset_name = "seq2seq_overlapO" #seq2seq_overlapO-for train, seq2seq_overlapX-for test 
    idx_list = [0, 30]
    batch_size= 16

    dataset = PDatasetA3D3(dir_data, name_neural, name_coord, seq_len, output_idx, dataset_name, idx_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn_custom)

    gt_stack, pred_stack = None, None
    for idx_batch, data in enumerate(dataloader):
        # Load data
        inputs, labels, neural_idx, coord_idx = data
        print(idx_batch)
        print('inputs: ', inputs.shape) #(batch, seq_len, num_neurons)
        print('labels: ', labels.shape) #(batch, seq_len_coord, num_limb=8)
        print('neural_idx: ', len(neural_idx)) #batch
        print('coord_idx:', len(coord_idx)) #batch

        # Get prediction. Here, putting random numbers
        pred = np.random.random_sample(labels.shape)
        print('pred: ', pred.shape)

        # Stack gt
        labels_reshape = np.reshape(labels.data.cpu().numpy(), (labels.shape[0]*labels.shape[1], labels.shape[2]))
        if gt_stack is None: gt_stack = labels_reshape
        else: gt_stack = np.concatenate((gt_stack, labels_reshape))

        # Stack pred
        pred_reshape = np.reshape(pred, (pred.shape[0]*pred.shape[1], pred.shape[2])) #pred.data.cpu().numpy()
        if pred_stack is None: pred_stack = pred_reshape
        else: pred_stack = np.concatenate((pred_stack, pred_reshape))
    print('gt_stack: ', gt_stack.shape)
    print('pred_stack: ', pred_stack.shape)

    if 'seq2seq' in dataset_name:
        pred_stack, gt_stack = reverse_mask(pred_stack, gt_stack)
    print('After reverse mask')
    print('gt_stack: ', gt_stack.shape)
    print('pred_stack: ', pred_stack.shape)