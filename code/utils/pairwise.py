import torch
import numpy as np

def mat2label(adj):
    adj_preds = []
    for i in adj:
        if isinstance(i, np.ndarray):
            temp = i
        else:
            temp = i.cpu().detach().numpy()
        for idx, j in enumerate(temp):
            if j == 1: 
                adj_preds.append(idx)
                break
            if idx == len(temp)-1:
                adj_preds.append(-1)

    return adj_preds
    
def onehot_encoder(label_list):
    if isinstance(label_list, np.ndarray):
        labels_arr = label_list
    else:
        try:
            labels_arr = np.array(label_list.cpu().detach().numpy())
        except:
            labels_arr = np.array(label_list)
    
    num_classes = max(labels_arr) + 1
    max_val = labels_arr.max()
    
    onehot_mat = np.zeros((len(labels_arr), num_classes+1))

    for i in range(len(labels_arr)):
        onehot_mat[i, labels_arr[i]] = 1

    return onehot_mat


def pw_label(label_list):
    # change to one-hot form
    class_matrix = torch.from_numpy(onehot_encoder(label_list))
    # get N * N matrix
    pw_matrix = torch.mm(class_matrix, class_matrix.t())

    return pw_matrix
