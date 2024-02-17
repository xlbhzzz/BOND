import torch
import random
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch_geometric.nn import GAE
from module.att_gnn import ATT_GNN
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

from utils import set_params
from utils import pw_label
from utils import post_match
from utils import pairwise_evaluate
from utils import save_results
from utils import load_dataset
from utils import load_graph
from utils import mat2label


args = set_params()

seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device(("cuda:"+str(args.gpu)) if torch.cuda.is_available() and args.cuda else "cpu")


def train(name, pubs, results):
    clus_label_box = []
    clas_label_box = []
    for threshold_a in args.th_a:
        for threshold_o in args.th_o:
            for threshold_v in args.th_v:
                all_embs = []
                all_logits = []
                print(f"====th_a-> {threshold_a}, \
                th_o->{threshold_o}, \
                th_v->{threshold_v}====")
                for it in range(args.repeat_num):
                    if args.repeat_num > 1:
                        print(f"repeat: {it+1}/{args.repeat_num}")
                    # Load data
                    label, ft_list, data = load_graph(name, 
                                                    th_a=threshold_a, 
                                                    th_o=threshold_o,
                                                    th_v=threshold_v)
                    num_cluster = int(ft_list.shape[0]*args.compress_ratio)
                    layer_shape = []
                    input_layer_shape = ft_list.shape[1]
                    hidden_layer_shape = args.hidden_dim
                    output_layer_shape = num_cluster #adjust output-layer size of FC layer.
                    
                    layer_shape.append(input_layer_shape)
                    layer_shape.extend(hidden_layer_shape)
                    layer_shape.append(output_layer_shape)

                    # Init model
                    model = GAE(ATT_GNN(layer_shape))
                    ft_list = ft_list.float()
                    ft_list = ft_list.to(device)
                    data = data.to(device)
                    model.to(device)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
                    
                    name_pubs = []
                    if args.mode == 'train':
                        for aid in pubs[name]:
                            name_pubs.extend(pubs[name][aid])
                    else:
                        for pid in pubs[name]:
                            name_pubs.append(pid)

                    for epoch in range(args.epochs):
                        # Train
                        model.train()
                        optimizer.zero_grad()
                        logits, embd = model.encode(ft_list, data.edge_index, data.edge_attr)
                        dis = pairwise_distances(embd.cpu().detach().numpy(), metric='cosine')
                        db_label = DBSCAN(eps=args.db_eps, min_samples=args.db_min, metric='precomputed').fit_predict(dis) 
                        db_label = torch.from_numpy(db_label)
                        db_label = db_label.to(device) 
                        
                        lc_label = pw_label(db_label)
                        lc_label = lc_label.float()
                        lc_label = lc_label.to(device)

                        gl_label = torch.matmul(logits, logits.t())
                        
                        loss_cluster = F.binary_cross_entropy_with_logits(gl_label, lc_label)
                        loss_recon = model.recon_loss(embd, data.edge_index)
                        w_cluster = args.cluster_w
                        w_recon = 1 - w_cluster
                        loss_train = w_cluster * loss_cluster + w_recon * loss_recon
                        
                        loss_train.backward()
                        optimizer.step()


                    # Evaluate
                    with torch.no_grad():
                        model.eval()
                        logits, embd = model.encode(ft_list, data.edge_index, data.edge_attr)
                    all_embs.append(embd.cpu().detach().numpy())
                    all_logits.append(logits.cpu().detach().numpy())
    
                #==========================bagging============================
                all_embs = np.array(all_embs)
                all_dis = np.zeros((len(name_pubs), len(name_pubs)))

                for k in range(args.repeat_num):
                    all_dis = all_dis + pairwise_distances(all_embs[k], metric="cosine")
                all_dis = all_dis / args.repeat_num
                all_logits = np.average(all_logits)
                
                db_label = DBSCAN(eps=args.db_eps, min_samples=args.db_min, metric='precomputed').fit_predict(all_dis)
                db_label = torch.from_numpy(db_label) 
                db_label = db_label.to(device)  
                
                clus_labels = db_label.detach().cpu().numpy()
                if args.post_match == True:
                    clus_labels = post_match(clus_labels, name_pubs, name, args.mode)

                clus_label_box.append(clus_labels)


    #===========================ensemble===========================
    clus_mat_box = np.zeros([len(name_pubs), len(name_pubs)])

    for cur_label in clus_label_box:
        cur_mat = pw_label(cur_label)
        # print(cur_mat)
        clus_mat_box = np.add(clus_mat_box,cur_mat)


    clus_mat_box /= len(clus_label_box)
    clus_ensb_out = np.zeros([len(name_pubs), len(name_pubs)])

    #voting
    for idx, i in enumerate(clus_mat_box):
        for jdx, j in enumerate(i):
            if j > 0.5:
                clus_ensb_out[idx][jdx] = 1

    pred = mat2label(clus_ensb_out)

    #===========================saving results=======================
    results[name] = pred 
    #===========================ensambel-eva=========================
    if args.mode == "train":
        true_label = label.detach().cpu().numpy()     
        prec, rec, f1 = pairwise_evaluate(true_label, pred)
        pred_label_num = len(set(pred))
        true_label_num = len(set(true_label))

        print(f"name:{name}, clus:{f1}")

        return prec, rec, f1, results
    else:
        return results

    
if __name__ == '__main__':
    names, pubs = load_dataset(args.mode)
    results = {}

    f1_list = []
    for name in names:
        print("trainning:", name)
        results[name] = []
        if args.mode == "train":
            prec, rec, f1, results = train(name, pubs, results)
            f1_list.append(f1)
        else:
            results = train(name, pubs, results)
        
    result_path = save_results(names, pubs, results)
    if args.mode == "train":
        print("Done! Average F1: {:.4f}".format(np.average(f1_list)))
    else:
        print("Done! Results saved:", result_path)
