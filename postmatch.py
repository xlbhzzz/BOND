import numpy as np
from os.path import join
from utils import set_params

args = set_params()

def tanimoto(p, q):
    c = [v for v in p if v in q]
    return float(len(c) / (len(p) + len(q) - len(c)))


def generate_pair(pubs, name, outlier, mode):
    dirpath = join(args.save_path, 'relations', mode, name)

    paper_org = {}
    paper_conf = {}
    paper_author = {}
    paper_word = {}

    temp = set()
    with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_org:
                paper_org[p] = []
            paper_org[p].append(a)
    temp.clear()

    with open(dirpath + "/paper_venue.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_conf:
                paper_conf[p] = []
            paper_conf[p] = a
    temp.clear()

    with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_author:
                paper_author[p] = []
            paper_author[p].append(a)
    temp.clear()

    with open(dirpath + "/paper_title.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_word:
                paper_word[p] = []
            paper_word[p].append(a)
    temp.clear()

    paper_paper = np.zeros((len(pubs), len(pubs)))
    for i, pid in enumerate(pubs):
        if i not in outlier:
            continue
        for j, pjd in enumerate(pubs):
            if j == i:
                continue
            ca = 0
            cv = 0
            co = 0
            ct = 0

            if pid in paper_author and pjd in paper_author:
                ca = len(set(paper_author[pid]) & set(paper_author[pjd])) * 1.5
            if pid in paper_conf and pjd in paper_conf and 'null' not in paper_conf[pid]:
                cv = tanimoto(set(paper_conf[pid]), set(paper_conf[pjd])) * 1.0
            if pid in paper_org and pjd in paper_org:
                co = tanimoto(set(paper_org[pid]), set(paper_org[pjd])) * 1.0
            if pid in paper_word and pjd in paper_word:
                ct = len(set(paper_word[pid]) & set(paper_word[pjd])) * 0.33

            paper_paper[i][j] = ca + cv + co + ct
            
    # print("generate pairs done! the shape: ")
    # print(paper_paper.shape)
    return paper_paper


def post_match(pred, pubs, name, mode):

    #1 outlier from dbscan labels
    outlier = set()
    for i in range(len(pred)):
        if pred[i] == -1:
            outlier.add(i)

    #2 outlier from building graphs (relational)
    datapath = join(args.save_path, 'graph', mode, name)
    with open(join(datapath, 'rel_cp.txt'), 'r') as f:
        rel_outlier = [int(x) for x in f.read().split('\n')[:-1]] 

    for i in rel_outlier:
        outlier.add(i)
    
    print(f"post matching {len(outlier)} outliers")
    paper_pair = generate_pair(pubs, name, outlier, mode)
    paper_pair1 = paper_pair.copy()
    
    K = len(set(pred))

    for i in range(len(pred)):
        if i not in outlier:
            continue
        j = np.argmax(paper_pair[i])
        while j in outlier:
            paper_pair[i][j] = -1
            last_j = j
            j = np.argmax(paper_pair[i])
            if j == last_j:
                break

        if paper_pair[i][j] >= 1.5:
            pred[i] = pred[j]
        else:
            pred[i] = K
            K = K + 1

    for ii, i in enumerate(outlier):
        for jj, j in enumerate(outlier):
            if jj <= ii:
                continue
            else:
                if paper_pair1[i][j] >= 1.5:
                    pred[j] = pred[i]
    return pred
