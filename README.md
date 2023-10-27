# BOND
This repository contains the source code of paper "BOND: Bootstrapping From-Scratch Name Disambiguation with Multi-task Promoting".

![Overview](/Bond.png)

## Datasets
The datasets WhoIsWho can be downloaded from: https://www.aminer.cn/whoiswho

Paper embedding can be downloaded from: https://pan.baidu.com/s/1A5XA9SCxvENM2kKPUv6X4Q?pwd=c9kk 
Password: c9kk

## Requirements
    gensim==4.3.0
    matplotlib==3.7.1
    numpy==1.24.3
    pandas==1.5.3
    pinyin==0.4.0
    scikit-learn==1.2.2
    scipy==1.10.1
    torch==1.12.1
    torch-geometric==2.2.0
    tqdm==4.65.0

## Usage

**BOND**

Download dataset "src" and embedding "paper_emb", organize as follows:


    .
    ├── code
    ├── data
        ├── paper_emb
        │   ├── test
        │   ├── train
        │   └── valid
        └── src
            ├── sna-test
            ├── sna-valid
            └── train


Execute the following command:

    python code/preprocess.py
    python code/main.py


**BOND+**

To be continued.


