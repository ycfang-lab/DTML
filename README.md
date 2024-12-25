# DTML

## Dataset

The following datasets are used in this project for Oracle Bone Character (OBC) recognition:

| Dataset Name       | Dataset Link                                                                 | Paper Link                                                                                          |
|--------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| **Oracle-241**     | [Dataset](https://github.com/wm-bupt/STSN)                                  | [Paper](https://ieeexplore.ieee.org/abstract/document/9757826)                                     |
| **OBI125**         | [Dataset](http://www.ihpc.se.ritsumei.ac.jp/obidataset.html)                | [Paper](https://dl.acm.org/doi/abs/10.1145/3532868)                                                |
| **Oracle-50K**     | [Dataset](https://github.com/wenhui-han/Oracle-50K)                | [Paper](https://openaccess.thecvf.com/content/ACCV2020/html/Han_Self-supervised_Learning_of_Orc-Bert_Augmentator_for_Recognizing_Few-Shot_Oracle_Characters_ACCV_2020_paper.html)                            |

---

## Preprocessing

To prepare the datasets **OBI125** and **Oracle50K** for training, we preprocess these datasets to extract their common character subsets. You can perform this preprocessing by running the following script:

```bash
python preprocessing/Oracle50k_OBI125.py
```

---

## Training and Validation

### Training
To train the DTML model, use the following command:

```bash
python main.py --cuda 0 --dataset Oracle241 --source h --target s --model DTML --preloading
```

### Training
To evaluate the DTML model, use the following command:

```bash
python main.py --weight /checkpoint --cuda 0 --dataset Oracle241 --source h --target s --batch_size 256 --preloading --evaluate
```

---

## Citation

Waiting for updates
