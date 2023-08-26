# PEdger
Yuanbin Fu, and Xiaojie Guo. *Practical Edge Detection via Robust Collaborative Learning*.  In ACM Multimedia, 2023

# Preparing Data
Download the augmented BSDS and PASCAL VOC datasets from:

http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz

http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz

Download the augmented NYUD dataset from:

http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz

# Pretrained Model
``checkpoint.pth” in this repository is our pre-trained model.

# Training
change the data path in ``main.py" to your own path, then run:

`python main.py`

# Testing
change the data path and checkpoint path in ``test.py" to your own path, then run:

`python test.py`
