# ActiSiamese
We have witnessed in recent years an ever-growing volume of information becoming available in a streaming manner in various application areas. As a result, there is an emerging need for online learning methods that train predictive models on-the-fly. A series of open challenges, however, hinder their deployment in practice. These are, learning as data arrive in real-time one-by-one, learning from data with limited ground truth information, learning from nonstationary data, and learning from severely imbalanced data, while occupying a limited amount of memory for data storage. We propose the ActiSiamese algorithm, which addresses these challenges by combining online active learning, siamese networks, and a multi-queue memory. It develops a new density-based active learning strategy which considers similarity in the latent (rather than the input) space. We conduct an extensive study that com- pares the role of different active learning budgets and strategies, the performance with/without memory, the performance with/without ensembling, in both synthetic and real-world datasets, under different data nonstationarity characteristics and class imbalance levels. ActiSiamese outperforms baseline and state-of-the-art algorithms, and is effective under severe imbalance, even only when a fraction of the arriving instances’ labels is available. We publicly release our code to the community. 

# Paper
You can get a free copy of the pre-print version from Zenodo [(link)](https://zenodo.org/record/7135177#.Yzq6KexBxTZ) or arXiv [(link)](https://arxiv.org/abs/2210.01090).

Alternatively, you can get the published version from the publisher’s website (behind a paywall, [link](https://www.sciencedirect.com/science/article/abs/pii/S0925231222011481)).

# Citation request
If you have found our paper and / or part of our code useful, please cite our work as follows:

- K. Malialis, C. G. Panayiotou, M. M. Polycarpou, Nonstationary data stream classification with online active learning and siamese neural networks, Neurocomputing, Volume 512, Pages 235-252, 2022, doi: 10.1016/j.neucom.2022.09.065.

# Instructions
Python 3.7. Please also check the “instructions.txt” file.

# Requirements
Please check the “requirements.txt” file for the necessary libraries and packages.
