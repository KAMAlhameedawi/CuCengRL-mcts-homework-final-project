# CuCengRL-mcts-homework-final-project
Monte Carlo Tree Search and Deep Neural Network for Neural Architecture Search  
# AlphaX-NASBench101

**Please check [LaNAS/LA-MCTS](https://github.com/facebookresearch/LaMCTS) repository for our latest results**

AlphaX is a new Neural Architecture Search (NAS) agent that uses MCTS for efficient model architecture search with Meta-DNN as a predictive model to estimate the accuracy of a sampled architecture. Compared with Random Search, AlphaX builds an online model that guides the future search. In contrast to greedy methods, e.g., Q-learning, Regularized Evolution, or Top-K methods, AlphaX dynamically trades off exploration and exploitation and can escape from local optimum with fewer numbers of search trials. For details of AlphaX, please refer to [Neural Architecture Search using Deep Neural Networks and Monte Carlo Tree Search](https://arxiv.org/pdf/1805.07440.pdf).

This repository hosts the implementation of AlphaX for searching on a design domain defined by NASBench-101. NASBench-101 is a NAS dataset that contains 420k+ networks with their actual training, validation accuracies. For details of NASBench-101, please check [here](https://github.com/google-research/nasbench).

The comparisons of sample efficiency, MCTS vs. various baselines in NASBench-101, are shown below:  

![nasbench_speed](https://github.com/linnanwang/AlphaX-NASBench101/blob/master/nasbench_speed.png?raw=true)

## Ensure Fair Evaluations

Our encoding mechanism is the same as NASBench, formulating a search space of 500,000,000 architectures. While NASBench only contains 420,000 architecture-accuracy pairs, and we return 0 for those not in the dataset but in the search space. The predictive models such as using a simple MLP can perform well (<= 6000 to find the global optimum on the dataset, see this repo [MLP-NASBench-101](https://github.com/linnanwang/MLP-NASBench-101)) if only trained and predicted using 420,000 architectures in NASBench. However, this result is unfair and invalid as in the real scenarios; it will be impossible to enumerate and predict every architecture in the search space, e.g., NASNet 10^20. In fact, using a pure predictive model to guide the search is proven to be less efficient than SMBO alternatives such as Bayesian Optimizations for lacking a mechanism to explore.

Therefore, please pay attention to these minor details! A good result does not necessarily mean a good algorithm!

## Visualizations

This is how AlphaX progressively probes the search domain. Each node represents an MCTS state; the node color reflects its value, i.e., accuracy, indicating how promising a search branch.

![mcts_viz](https://github.com/linnanwang/AlphaX-NASBench101/blob/master/mcts_viz.png?raw=true)

## Current Caveat

However, this work still needs to explicitly define actions, which are not optimized toward a performance metric. Our recent work has successfully solved this issue and has shown orders of improvement in sample efficiency (LaNAS), [LaNAS Paper](https://linnanwang.github.io/latent-actions.pdf). We will release LaNAS under the repositories of Facebook AI Research soon.

The work has been published in AAAI-20; please cite our work if it helps your research:
```bibtex
@article{wang2019alphax,
  title={Alphax: exploring neural architectures with deep neural networks and monte carlo tree search},
  author={Wang, Linnan and Zhao, Yiyang and Jinnai, Yuu and Tian, Yuandong and Fonseca, Rodrigo},
  journal={arXiv preprint arXiv:1903.11059},
  year={2019}
}
```

## The Best Architecture in the Search, AlphaX-1

97.2 test accuracy on CIFAR-10 (the latest is 98, will update later)

Go to the folder `alphax-1-net`:

```bash
python model_test.py
```

Final top-1 test accuracy is 97.22
Final top-5 test accuracy is 99.95

## Requirements

```bash
Python >= 3.5.5, numpy >= 1.9.1, keras >= 2.1.6, jsonpickle
```

## Setup

1. Clone this repo.

```bash
git clone git@github.com:linnanwang/AlphaX-NASBench101.git
cd AlphaX-NASBench101
```

After 78 batches, you will get:
Final top-1 test accuracy is 97.22
Final top-5 test accuracy is 99.95

2. (optional) Create a virtualenv for this library.

```bash
virtualenv --system-site-packages -p python3 ./venv
source venv/bin/activate
```

3. Install the project along with dependencies.

```bash
pip install numpy
pip install keras
pip install jsonpickle
```

## Download the Dataset

The full NASBench dataset in our format is available [here](https://drive.google.com/file/d/100xB4Mj7Hc5I0ljVPo7ATmC2kfhytHuN/view?usp=sharing).
