# On the Power of Small-size Graph Neural Networks for Linear Programming (GD-Net) #

## This is the git repository for GD-Net, a neural network designed to handle end-to-end solution mapping tasks for covering/packing LPs. This work has been accepted by NeurIPS 2024 ##

## Introduction ##
Graph neural networks (GNNs) have recently emerged as powerful tools for addressing complex optimization problems. It has been theoretically demonstrated that GNNs can universally approximate the solution mapping functions of linear programming (LP) problems. However, these theoretical results typically require GNNs to have large parameter sizes. Conversely, empirical experiments have shown that relatively small GNNs can solve LPs effectively, revealing a significant discrepancy between theoretical predictions and practical observations. In this work, we aim to bridge this gap by providing a theoretical foundation for the effectiveness of small-size GNNs. We prove that polylogarithmic-depth, constant-width GNNs are sufficient to solve packing and covering LPs, two widely used classes of LPs. Our proof leverages the capability of GNNs to simulate a variant of the gradient descent algorithm on a carefully selected potential function. Additionally, we introduce a new GNN architecture, termed GD-Net. Experimental results demonstrate that GD-Net significantly outperforms conventional GNN structures while using fewer parameters.

### Basical manual: ###
- Use gen*.py to generate instances

- Use train*.py to train your models

- Use test*.py to test the obtained model

---

### Detailed file explanation ###

1. ./ecole
   > folder contains code to generate instances that require Ecole package
  
2. ./gcns
  > This folder is not included in this repository as you can directly access it from https://github.com/chendiqian/IPM_MPNN, which is the compared GNNs in the article.

3. ./gen_cover.py
4. ./gen_lp.py
6. ./gen_maxflow.py
  > code to generate covering/packing/maxflow instances.

7. ./helper.py
8. convert_to_mps.py
  > helper functions that allows a smoother operation

9. ./model.py
  > including all of our tested models (TODO::clean code)

10. ./ori_alg.py
  > Use the original algorithm to solve the optimization problem, could use warm starts.

11. ./test_covering.py
12. ./test_model3.py
13. ./test_gnns.py
14. <s>./test_gcn.py </s>
14. <s>./test_mu.py </s>
  > Testing script for the obtained covering/packing/GNNs models

16. ./train_covering.py
19. ./train_model3.py
18. ./train_gnns.py
15. <s>./train.py</s>
17. <s>./train_gcn.py</s>
20. <s>./train_mu.py</s>
  > Training script for training covering/packing/GNNs models.

## Citation ##
```text
@misc{li2024onthepower
   title={On the Power of Small-size Graph Neural Networks for Linear Programming},
   author={Li, Qian and Ding, Tian and Yang, Linxin and Ouyang, Minghui and Shi, Qingjiang and Sun, Ruoyu},
   year={2024},
   journal={NeurIPS},   
}
```
