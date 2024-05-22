# GD-Net #

## This is the git repository for GD-Net, a neural network designed to handle end-to-end solution mapping tasks for covering/packing LPs. ##

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
6. <s>./gen_mis.py </s>
  > code to generate covering/packing instances.

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
