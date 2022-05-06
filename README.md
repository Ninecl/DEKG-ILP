# DEKG-ILP
The source code of Disconnected Emerging Knowledge Graph Oriented Inductive Link Prediction


# Requirements
The required packages are listed in `requirement.txt`


# The extended inductive link prediction experiments
All train and test data can be found in the `data` floder. Specifically, we train our model on `train.txt` in **{dataset}_{version}**. The main results are evaluated on the `test.txt` in **{dataset}_{version}_mix**, the results with *enclosing links* are evaluated on `test.txt`in **{dataset}_{version}_enc**, the results with *bridging links* are evaluated on `test.txt`in **{dataset}_{version}_bri**.

For example, to train the model DEKG-ILP on EQ of FB15k-237, run the following command:
``` python
python train.py -d FB15k-237_EQ -e DEKG-ILP_FB15k-237_EQ
```
To test DEKG-ILP, run the following commands:
``` python
# main result
python test_rank.py -d FB15k-237_EQ_mix -e DEKG-ILP_FB15k-237_EQ
# enclosing links only
python test_rank.py -d FB15k-237_EQ_enc -e DEKG-ILP_FB15k-237_EQ
# bridging links only
python test_rank.py -d FB15k-237_EQ_bri -e DEKG-ILP_FB15k-237_EQ
```


# Acknowledgement
Our code refer to the code of [Grail](https://github.com/kkteru/grail). Thanks for their contributions very much.