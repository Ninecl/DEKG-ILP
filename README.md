# DEKG-ILP
The source code of Disconnected Emerging Knowledge Graph Oriented Inductive Link Prediction


# Requirements
The required packages are listed in `requirement.txt`


# The extended inductive link prediction experiments
All train and test data can be found in the `data` floder. Specifically, we train our model on `train.txt` in **{dataset}_v{version}**. The main results are evaluated on the `test.txt` in **{dataset}_{version}_mix**, the results with *enclosing links* are evaluated on `test.txt`in **{dataset}_{version}_ind**, the results with *bridging links* are evaluated on `test.txt`in **{dataset}_{version}_bri**.

For example, to train the model DEKG-ILP on version 1 of FB15k-237, run the following command:
``` python
python train -d FB15k-237_v1 -e DEKG-ILP_FB15k-237_v1
```
To test DEKG-ILP, run the following commands:
``` python
# main result
python test_rank.py -d FB15k-237_v1_mix -e DEKG-ILP_FB15k-237_v1
# enclosing links only
python test_rank.py -d FB15k-237_v1_ind -e DEKG-ILP_FB15k-237_v1
# bridging links only
python test_rank.py -d FB15k-237_v1_bri -e DEKG-ILP_FB15k-237_v1
```


# Acknowledgement
Our code refer to the code of [Grail](https://github.com/kkteru/grail). Thanks for their contributions very much.