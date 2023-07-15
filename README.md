
# Algorithm
|             Model              |                                      Model Code                                      |                           Example Code                           |
| :----------------------------: | :----------------------------------------------------------------------------------: | :--------------------------------------------------------------: |
| Matrix Factorization with SGD  |                             [`model`](./src/model/MF.py)                             | [`movielens example`](./notebook/example_MF_SGD_movielens.ipynb) |
| Neural Collaborative Filtering |    [`model`](./src/model/NCF.py), [`lit_model`](./src/lit_model/NCF_lit_model.py)    |             [`KMRD-small example`](./src/run_NCF.py)             |
|     Factorization Machine      |                             [`model`](./src/model/FM.py)                             |  [`KMRD-small example`](./notebook/example_FM_KMRD_small.ipynb)  |
|             DeepFM             | [`model`](./src/model/DeepFM.py), [`lit_model`](./src/lit_model/DeepFM_lit_model.py) |           [`KMRD-small example`](./src/run_DeepFM.py)            |