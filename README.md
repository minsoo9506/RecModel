- 추천알고리즘을 구현하는 레포지토리입니다.
- 라이브러리화 하지는 않습니다.
- '구현을 통해 모델 이해하기', '다른 코드 읽어보기'를 목표로 합니다.

# Algorithm
### Factorization Machines
- [Factorization Machines, 2010](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
- code: [`torch model`](./src/model/FM.py), [`KMRD-small example`](./notebook/FM_KMRD_small.ipynb)

### NeuMF
- [Neural Collaborative Filtering, 2017](https://arxiv.org/pdf/1708.05031.pdf)
- code: [`torch model`](./src/model/NeuMF.py), [`lit model`](./src/lit_model/lit_NeuMF.py), [`KMRD-small example`](./notebook/NeuMF_KMRD_small.ipynb)

### DeepFM
- [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017](https://arxiv.org/abs/1703.04247)
- code: [`torch model`](./src/model/DeepFM.py), [`lit model`](./src/lit_model/lit_DeepFM.py), [`KMRD-small example`](./notebook/DeepFM_KMRD_small.ipynb)

### Two-Tower
- [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations, RecSys 2019](https://research.google/pubs/pub48840/)
- code: [`torch model`](./src/model/TwoTower.py), [`lit model`](./src/lit_model/lit_TwoTower.py), [`KuaiRec example`](./notebook/Two_Tower_KuaiRec.ipynb)
>>>>>>> develop

# Benchmark
## Data
- 데이터에 대한 설명, 예시데이터 추가

## Result
- 모델을 돌린 metric 결과, 훈련시간 등