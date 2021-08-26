# Recommender System Code Repo

-   추천시스템과 관련한 코드를 보고 구현 및 변경 실험해보는 repository
    -   아직 공부중이며 각 repo는 완벽하지 않을 수 있습니다.
-   데이터 형태부터 평가 방식까지 다룰 부분이 상당히 많아서 알아야 할 부분이 많다.
-   잘 구현해서 배포한 라이브러리 들이 많이 있어 참조할 부분이 많다.
- 읽을 거리
    - http://hoondongkim.blogspot.com/search/label/Recommendation

## Head

-   추천 알고리즘은 크게 2 가지 목적으로 분류할 수 있는 듯 하다.
-   수정하여 목적에 맞게 변형하여 사용할 수는 있을 듯하다.
    -   Top-N 알고리즘
        -   넷플릭스 추천이나 지니 음악추천 등의 알고리즘이 여기에 속한다.
        -   평가는 추천한 값을 실제로 좋아할 지에 대해서 진행한다.
        -   BPR을 적용하면 다른 알고리즘들을 해당 방식으로 효과적이게 적용할 수 있을 것 같다.
    -   Interaction
        -   해당 아이템을 좋아할 지에 대한 binary clf
        -   평가는 AUC로 주로 진행한다.
    -   Sequential
        -   이전 까지 정보를 바탕으로 다음 아이템을 추천
        -   공부 전

## Contents

### FMs

-   Factorization Machines(FM) - 2010
-   Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks(AFM) - 2017
-   DeepFM: A Factorization-Machine based Neural Network for CTR Prediction(DFM) - 2017
-   xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems(xDFM) - 2018

### MFs

- Neural network-based Collaborative Filtering(NCF) - 2017

### Others

- Wide & Deep Learning for Recommender Systems - 2016
- Product-based Neural Networks for User Response Prediction(PNN) - 2016
- Deep & Cross Network for Ad Click Predictions(CDN) - 2017
- AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks - 2019

## 성능비교

-   완벽한 비교라기보다는 구현이 잘 되었는지 확인하기 위한 간단한 테이블
-   사용한 데이터, 파라미터 튜닝에 의해서 달라질 수 있음
    -   실험은 간단한 데이터에 적용하였으며, 구조는 비슷하게 조정하여 실험(각 파일에 들어가서 세부 정보를 볼 수 있음)
-   모델링 방식이 공부하면서 변할 수 있어 추후 수정 예정

### 데이터

-   무비렌즈 100k
    -   **유저와 영화 아이디 정보만 사용**
        -   continuous 등 다른 정보 미포함

### CTR

#### ML - 100k

|    Algo     |  AUC   | Precision | Recall |
| :---------: | :----: | :-------: | :----: |
|     FM      | 0.7663 |  0.6991   | 0.8163 |
| wide & deep | 0.7878 |  0.7322   | 0.7830 |
|     AFM     | 0.7778 |  0.7174   | 0.7906 |
| PNN(inner)  | 0.7879 |  0.7309   | 0.7820 |
| PNN(outer)  | 0.7907 |  0.7307   | 0.8007 |
|     DFM     | 0.7888 |  0.7348   | 0.7813 |
|     CDN     | 0.7906 |  0.7419   | 0.7671 |
|    xDFM     | 0.7906 |  0.7255   | 0.8056 |
|   AutoInt   | 0.7872 |  0.7214   | 0.8109 |

### Hit Ratio

-   훈련 파라미터에 따라서 결과가 상당히 달라질 것 같다.

|     Algo     | HR@10  | NDCG@10 |
| :----------: | :----: | :-----: |
|   MF(ALS)    | 0.6935 | 0.3973  |
| BPR(emb_dot) | 0.7391 | 0.3801  |
|     GMF      | 0.7826 | 0.3574  |
|     MLP      | 0.8017 | 0.3836  |
|     NMF      | 0.8091 | 0.3449  |

### TODO
- torch 버전으로 번역하기
- ml-1m 데이터로 확장하기 [source](https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data)
- AE기반의 경우 sparse matrix 형태로 적용 방법 찾기

