# Session based

## SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS(2016) - GRU4Rec

- GRU4Rec
- input 으로 ohe
- session 초기화
- rank1 loss <= 요거 하나 건질만 함

- 모델이 구조적으로 이상해보이는 부분이 많음
  - sequence 모델 발전 전이라 그런듯

## Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding(2018) - Caser

- sequence item을 임베딩 => cnn으로 압축 => fcn => pred
  - horizontal / vertical conv (전에 시퀀스 데이터 다룰 때 쓰면 시도해봐야지 했던 구조!)
  - user embedding 도 fcn에 합쳐줌 (이유가 있긴 하지만 실 적용을 위한 모델 구현시에는 빼는게 좋을 지도)
- final layer 구현부분은 이상하게 느껴진다
  - 다른 아이템 임베딩을 가지고 구현
- svae 방식을 적용하여 뒷부분 바꿔보는 것도 의미가 있을 듯
  - 또는 transformer 방식을 적용

## A Simple Convolutional Generative Network for Next Item Recommendation(2018) - NextItNet

- Caser를 대놓고 비판+개선 
- pooling 은 추천 task에서 효과적이지 않음! => 제거
- deep + 1d-dilated conv structure
- sequence 내의 모든 아이템이 다음 아이템의 분포를 추정하는데 사용됨
  - attention/bert 류의 모델에서 사용한 방식과 같아보임
  - 마지막 layer에 k*n 이 나오도록 conv + softmax 걸어서 학습
- 후기
  - dilated filter를 사용하여 경량화하며 inter-dependencies를 함께 학습하게 하는 아이디어는 굳
  - 전체적인 구조가 이후에 나온 att-based 방식과 비교하여 큰 차이가 있다고는 잘 모르겠음
  - sub-session도 단점이 좀 명확한 것 같아 의문

## **Sequential Variational Auto-encoder(2018) - SAVE**

- sequence + vae
  - 최근 vae 기반의 모델이 좋은 성능을 보여주고 있는 것을 고려하면 나쁘지 않은 선택일 수도
- item => embed => rnn => vae => next time item
  - 생각보다 너무 아무것도 없음
  - latent 과정에서 rnn이 들어갈 줄 알았는데 임베딩 과정에만 들어감(encoder 진입 전)
- 그래도 성능이 잘 나온다는 것을 보면 이런 구조를 기반으로 발전시키는 것도 도움이 될 지도

## Self-Attentive Sequential Recommendation(2018) - SASRec

- RNN 기반의 방식은 제대로 dynamics를 반영하지 못하므로 attention 기반의 방식을 활용
- 입력은 유저 히스토리, 출력은 입력에서 1 step 후의 히스토리
- attention과정에서 이후 값을 보지 못하도록 masking
- Pointwise feed forward net이 conv1d를 사용한다는 점이 독특
  - 하나의 임베딩 사이즈만큼 convolution 
  - (+) residual connection
- prediction layer에서는 임베딩과 pointwise 연산
  - sigmoid 통과시켜 pos / neg에 대한 log-loss
- (사견) 전반적으로 깔끔한 구조인듯
  - conv로 훑는게 독특 => 실효성에 대해서는 dense로 연결한 것과 실험이 없어서 확인해보면 좋을듯

## Hierarchical Gating Networks for Sequential Recommendation(2019) - HGN

- cnn, rnn, attention module 대신에 gating module을 사용
  - sequence에 대한 gating
    - user 정보 포함
  - instance gating
    - user 정보 포함
- long-term, short-term preference를 catch하기 위해서 모델을 설계
- feature gating -> instance gating -> pooling -> MF(item production)
- 후기
  - gating module의 의미는 알 수 있으나 다른 방법에 비해 어떤 점이 특출난지 와닿지는 않음

## BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer(2019)

- Attention 만 활용하여 예측 모델 만듬
  - transformer encoder
- masked lm 방식으로 훈련
  - input에 랜덤 마스크
  - loss = ce_loss <- masking 된 부분만 계산하도록 하기
- 기존 bert 는 pre-trained 모델이 목표지만 이 모델은 end-to-end
- 예측을 위해 인풋 마지막에 mask 추가
  - (x, ?) => (x, y) 형태
  - 예측할 때는 마지막만 사용하면 됨
    - multi period 는 언급이 없음 => 필요하다면 학습할 때 부터 처리해야 할 듯

## PEN4Rec: Preference Evolution Networks for Session-based Recommendation(2021)

- 유저의 히스토리를 그래프로 표현
  - 순차적 방향 그래프
  - GNN 기반으로 stage 1 작업
  - multi-hop 에 대한 transition 도 캐치하겠다
- 유저의 선호도가 변화한다(evolving)고 보고 이것을 모델링하는 것이 중요
- attention 매커니즘을 활용
- 설명만 봐서는 네트워크 구조를 어떻게 구현한지 이해가 안되는 부분이 많다
  - 코드가 공개되면 다시 한 번 볼 필요가 있을 듯

# Others

## Recommending What Video to Watch Next: A Multitask Ranking System(2019)

- mmoe + bias reduction
  - bias 줄 수 있는 부분을 학습때는 함께 사용하여 학습 => serving시에는 고정
    - regularizer 로 처리할수도 있으나 부적합하다고 판단
    - shallow tower 로 wide component역할
- 다양한 방식들이 있지만 실제로 serving 하기 위한 모델로는 부적합하기 때문에 위와 같은 구조를 선택
  - mha, cnn ...

- 후기
  - 처음으로 practical use 관점에서 모델 개발에 대한 시각을 접함
  - 실제 서비스하면서 최근 데이터로 계속 학습한다고 하는데 어떤 방식으로 처리하는지 궁금하다
    - 임베딩이 들어가면 매 번 다시짜는 것인가..
    - 그런데 임베딩 행렬의 크기가 그러면 끝도 없이 늘어나게 되는데..

