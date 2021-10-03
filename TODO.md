# SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS(2016) - GRU4Rec

- GRU4Rec
- input 으로 ohe
- session 초기화
- rank1 loss <= 요거 하나 건질만 함

- 모델이 구조적으로 이상해보이는 부분이 많음
  - sequence 모델 발전 전이라 그런듯

# Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding(2018) - Caser

- sequence item을 임베딩 => cnn으로 압축 => fcn => pred
  - horizontal / vertical conv (전에 시퀀스 데이터 다룰 때 쓰면 시도해봐야지 했던 구조!)
  - user embedding 도 fcn에 합쳐줌 (이유가 있긴 하지만 실 적용을 위한 모델 구현시에는 빼는게 좋을 듯)
- final layer 구현부분은 이상하게 느껴진다
  - 다른 아이템 임베딩을 가지고 구현
- svae 방식을 적용하여 뒷부분 바꿔보는 것도 의미가 있을 듯
  - 또는 transformer 방식을 적용

# **Sequential Variational Auto-encoder(2018) - SAVE**

- sequence + vae
  - 최근 vae 기반의 모델이 좋은 성능을 보여주고 있는 것을 고려하면 나쁘지 않은 선택일 수도
- item => embed => rnn => vae => next time item
  - 생각보다 너무 아무것도 없음
  - latent 과정에서 rnn이 들어갈 줄 알았는데 임베딩 과정에만 들어감(encoder 진입 전)
- 그래도 성능이 잘 나온다는 것을 보면 이런 구조를 기반으로 발전시키는 것도 도움이 될 지도

# Self-Attentive Sequential Recommendation(2018) - SASRec

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

# BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer(2019)

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

# PEN4Rec: Preference Evolution Networks for Session-based Recommendation(2021)

- 유저의 히스토리를 그래프로 표현
  - GNN 기반으로 stage 1 작업
  - multi-hop 에 대한 transition 도 캐치하겠다
- 유저의 선호도가 변화한다(evolving)고 보고 이것을 모델링하는 것이 중요
- attention 매커니즘을 활용
- 설명만 봐서는 네트워크 구조를 어떻게 구현한지 이해가 안되는 부분이 많다
  - 코드가 공개되면 다시 한 번 볼 필요가 있을 듯