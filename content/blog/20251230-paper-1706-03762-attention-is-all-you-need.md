---
title: "[논문 리뷰] Attention Is All You Need"
date: "2025-12-30"
excerpt: "Transformer 아키텍처를 최초로 제안한 획기적인 논문. RNN과 CNN을 완전히 배제하고 Attention 메커니즘만으로 시퀀스 모델링의 새로운 패러다임을 제시하여, BERT, GPT 등 현대 자연어처리의 기반을 마련했습니다."
category: "Paper Review"
tags: ["Paper Review", "Transformer", "Attention", "NLP", "Deep Learning"]
thumbnail: "/assets/images/blog/20251230-paper-1706-03762-attention-is-all-you-need.jpg"
---

# [논문 리뷰] Attention Is All You Need

**저자**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
**출판**: NeurIPS 2017
**논문 링크**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## TL;DR

Attention 메커니즘만을 사용한 **Transformer** 모델이 기존의 순환 신경망(RNN)과 합성곱 신경망(CNN) 기반의 모델을 완전히 대체하며 뛰어난 성능을 달성했습니다. 병렬화가 용이하여 훈련 시간을 크게 단축시켰고, WMT 2014 영어-독일어 번역에서 **28.4 BLEU**, 영어-프랑스어 번역에서 **41.8 BLEU**를 기록하며 새로운 최고 성능을 수립했습니다. 이 논문은 BERT, GPT, T5 등 현대 자연어처리의 기반이 된 혁명적인 연구입니다.

---

## 1. 연구 배경 및 의의

### 1.1 자연어처리의 역사적 맥락

2017년 이전까지 자연어처리 분야에서 시퀀스 변환(sequence transduction) 작업은 주로 **순환 신경망(RNN)**, 특히 **LSTM**(Long Short-Term Memory)과 **GRU**(Gated Recurrent Unit)를 사용했습니다. 이러한 모델들은 다음과 같은 구조적 특징을 가지고 있었습니다:

- **순차적 처리**: 입력 시퀀스를 왼쪽에서 오른쪽으로 순차적으로 처리
- **은닉 상태의 전달**: 이전 시점의 정보를 은닉 상태(hidden state)를 통해 다음 시점으로 전달
- **Encoder-Decoder 구조**: Sutskever et al. (2014)이 제안한 seq2seq 모델이 표준

### 1.2 기존 접근법의 한계

Transformer가 등장하기 전 RNN/CNN 기반 모델들은 다음과 같은 근본적인 한계를 가지고 있었습니다:

#### (1) RNN 기반 모델의 문제점

**병렬화 불가능**
- 시퀀스를 순차적으로 처리해야 하므로 t번째 시점의 계산이 t-1번째 시점의 결과에 의존
- GPU의 병렬 연산 능력을 효과적으로 활용할 수 없음
- 긴 시퀀스(예: 1000+ 토큰)를 처리할 때 훈련 시간이 기하급수적으로 증가

**장거리 의존성(Long-range Dependency) 문제**
- 시퀀스의 시작 부분과 끝 부분 사이의 관계를 학습하기 어려움
- LSTM/GRU가 이를 완화했지만 완전히 해결하지는 못함
- Gradient vanishing/exploding 문제가 여전히 존재

**메모리 제약**
- 배치 크기가 시퀀스 길이에 반비례하여 제한됨
- 긴 시퀀스를 처리할 때 큰 배치를 사용하기 어려움

#### (2) CNN 기반 모델의 문제점

ByteNet, ConvS2S 등의 합성곱 기반 모델들도 시도되었지만:

- **제한된 수용 영역(Receptive Field)**: 두 위치 간의 관계를 학습하기 위해 필요한 연산 수가 거리에 비례하여 증가
- **계산 복잡도**: O(n/k) (ConvS2S) 또는 O(log_k(n)) (ByteNet)로, 여전히 거리에 의존적
- **위치 정보 인코딩의 어려움**: 시퀀스의 순서 정보를 자연스럽게 인코딩하기 어려움

### 1.3 Attention 메커니즘의 등장

Bahdanau et al. (2015)이 제안한 Attention 메커니즘은 RNN의 고정된 크기의 context vector 문제를 해결했습니다:

- Decoder가 입력 시퀀스의 모든 위치에 직접 접근 가능
- 각 출력 시점마다 가장 관련 있는 입력 부분에 "주목(attend)"
- 하지만 여전히 RNN 구조 내부에서 보조적으로만 사용됨

**Transformer의 혁신**: Attention을 보조 메커니즘이 아닌 **유일한 구조**로 사용하여 RNN/CNN을 완전히 제거

---

## 2. 기존 RNN/CNN 기반 모델의 한계 상세 분석

### 2.1 계산 복잡도 비교

다양한 레이어 유형의 계산 복잡도를 비교하면:

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|-----------|---------------------|----------------------|-------------------|
| Self-Attention | O(n² · d) | O(1) | O(1) |
| Recurrent | O(n · d²) | O(n) | O(n) |
| Convolutional | O(k · n · d²) | O(1) | O(log_k(n)) |
| Self-Attention (restricted) | O(r · n · d) | O(1) | O(n/r) |

여기서:
- n: 시퀀스 길이
- d: 표현 차원 (representation dimension)
- k: 커널 크기 (convolutional layer)
- r: 제한된 이웃 크기 (restricted self-attention)

**핵심 관찰**:
- Self-Attention의 **Sequential Operations**가 O(1): 완벽한 병렬화 가능
- **Maximum Path Length**가 O(1): 어떤 두 위치든 한 번의 연산으로 연결 가능
- 계산 복잡도는 O(n² · d)이지만, 대부분의 실용적 경우 n < d이므로 Recurrent보다 효율적

### 2.2 해석 가능성(Interpretability)

Self-Attention은 추가적인 이점을 제공합니다:

- **Attention 가중치 시각화**: 모델이 어느 단어에 주목하는지 명확히 확인 가능
- **문법적/의미적 구조 학습**: Multi-head attention의 각 헤드가 다른 언어학적 현상을 포착
- **디버깅 용이성**: 문제가 발생했을 때 어느 부분에서 잘못되었는지 추적 가능

---

## 3. Transformer 아키텍처 상세 분석

### 3.1 전체 구조 개요

Transformer는 **Encoder-Decoder** 구조를 따르며, 둘 다 여러 개의 동일한 레이어를 쌓아 올린 형태입니다.

```
Input → Encoder Stack → Encoder Output → Decoder Stack → Output
                          ↓
                   (Cross-Attention에서 사용)
```

**주요 구성 요소**:
1. **Encoder**: N=6개의 동일한 레이어로 구성
2. **Decoder**: N=6개의 동일한 레이어로 구성
3. **각 레이어**: Multi-Head Attention + Position-wise FFN
4. **Residual Connection + Layer Normalization**: 모든 서브레이어에 적용

### 3.2 Encoder 구조

각 Encoder 레이어는 두 개의 서브레이어로 구성됩니다:

#### (1) Multi-Head Self-Attention
- 입력 시퀀스의 모든 위치가 서로를 참조
- 각 단어가 문맥 내 다른 모든 단어와의 관계를 학습

#### (2) Position-wise Feed-Forward Network
- 각 위치에 독립적으로 적용되는 완전 연결 신경망
- 비선형 변환을 통해 표현력 증가

**수식적 표현**:
```
EncoderLayer(x) = LayerNorm(x + Sublayer(x))

여기서:
- Sublayer₁(x) = MultiHeadAttention(x, x, x)
- Sublayer₂(x) = FFN(x)
```

**Residual Connection의 중요성**:
- Gradient flow 개선: 깊은 네트워크에서도 안정적인 학습
- 항등 함수(identity mapping)를 학습 가능: 레이어가 필요 없으면 건너뛸 수 있음
- 모든 서브레이어의 출력 차원을 d_model = 512로 통일

### 3.3 Decoder 구조

Decoder는 Encoder와 유사하지만 세 개의 서브레이어를 가집니다:

#### (1) Masked Multi-Head Self-Attention
- **Masking**: 미래 위치의 정보를 보지 못하도록 차단
- Auto-regressive 속성 유지: i번째 출력이 i 이전 위치만 참조

#### (2) Encoder-Decoder Attention (Cross-Attention)
- Query: Decoder의 이전 레이어 출력
- Key, Value: Encoder의 최종 출력
- 출력이 입력 시퀀스의 어느 부분에 주목할지 결정

#### (3) Position-wise Feed-Forward Network
- Encoder와 동일한 구조

**수식적 표현**:
```
DecoderLayer(x, enc_output) = LayerNorm(x + Sublayer(x))

여기서:
- Sublayer₁(x) = MaskedMultiHeadAttention(x, x, x)
- Sublayer₂(x) = MultiHeadAttention(x, enc_output, enc_output)
- Sublayer₃(x) = FFN(x)
```

---

## 4. Scaled Dot-Product Attention 수식 및 원리

### 4.1 기본 Attention 메커니즘

Attention 함수는 **Query**와 **Key-Value** 쌍을 출력에 매핑하는 함수입니다.

**핵심 수식**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

여기서:
- **Q** (Query): (n × d_k) 행렬, "무엇을 찾고 싶은가"
- **K** (Key): (m × d_k) 행렬, "각 값의 주소"
- **V** (Value): (m × d_v) 행렬, "실제 값"
- **d_k**: Key의 차원
- **n**: Query 시퀀스 길이
- **m**: Key-Value 시퀀스 길이

### 4.2 계산 과정 단계별 분석

#### Step 1: Dot-Product 계산
```
S = QK^T  (n × m 행렬)
```
- Q의 각 행(query vector)과 K의 각 행(key vector) 사이의 유사도 계산
- S[i,j]는 i번째 query와 j번째 key의 유사도

#### Step 2: Scaling
```
S_scaled = S / √d_k
```

**왜 Scaling이 필요한가?**

d_k가 클 때의 문제:
- Dot product의 값이 매우 커질 수 있음 (분산이 d_k에 비례)
- Softmax의 입력값이 극단적으로 크거나 작아짐
- Gradient가 매우 작아지는 영역으로 softmax가 밀림 (saturation)

**수학적 설명**:
Query와 Key의 각 요소가 평균 0, 분산 1인 독립 확률변수라고 가정하면:
- q · k = Σ(q_i · k_i)의 평균은 0
- 분산은 d_k

따라서 √d_k로 나누면 분산이 1로 정규화됩니다.

**실험적 검증**:
논문 저자들은 d_k가 작을 때(예: d_k=64)는 scaling 여부가 큰 차이를 보이지 않지만, d_k가 클 때(예: d_k=512)는 scaling이 없으면 성능이 크게 저하됨을 확인했습니다.

#### Step 3: Softmax
```
A = softmax(S_scaled)  (각 행이 확률 분포)
```
- 각 query에 대해 모든 key에 대한 attention weight 계산
- Σ_j A[i,j] = 1 (확률 분포)

#### Step 4: Weighted Sum
```
Output = A · V
```
- Attention weight로 value들의 가중합 계산
- i번째 출력 = Σ_j A[i,j] · V[j]

### 4.3 Additive Attention과의 비교

Bahdanau et al.이 제안한 **Additive Attention**:

$$
\text{Attention}(q, k) = v^T \tanh(W_q q + W_k k)
$$

**비교**:
- **계산 복잡도**: Dot-product가 이론적으로 동일하지만 실제로 더 빠름
- **최적화**: 고도로 최적화된 행렬 곱셈 코드 사용 가능
- **성능**: Scaling을 적용하면 dot-product가 더 우수
- **메모리**: Additive attention이 추가 파라미터 필요 (W_q, W_k, v)

---

## 5. Multi-Head Attention 구조 및 이점

### 5.1 Multi-Head Attention의 동기

단일 attention을 d_model 차원에서 수행하는 대신, **여러 번 다른 학습된 선형 투영(projection)으로** 수행하는 것이 더 유익합니다.

**핵심 아이디어**:
- 모델이 다양한 representation subspace의 정보를 동시에 학습
- 각 head가 다른 측면의 관계를 포착

### 5.2 수식 정의

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

여기서 각 head는:

$$
\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

**파라미터**:
- W^Q_i ∈ R^(d_model × d_k): Query 투영 행렬
- W^K_i ∈ R^(d_model × d_k): Key 투영 행렬
- W^V_i ∈ R^(d_model × d_v): Value 투영 행렬
- W^O ∈ R^(h·d_v × d_model): 출력 투영 행렬

**논문의 설정**:
- h = 8 (병렬 attention 레이어 수)
- d_k = d_v = d_model / h = 64
- 총 계산 비용은 전체 차원의 단일 head attention과 유사

### 5.3 각 Head가 학습하는 것

논문의 실험 결과, 각 attention head가 다른 언어적 현상을 포착함이 밝혀졌습니다:

**예시**:
- **Head 1**: 구문 의존성(syntactic dependencies) - 주어와 동사의 관계
- **Head 2**: 공지시(coreference) - 대명사와 선행사의 관계
- **Head 3**: 장거리 의존성(long-range dependencies)
- **Head 4**: 인접 단어 관계(local dependencies)
- **Head 5-8**: 의미적 유사성, 품사 관계 등

### 5.4 계산 효율성

**차원 분석**:
- 단일 head (d_model 차원): d_model²
- Multi-head (h개, 각 d_model/h 차원): h · (d_model/h)² = d_model²/h
- 출력 투영 W^O: h · d_v · d_model = d_model²

**총 계산량**: 단일 head와 동일하면서도 표현력은 훨씬 풍부

### 5.5 Transformer에서의 세 가지 사용

#### (1) Encoder-Decoder Attention
- Query: Decoder layer
- Key, Value: Encoder output
- Decoder의 모든 위치가 입력 시퀀스의 모든 위치에 접근

#### (2) Encoder Self-Attention
- Query, Key, Value: 모두 이전 encoder layer의 출력
- 각 위치가 이전 레이어의 모든 위치에 접근

#### (3) Decoder Self-Attention
- Query, Key, Value: 모두 이전 decoder layer의 출력
- **Masking**: 왼쪽 위치만 접근 가능 (auto-regressive)
- i번째 위치는 i 이하의 위치만 참조

---

## 6. Position-wise Feed-Forward Networks

### 6.1 구조 및 수식

각 encoder와 decoder 레이어는 attention 서브레이어 외에 완전 연결 feed-forward network를 포함합니다.

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

**구성**:
- 두 개의 선형 변환과 ReLU 활성화 함수
- **입력/출력 차원**: d_model = 512
- **은닉층 차원**: d_ff = 2048
- 4배 확장 후 축소

### 6.2 "Position-wise"의 의미

**중요한 특징**:
- 같은 파라미터가 모든 위치에 적용됨
- 하지만 **레이어마다 다른 파라미터** 사용
- 커널 크기가 1인 두 번의 convolution으로 해석 가능

**수식으로 표현**:
```
각 위치 i에 대해:
output[i] = FFN(input[i])

즉, 시퀀스의 각 위치가 독립적으로 동일한 변환을 거침
```

### 6.3 역할 및 중요성

**왜 필요한가?**

Attention 메커니즘은 선형 변환입니다:
- Query, Key, Value 모두 선형 투영
- Weighted sum도 선형 연산

**FFN의 역할**:
- **비선형성 추가**: ReLU를 통한 비선형 변환
- **표현력 증가**: 더 복잡한 함수 학습 가능
- **차원 확장/축소**: 정보 병목(bottleneck) 방지

**실험적 검증**:
- FFN 제거 시 성능 크게 저하
- d_ff = 2048이 최적 (더 크거나 작으면 성능 하락)

---

## 7. Positional Encoding

### 7.1 문제 정의

Transformer는 recurrence나 convolution을 사용하지 않으므로:
- 시퀀스의 순서 정보가 전혀 없음
- "I love you"와 "you love I"를 구분할 수 없음

**해결책**: Positional Encoding을 입력에 추가

### 7.2 Sinusoidal Positional Encoding

논문에서 사용한 방법:

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$

$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

여기서:
- **pos**: 시퀀스 내 위치 (0, 1, 2, ...)
- **i**: 차원 인덱스 (0, 1, ..., d_model/2 - 1)
- **짝수 차원**: sine 함수
- **홀수 차원**: cosine 함수

### 7.3 설계 원리

**주기성(Wavelength)**:
- 각 차원의 주기가 2π에서 10000·2π까지의 기하급수
- 낮은 차원: 빠르게 변화 (세밀한 위치 정보)
- 높은 차원: 천천히 변화 (거시적 위치 정보)

**상대적 위치 표현**:
임의의 고정된 offset k에 대해, PE_{pos+k}를 PE_{pos}의 선형 함수로 표현 가능:

$$
\begin{bmatrix} \sin(pos+k) \\ \cos(pos+k) \end{bmatrix} =
\begin{bmatrix} \cos k & \sin k \\ -\sin k & \cos k \end{bmatrix}
\begin{bmatrix} \sin pos \\ \cos pos \end{bmatrix}
$$

이는 모델이 상대적 위치를 쉽게 학습할 수 있게 합니다.

### 7.4 학습 가능한 Positional Embedding과의 비교

**두 가지 선택**:
1. **학습 가능한 positional embedding** (BERT 등에서 사용)
2. **고정된 sinusoidal encoding** (Transformer 원 논문)

**실험 결과**:
- 두 방식이 거의 동일한 성능
- Sinusoidal의 장점:
  - 훈련 시보다 긴 시퀀스에 외삽(extrapolate) 가능
  - 파라미터 수 감소
  - 명시적인 수학적 의미

**현대 모델의 선택**:
- BERT: 학습 가능한 positional embedding
- GPT-2/3: 학습 가능한 positional embedding
- T5: Relative positional encoding (더 발전된 형태)

---

## 8. 학습 및 정규화

### 8.1 Optimizer

**Adam Optimizer** 사용:
- β₁ = 0.9
- β₂ = 0.98
- ε = 10⁻⁹

### 8.2 Learning Rate Schedule

**Warmup 전략**:

$$
lrate = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
$$

**특징**:
- 처음 warmup_steps 동안 learning rate를 선형적으로 증가
- 이후 step number의 역 제곱근에 비례하여 감소
- warmup_steps = 4000

**효과**:
- 초기 불안정성 방지
- 더 큰 learning rate 사용 가능
- 더 빠른 수렴

### 8.3 Regularization

#### (1) Residual Dropout
- 각 서브레이어의 출력에 dropout 적용
- Residual connection과 더하기 전
- P_drop = 0.1

#### (2) Attention Dropout
- Attention weights에 dropout 적용
- Softmax 직후, value와 곱하기 전
- P_drop = 0.1

#### (3) Label Smoothing
- ε_ls = 0.1
- One-hot 타겟 대신 smoothed distribution 사용
- 과적합 방지 및 일반화 성능 향상

**효과**:
- Perplexity는 약간 증가 (불확실성 증가)
- BLEU 점수는 향상 (번역 품질 개선)
- 모델이 과도하게 확신하는 것을 방지

---

## 9. 실험 결과 상세 분석

### 9.1 기계 번역 성능

#### WMT 2014 영어-독일어

| Model | BLEU | Parameters | Training Cost |
|-------|------|-----------|---------------|
| ByteNet | 23.75 | - | - |
| Deep-Att + PosUnk | 23.75 | - | - |
| GNMT + RL | 24.6 | - | - |
| ConvS2S | 25.16 | - | - |
| MoE | 26.03 | - | - |
| **Transformer (base)** | **27.3** | 65M | **3.3 days (8 P100)** |
| **Transformer (big)** | **28.4** | 213M | **3.5 days (8 P100)** |

**주목할 점**:
- 기존 최고 성능 대비 +2.0 BLEU 이상 향상
- 훈련 비용이 기존 모델의 1/10 수준
- Base model도 이전 최고 성능 초과

#### WMT 2014 영어-프랑스어

| Model | BLEU | Training Cost |
|-------|------|---------------|
| Previous SOTA (ensemble) | 41.0 | - |
| **Transformer (big)** | **41.8** | **1/4 of previous** |

**특징**:
- 단일 모델로 이전 앙상블 성능 초과
- dropout rate 0.1 → 0.3 증가로 과적합 방지

### 9.2 Model Variations 분석

논문은 Transformer의 다양한 구성 요소에 대한 ablation study를 수행했습니다:

#### (A) Attention Head 수의 영향

| Heads (h) | d_k | d_v | BLEU | PPL |
|-----------|-----|-----|------|-----|
| 1 | 512 | 512 | 25.8 | 6.01 |
| 4 | 128 | 128 | 25.9 | 5.88 |
| 8 | 64 | 64 | **26.0** | **5.86** |
| 16 | 32 | 32 | 25.5 | 5.90 |

**결론**:
- h=8이 최적
- 너무 많은 head는 오히려 성능 저하
- Multi-head의 효과 확인

#### (B) Attention Key 차원의 영향

| d_k | BLEU | Quality |
|-----|------|---------|
| 256 | 25.6 | Worse |
| 512 | 25.8 | Worse |
| 1024 | 25.5 | Worse |

**결론**:
- 더 큰 d_k가 반드시 좋은 것은 아님
- Compatibility 함수의 복잡도가 중요

#### (C) Model 크기의 영향

| d_model | d_ff | h | BLEU | Parameters |
|---------|------|---|------|-----------|
| 256 | 1024 | 4 | 24.9 | Small |
| 512 | 2048 | 8 | **26.0** | 65M |
| 1024 | 4096 | 16 | 25.7 | Large |

**결론**:
- Base model (d_model=512)이 성능/효율성 균형점
- 단순히 크기를 늘린다고 성능이 계속 향상되지 않음

#### (D) Dropout의 영향

| P_drop | BLEU | PPL |
|--------|------|-----|
| 0.0 | 25.3 | 6.15 |
| 0.1 | **26.0** | **5.86** |
| 0.2 | 25.8 | 5.92 |

**결론**:
- Dropout이 필수적
- 0.1이 최적 (큰 데이터셋에서는 0.3도 효과적)

#### (E) Positional Encoding 비교

| Method | BLEU |
|--------|------|
| Learned | 25.9 |
| Sinusoidal | **26.0** |

**결론**:
- 두 방식이 거의 동일
- Sinusoidal이 외삽 능력에서 유리

### 9.3 영어 Constituency Parsing

번역 외 다른 task에서도 테스트:

**WSJ Dataset**:
- 40K 문장으로 훈련
- 특별한 튜닝 없이 Transformer 적용

**결과**:
- F1 Score: **92.7** (이전 SOTA: 91.3)
- RNN 기반 모델 대비 우수
- 작은 데이터셋에서도 효과적임을 증명

---

## 10. 핵심 그림 및 표 설명

### 10.1 Figure 1: The Transformer Architecture

논문의 가장 중요한 그림으로, Transformer의 전체 구조를 보여줍니다:

**왼쪽 (Encoder)**:
- 입력 임베딩 + Positional Encoding
- N=6개의 동일한 레이어
- 각 레이어: Multi-Head Attention → Add & Norm → FFN → Add & Norm

**오른쪽 (Decoder)**:
- 출력 임베딩 + Positional Encoding
- N=6개의 동일한 레이어
- 각 레이어:
  1. Masked Multi-Head Attention → Add & Norm
  2. Multi-Head Attention (encoder 참조) → Add & Norm
  3. FFN → Add & Norm
- 최종: Linear → Softmax

**연결**:
- Encoder의 출력이 Decoder의 모든 레이어에 입력
- Decoder는 auto-regressive하게 출력 생성

### 10.2 Figure 2: Scaled Dot-Product Attention vs Multi-Head Attention

**왼쪽**: Scaled Dot-Product Attention의 계산 흐름
- MatMul (Q, K^T)
- Scale (÷√d_k)
- Mask (optional, decoder에서 사용)
- SoftMax
- MatMul (결과, V)

**오른쪽**: Multi-Head Attention
- 여러 개의 Linear 투영 (h개)
- 각각 Scaled Dot-Product Attention 적용
- Concat
- 최종 Linear 투영

### 10.3 Table 1: Maximum Path Lengths, Per-Layer Complexity

앞서 설명한 계산 복잡도 비교 표로, Self-Attention의 효율성을 보여줍니다.

### 10.4 Table 2: English-German Translation Results

WMT 2014 영어-독일어 번역 결과 비교표:
- Transformer가 모든 이전 모델 초과
- 훈련 비용 대비 성능이 압도적

### 10.5 Table 3: Variations on the Transformer Architecture

Ablation study 결과:
- 각 구성 요소의 중요성 입증
- 최적 하이퍼파라미터 발견

---

## 11. 이 논문이 미친 영향

### 11.1 자연어처리의 패러다임 전환

Transformer는 NLP의 역사를 **"Transformer 이전"**과 **"Transformer 이후"**로 나눌 만큼 혁명적이었습니다.

#### Before Transformer (2017 이전)
- RNN/LSTM 중심
- 작은 모델 (<100M parameters)
- Task-specific 아키텍처
- 제한된 사전학습

#### After Transformer (2017 이후)
- Attention 중심
- 대규모 모델 (>1B parameters)
- 범용 아키텍처
- 대규모 사전학습 + Fine-tuning

### 11.2 직접적인 후속 연구

#### (1) BERT (2018, Google)
**"Bidirectional Encoder Representations from Transformers"**

- Transformer의 **Encoder만** 사용
- Masked Language Modeling (MLM) 사전학습
- 양방향 context 학습
- 11개 NLP task에서 SOTA 달성

**영향**:
- Pre-training + Fine-tuning 패러다임 확립
- NLP의 "ImageNet moment"
- BERT-base: 110M params, BERT-large: 340M params

#### (2) GPT (2018, OpenAI)
**"Generative Pre-trained Transformer"**

- Transformer의 **Decoder만** 사용
- Auto-regressive Language Modeling
- Generative task에 특화

**진화**:
- GPT-2 (2019): 1.5B params, zero-shot learning
- GPT-3 (2020): 175B params, few-shot learning
- GPT-4 (2023): Multimodal, 추정 1T+ params

#### (3) T5 (2019, Google)
**"Text-to-Text Transfer Transformer"**

- 모든 NLP task를 text-to-text로 통일
- Encoder-Decoder 구조 유지
- 11B params

#### (4) XLNet, RoBERTa, ALBERT, ELECTRA...
수많은 변형 및 개선 모델들이 등장

### 11.3 다른 도메인으로의 확장

#### Computer Vision
**Vision Transformer (ViT, 2020)**:
- 이미지를 패치로 나누어 시퀀스처럼 처리
- ImageNet에서 CNN 성능 초과
- SWIN Transformer, DeiT 등 후속 연구

**DETR (Detection Transformer)**:
- Object detection을 set prediction으로
- Anchor box 제거

#### Speech Processing
**Wav2Vec 2.0**:
- 음성을 Transformer로 직접 처리
- Self-supervised learning

**Whisper (OpenAI)**:
- 대규모 음성 인식 모델
- 다국어, robust

#### Multimodal Learning
**CLIP (OpenAI)**:
- Vision + Language
- Zero-shot image classification

**Flamingo, DALL-E, Stable Diffusion**:
- Text-to-image generation
- Transformer가 핵심 구성 요소

#### Reinforcement Learning
**Decision Transformer**:
- RL을 sequence modeling 문제로
- Trajectory optimization

### 11.4 산업계 영향

#### Google
- BERT → Search 품질 향상
- LaMDA, PaLM: 대화 AI
- Bard (Gemini): ChatGPT 경쟁

#### OpenAI
- GPT-3 → ChatGPT (2022년 11월)
- 100M+ 사용자 (2개월)
- GPT-4: Multimodal capabilities

#### Meta
- LLaMA: Open-source LLM
- Research democratization

#### Microsoft
- Azure OpenAI Service
- Bing Chat
- GitHub Copilot

### 11.5 사회적 영향

**긍정적 영향**:
- 언어 장벽 해소 (번역)
- 접근성 향상 (TTS, STT)
- 생산성 증가 (코딩 도우미, 글쓰기 보조)
- 교육 혁신 (개인화 학습)

**우려 사항**:
- 환경 비용 (대규모 모델 훈련)
- 편향성 및 공정성
- 잘못된 정보 생성
- 일자리 대체 우려
- AI 안전성 및 정렬 문제

---

## 12. 한계점 및 향후 연구 방향

### 12.1 논문에서 언급한 한계

#### (1) 긴 시퀀스 처리
- Self-attention의 O(n²) 복잡도
- 메모리 사용량이 시퀀스 길이의 제곱에 비례
- 실용적 한계: ~512-1024 토큰

**해결 연구**:
- Sparse Transformer: O(n√n)
- Linformer: O(n)
- Performer: O(n)
- Longformer: Sliding window + global attention

#### (2) 비텍스트 데이터
- 이미지, 오디오, 비디오는 시퀀스가 매우 김
- 국소적 주목 메커니즘 필요

**해결 연구**:
- Vision Transformer: Patch 기반
- Swin Transformer: Hierarchical
- Perceiver: Cross-attention

#### (3) 해석 가능성
- Attention weight이 항상 명확한 설명을 제공하지는 않음
- "Attention is not Explanation" 논쟁

**후속 연구**:
- Attention rollout
- Attention flow
- Integrated gradients

### 12.2 후속 연구에서 발견된 한계

#### (1) Data Efficiency
- 대규모 데이터 필요
- Few-shot, zero-shot 학습에 한계

**개선**:
- Meta-learning
- Prompt engineering
- In-context learning (GPT-3)

#### (2) Computational Cost
- 모델 크기가 급격히 증가
- 환경 비용 및 접근성 문제

**개선**:
- Model compression
- Knowledge distillation
- Efficient training (FlashAttention)
- MoE (Mixture of Experts)

#### (3) Generalization
- Distribution shift에 취약
- Spurious correlation 학습

**개선**:
- Robust training
- Data augmentation
- Adversarial training

### 12.3 미래 연구 방향

#### (1) Efficient Transformers
- O(n) 또는 O(n log n) complexity
- Memory-efficient attention
- Dynamic computation

#### (2) Multimodal Transformers
- Vision + Language + Audio 통합
- Universal representation learning
- Cross-modal transfer

#### (3) Continual Learning
- Catastrophic forgetting 방지
- Lifelong learning
- 효율적인 fine-tuning (LoRA, Adapter)

#### (4) Neuro-Symbolic AI
- Symbolic reasoning + Neural networks
- Interpretability 향상
- Compositional generalization

---

## 13. 개인 의견 및 결론

### 13.1 논문의 혁신성

Transformer는 단순히 성능이 좋은 모델을 넘어서 **패러다임 자체를 바꾼** 연구입니다. 특히 다음 측면에서 혁신적입니다:

**1. 단순성과 우아함**
- RNN의 복잡한 gating mechanism 제거
- CNN의 복잡한 계층 구조 제거
- Attention이라는 단일 메커니즘으로 통일

**2. 병렬화**
- GPU/TPU의 성능을 완전히 활용
- 대규모 모델 훈련의 길을 열음
- 연구 속도 가속화

**3. 범용성**
- NLP에서 시작했지만 거의 모든 도메인으로 확장
- "Attention is all you need"가 다양한 분야에서 입증됨
- Universal architecture로서의 가능성

### 13.2 실무적 가치

**산업계 관점**:
- 번역, 검색, 추천 시스템에서 즉각적인 개선
- API 형태로 쉽게 배포 가능 (OpenAI API, Hugging Face)
- 비전문가도 pre-trained model로 활용 가능

**연구자 관점**:
- 명확한 baseline
- 재현 가능성 높음 (코드 공개)
- 다양한 변형 실험 가능

### 13.3 7년 후의 평가 (2017→2024)

2017년 발표 당시에는 "흥미로운 번역 모델" 정도로 여겨졌지만, 지금은:

**실현된 영향**:
- ChatGPT의 기반 기술
- 수십억 달러 규모의 산업 창출
- AI의 대중화
- Generative AI 혁명의 시작점

**예상을 넘어선 발전**:
- 당시 상상하지 못한 Few-shot learning (GPT-3)
- Emergent abilities (대규모 모델의 창발적 능력)
- Multimodal understanding (GPT-4, Gemini)
- Code generation (Copilot, CodeX)

### 13.4 남은 과제

**기술적**:
- 효율성: O(n²) 복잡도 해결
- 긴 context: 수만 토큰 이상 처리
- Reasoning: 논리적 추론 능력 향상
- Grounding: 실세계 지식과의 연결

**사회적**:
- 공정성 및 편향 해소
- 환경 영향 최소화
- AI 안전성 및 정렬
- 접근성 및 민주화

### 13.5 최종 평가

"Attention Is All You Need"는 **21세기 AI 연구의 가장 중요한 논문** 중 하나입니다.

**이유**:
1. **기술적 우수성**: 명확하고 효율적인 아키텍처
2. **재현 가능성**: 코드 공개 및 상세한 설명
3. **확장성**: 다양한 도메인과 규모로 확장
4. **영향력**: 수천 개의 후속 연구 촉발
5. **실용성**: 실제 제품에 광범위하게 적용

Transformer는 단순히 좋은 모델이 아니라, **AI 연구의 방향을 완전히 바꾼** 연구입니다. RNN 시대에서 Attention 시대로의 전환은 앞으로도 오랫동안 AI 역사의 중요한 전환점으로 기억될 것입니다.

**개인적 견해**:
Transformer의 성공은 "복잡함보다 단순함", "귀납적 편향보다 데이터와 계산", "특수성보다 범용성"이라는 현대 딥러닝의 철학을 잘 보여줍니다. 하지만 동시에, 효율성과 해석 가능성 같은 새로운 과제도 제시했습니다. 향후 10년은 Transformer의 한계를 극복하면서도 그 강점을 유지하는 방향으로 연구가 진행될 것으로 예상합니다.

---

## 14. 추가 자료

### 14.1 논문 및 코드

**원 논문**:
- [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- [NeurIPS 2017 Proceedings](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

**공식 코드**:
- [Tensor2Tensor GitHub](https://github.com/tensorflow/tensor2tensor)

**재구현**:
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 가장 유명한 주석 달린 구현
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 산업 표준 라이브러리
- [minGPT](https://github.com/karpathy/minGPT) - Andrej Karpathy의 교육용 구현

### 14.2 튜토리얼 및 강의

**블로그 포스트**:
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 가장 유명한 시각화 설명
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

**비디오 강의**:
- [Stanford CS224N: Transformers and Self-Attention](https://www.youtube.com/watch?v=5vcj8kSwBCY)
- [Yannic Kilcher's Paper Explanation](https://www.youtube.com/watch?v=iDulhoQ2pro)

**온라인 강좌**:
- [Hugging Face Course](https://huggingface.co/course/chapter1/1)
- [Fast.ai NLP Course](https://www.fast.ai/)

### 14.3 관련 논문

**선행 연구**:
- Bahdanau et al. (2015): "Neural Machine Translation by Jointly Learning to Align and Translate"
- Sutskever et al. (2014): "Sequence to Sequence Learning with Neural Networks"

**주요 후속 연구**:
- BERT (Devlin et al., 2018)
- GPT-2 (Radford et al., 2019)
- GPT-3 (Brown et al., 2020)
- Vision Transformer (Dosovitskiy et al., 2020)
- T5 (Raffel et al., 2019)

**Efficient Transformers**:
- Reformer (Kitaev et al., 2020)
- Linformer (Wang et al., 2020)
- Performer (Choromanski et al., 2020)
- FlashAttention (Dao et al., 2022)

### 14.4 도구 및 라이브러리

**구현 라이브러리**:
- [PyTorch](https://pytorch.org/) - torch.nn.Transformer
- [TensorFlow](https://www.tensorflow.org/) - tf.keras.layers.MultiHeadAttention
- [JAX](https://github.com/google/jax) - Flax

**사전학습 모델 Hub**:
- [Hugging Face Model Hub](https://huggingface.co/models)
- [TensorFlow Hub](https://tfhub.dev/)

**시각화 도구**:
- [BertViz](https://github.com/jessevig/bertviz) - Attention 시각화
- [exBERT](http://exbert.net/) - Interactive BERT exploration

### 14.5 데이터셋

**기계 번역**:
- [WMT Datasets](https://www.statmt.org/wmt23/)
- [IWSLT](https://iwslt.org/)

**일반 NLP**:
- [The Pile](https://pile.eleuther.ai/)
- [C4 (Colossal Clean Crawled Corpus)](https://www.tensorflow.org/datasets/catalog/c4)
- [BookCorpus](https://huggingface.co/datasets/bookcorpus)

---

## 참고문헌

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS 2017*.
2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR 2015*.
3. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL 2019*.
4. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Blog*.
5. Brown, T. B., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS 2020*.
6. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*.
7. Raffel, C., et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR 2020*.
8. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*.

---

**마지막 업데이트**: 2025-12-30
**작성자**: SuanLab Research Team
