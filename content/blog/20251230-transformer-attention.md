---
title: "트랜스포머 Attention 메커니즘의 이해"
date: "2025-12-30"
excerpt: "최근 몇 년간 자연어 처리(Natural Language Processing, NLP) 분야에서는 혁신적인 변화가 일어났습니다. 그 중심에는 단연 트랜스포머(Transformer) 모델이 자리 잡고 있습니다. 트랜스포머는 다양한 NLP 작업에서 뛰어난 성능을 보이며, 언어 모델, 번역, 요약, 질의응답 등 여러 응용 분야에서 사용됩니다. 이러한 트랜스포머의..."
category: "Deep Learning"
tags: ["Transformer", "Attention", "Deep Learning", "NLP", "Self-Attention"]
thumbnail: "/assets/images/blog/20251230-transformer-attention.jpg"
---

# 트랜스포머 Attention 메커니즘의 이해

## 도입부

최근 몇 년간 자연어 처리(Natural Language Processing, NLP) 분야에서는 혁신적인 변화가 일어났습니다. 그 중심에는 단연 트랜스포머(Transformer) 모델이 자리 잡고 있습니다. 트랜스포머는 다양한 NLP 작업에서 뛰어난 성능을 보이며, 언어 모델, 번역, 요약, 질의응답 등 여러 응용 분야에서 사용됩니다. 이러한 트랜스포머의 핵심에는 바로 '어텐션(Attention)' 메커니즘이 있습니다. 어텐션 메커니즘은 입력 데이터에서 어떤 부분에 집중해야 하는지를 알려주는 역할을 하며, 모델의 성능을 크게 향상시킵니다.

본 블로그 포스트에서는 트랜스포머 모델의 중심이 되는 어텐션 메커니즘을 깊이 있게 탐구해 보도록 하겠습니다. 어텐션 메커니즘이 왜 중요한지, 그리고 그것이 어떻게 작동하는지를 이해하는 것은 트랜스포머 모델을 효과적으로 활용하는 데 필수적입니다.

## 본문

### 1. Attention 메커니즘의 역사

#### 1.1. RNN과 시퀀스 투 시퀀스 모델의 한계

딥러닝 기반 NLP가 본격적으로 발전하기 시작한 2010년대 초반, 시퀀스 데이터를 처리하는 데는 주로 순환 신경망(Recurrent Neural Network, RNN)과 그 변형인 LSTM(Long Short-Term Memory), GRU(Gated Recurrent Unit)가 사용되었습니다.

특히 기계 번역과 같은 시퀀스-투-시퀀스(Sequence-to-Sequence) 작업에서는 인코더-디코더 구조가 널리 사용되었습니다. 인코더는 입력 시퀀스를 고정 길이의 컨텍스트 벡터로 압축하고, 디코더는 이 벡터를 사용하여 출력 시퀀스를 생성합니다.

하지만 이 구조에는 근본적인 문제가 있었습니다:

1. **정보 병목(Information Bottleneck)**: 입력 시퀀스의 모든 정보를 고정된 크기의 벡터 하나로 압축해야 하므로, 긴 시퀀스에서는 정보 손실이 발생합니다.
2. **장거리 의존성(Long-range Dependency) 문제**: 시퀀스가 길어질수록 초반 정보가 희석되어 학습이 어려워집니다.
3. **순차 처리의 한계**: RNN은 본질적으로 순차적으로 처리되므로 병렬화가 어렵고 학습 시간이 오래 걸립니다.

#### 1.2. Bahdanau Attention (2014)

이러한 문제를 해결하기 위해 2014년 Bahdanau et al.은 "Neural Machine Translation by Jointly Learning to Align and Translate" 논문에서 최초의 어텐션 메커니즘을 제안했습니다.

Bahdanau Attention의 핵심 아이디어는 다음과 같습니다:

- 디코더가 출력을 생성할 때, 인코더의 모든 hidden state를 참조할 수 있도록 함
- 각 디코딩 스텝마다 입력 시퀀스의 어느 부분에 집중할지를 동적으로 결정
- 컨텍스트 벡터를 고정된 것이 아니라 각 스텝마다 다르게 계산

Bahdanau Attention의 수식은 다음과 같습니다:

$$
e_{ij} = a(s_{i-1}, h_j)
$$

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$

여기서:
- $s_{i-1}$: 디코더의 이전 hidden state
- $h_j$: 인코더의 j번째 hidden state
- $a$: alignment 함수 (주로 작은 feedforward network)
- $\alpha_{ij}$: attention weight
- $c_i$: i번째 스텝의 컨텍스트 벡터

#### 1.3. Luong Attention (2015)

2015년 Luong et al.은 Bahdanau Attention을 개선한 여러 변형을 제안했습니다. 주요 차이점은:

1. **스코어 계산 방식의 다양화**:
   - Dot: $\text{score}(h_t, \bar{h}_s) = h_t^T \bar{h}_s$
   - General: $\text{score}(h_t, \bar{h}_s) = h_t^T W_a \bar{h}_s$
   - Concat: $\text{score}(h_t, \bar{h}_s) = v_a^T \tanh(W_a[h_t; \bar{h}_s])$

2. **Global vs Local Attention**: 모든 입력 위치를 고려하는 것(global)과 일부 윈도우만 고려하는 것(local) 중 선택 가능

#### 1.4. Self-Attention의 등장 (2017)

2017년 Vaswani et al.의 "Attention is All You Need" 논문은 RNN을 완전히 제거하고 오직 어텐션 메커니즘만으로 구성된 트랜스포머 모델을 제안했습니다. 이는 딥러닝 역사상 가장 영향력 있는 발견 중 하나로 평가받고 있습니다.

트랜스포머는 다음과 같은 혁신을 가져왔습니다:

1. **완전한 병렬 처리**: RNN의 순차적 처리를 제거하여 GPU를 효율적으로 활용
2. **Self-Attention**: 시퀀스 내 모든 위치 간의 관계를 직접 모델링
3. **Scaled Dot-Product Attention**: 간단하면서도 효과적인 어텐션 메커니즘
4. **Multi-Head Attention**: 여러 표현 부분공간에서 정보 학습

### 2. Self-Attention vs Cross-Attention

트랜스포머에서는 두 가지 유형의 어텐션이 사용됩니다: Self-Attention과 Cross-Attention입니다. 이 둘의 차이를 명확히 이해하는 것이 중요합니다.

#### 2.1. Self-Attention (자기 주의)

Self-Attention은 하나의 시퀀스 내에서 각 위치가 같은 시퀀스의 다른 모든 위치와의 관계를 학습합니다.

**특징:**
- Query, Key, Value가 모두 같은 입력 시퀀스에서 생성됨
- 시퀀스 내 요소 간의 의존성을 포착
- 인코더와 디코더 모두에서 사용됨

**예시:**
문장 "The animal didn't cross the street because it was too tired"에서 "it"이 "animal"을 가리키는지 "street"를 가리키는지를 파악하는 데 Self-Attention이 사용됩니다.

**코드 예제:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) - optional
        """
        # 같은 입력에서 Q, K, V 생성
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)

        # Attention scores 계산
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        scores = scores / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        # 마스킹 (옵션)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax로 attention weights 계산
        attention_weights = F.softmax(scores, dim=-1)

        # 가중 합 계산
        output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, d_model)

        return output, attention_weights

# 사용 예제
batch_size, seq_len, d_model = 2, 5, 64
x = torch.randn(batch_size, seq_len, d_model)
self_attn = SelfAttention(d_model)
output, weights = self_attn(x)
print(f"Output shape: {output.shape}")  # (2, 5, 64)
print(f"Attention weights shape: {weights.shape}")  # (2, 5, 5)
```

#### 2.2. Cross-Attention (교차 주의)

Cross-Attention은 두 개의 서로 다른 시퀀스 간의 관계를 학습합니다. 주로 인코더-디코더 구조에서 사용됩니다.

**특징:**
- Query는 한 시퀀스(디코더)에서, Key와 Value는 다른 시퀀스(인코더)에서 생성됨
- 두 시퀀스 간의 정렬(alignment)을 학습
- 디코더에서만 사용됨

**예시:**
기계 번역에서 영어 문장 "I love you"를 한국어 "나는 너를 사랑해"로 번역할 때, 각 한국어 단어가 어떤 영어 단어에 집중해야 하는지를 Cross-Attention이 학습합니다.

**코드 예제:**

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, query_seq, key_value_seq, mask=None):
        """
        Args:
            query_seq: (batch_size, tgt_len, d_model) - 디코더 입력
            key_value_seq: (batch_size, src_len, d_model) - 인코더 출력
            mask: (batch_size, tgt_len, src_len) - optional
        """
        # Query는 디코더에서, Key와 Value는 인코더에서
        Q = self.W_q(query_seq)  # (batch_size, tgt_len, d_model)
        K = self.W_k(key_value_seq)  # (batch_size, src_len, d_model)
        V = self.W_v(key_value_seq)  # (batch_size, src_len, d_model)

        # Attention scores 계산
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, tgt_len, src_len)
        scores = scores / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        # 마스킹
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # 가중 합
        output = torch.matmul(attention_weights, V)  # (batch_size, tgt_len, d_model)

        return output, attention_weights

# 사용 예제
batch_size, src_len, tgt_len, d_model = 2, 10, 7, 64
encoder_output = torch.randn(batch_size, src_len, d_model)
decoder_input = torch.randn(batch_size, tgt_len, d_model)
cross_attn = CrossAttention(d_model)
output, weights = cross_attn(decoder_input, encoder_output)
print(f"Output shape: {output.shape}")  # (2, 7, 64)
print(f"Attention weights shape: {weights.shape}")  # (2, 7, 10)
```

#### 2.3. Self-Attention vs Cross-Attention 비교

| 특성 | Self-Attention | Cross-Attention |
|------|----------------|-----------------|
| Query 출처 | 입력 시퀀스 자신 | 디코더 시퀀스 |
| Key/Value 출처 | 입력 시퀀스 자신 | 인코더 시퀀스 |
| 용도 | 시퀀스 내부 의존성 | 시퀀스 간 정렬 |
| 사용 위치 | 인코더, 디코더 모두 | 디코더만 |
| Attention shape | (seq_len, seq_len) | (tgt_len, src_len) |

### 3. Scaled Dot-Product Attention의 상세 원리

#### 3.1. 수식과 의미

Scaled Dot-Product Attention은 트랜스포머의 기본 빌딩 블록입니다:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

각 구성요소의 의미:
- $Q$ (Query): "무엇을 찾고 있는가?" - (batch_size, seq_len, d_k)
- $K$ (Key): "나는 무엇을 제공하는가?" - (batch_size, seq_len, d_k)
- $V$ (Value): "실제 정보 내용" - (batch_size, seq_len, d_v)
- $d_k$: Key 벡터의 차원

#### 3.2. Scaling Factor가 필요한 이유

스케일링 팩터 $\frac{1}{\sqrt{d_k}}$가 없다면 어떤 일이 벌어질까요?

$d_k$가 크면 내적 $QK^T$의 값이 매우 커지게 됩니다. 이는 다음과 같은 문제를 야기합니다:

1. **Softmax Saturation**: 매우 큰 값이 softmax에 들어가면 기울기가 매우 작아져 학습이 어려워집니다.
2. **수치 불안정성**: 매우 큰 값은 부동소수점 연산에서 오버플로우를 일으킬 수 있습니다.

**수학적 설명:**

$Q$와 $K$의 각 요소가 평균 0, 분산 1인 독립 랜덤 변수라고 가정하면, $d_k$차원 벡터의 내적은 평균 0, 분산 $d_k$를 가집니다. $\sqrt{d_k}$로 나누면 분산을 1로 정규화할 수 있습니다.

**실험적 검증:**

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compare_scaling():
    d_k = 64
    seq_len = 10

    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)

    # Without scaling
    scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
    weights_unscaled = F.softmax(scores_unscaled, dim=-1)

    # With scaling
    scores_scaled = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    weights_scaled = F.softmax(scores_scaled, dim=-1)

    print(f"Unscaled scores - Mean: {scores_unscaled.mean():.4f}, Std: {scores_unscaled.std():.4f}")
    print(f"Scaled scores - Mean: {scores_scaled.mean():.4f}, Std: {scores_scaled.std():.4f}")

    print(f"\nUnscaled weights - Max: {weights_unscaled.max():.4f}, Min: {weights_unscaled.min():.6f}")
    print(f"Scaled weights - Max: {weights_scaled.max():.4f}, Min: {weights_scaled.min():.6f}")

    # Softmax의 기울기 비교
    print(f"\nUnscaled gradient magnitude: {(weights_unscaled * (1 - weights_unscaled)).mean():.6f}")
    print(f"Scaled gradient magnitude: {(weights_scaled * (1 - weights_scaled)).mean():.6f}")

compare_scaling()
```

#### 3.3. Masking

Attention에서는 특정 위치를 참조하지 못하도록 마스킹이 필요합니다:

1. **Padding Mask**: 패딩된 위치를 무시
2. **Look-ahead Mask**: 디코더에서 미래 토큰을 보지 못하도록 함

```python
def create_padding_mask(seq, pad_token=0):
    """
    패딩 토큰 위치를 마스킹
    Args:
        seq: (batch_size, seq_len)
    Returns:
        mask: (batch_size, 1, 1, seq_len)
    """
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask

def create_look_ahead_mask(size):
    """
    디코더에서 미래 토큰을 보지 못하도록 마스킹
    Args:
        size: sequence length
    Returns:
        mask: (size, size)
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask  # 1은 보이고, 0은 마스킹

# 예제
seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
padding_mask = create_padding_mask(seq)
print("Padding mask:")
print(padding_mask.squeeze())

look_ahead = create_look_ahead_mask(5)
print("\nLook-ahead mask:")
print(look_ahead)
```

### 4. Multi-Head Attention 상세 설명

#### 4.1. Multi-Head Attention의 동기

왜 하나의 어텐션이 아니라 여러 개의 헤드를 사용할까요?

1. **다양한 표현 부분공간**: 각 헤드가 서로 다른 측면의 관계를 학습할 수 있습니다.
   - 어떤 헤드는 구문적 관계(syntactic)를 학습
   - 어떤 헤드는 의미적 관계(semantic)를 학습
   - 어떤 헤드는 장거리 의존성을 학습

2. **앙상블 효과**: 여러 헤드의 출력을 결합하면 더 풍부한 표현이 가능합니다.

3. **모델 용량 증가**: 파라미터를 늘리지 않고도 표현력을 높일 수 있습니다.

#### 4.2. Multi-Head Attention 수식

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

여기서 각 헤드는:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

파라미터 행렬:
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

일반적으로 $d_k = d_v = d_{model} / h$로 설정하여, 전체 계산 비용이 단일 헤드 어텐션과 유사하도록 합니다.

#### 4.3. Multi-Head Attention 구현

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 각 헤드에 대한 선형 변환
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 출력 선형 변환
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. 선형 변환
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. 헤드로 분할
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, d_k)

        # 4. 헤드 결합
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # 5. 최종 선형 변환
        output = self.W_o(attention_output)

        return output, attention_weights

# 사용 예제
d_model = 512
num_heads = 8
batch_size, seq_len = 2, 10

x = torch.randn(batch_size, seq_len, d_model)
mha = MultiHeadAttention(d_model, num_heads)
output, weights = mha(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

#### 4.4. Multi-Head Attention 시각화

각 헤드가 무엇을 학습하는지 시각화해보겠습니다:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention_heads(attention_weights, tokens, num_heads=4):
    """
    Multi-head attention weights 시각화
    Args:
        attention_weights: (batch_size, num_heads, seq_len, seq_len)
        tokens: 토큰 리스트
        num_heads: 시각화할 헤드 수
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(min(num_heads, 4)):
        attn = attention_weights[0, i].detach().cpu().numpy()
        sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
                   cmap='YlOrRd', ax=axes[i], cbar=True)
        axes[i].set_title(f'Head {i+1}', fontsize=14)
        axes[i].set_xlabel('Key', fontsize=12)
        axes[i].set_ylabel('Query', fontsize=12)

    plt.tight_layout()
    plt.savefig('multi_head_attention.png', dpi=150)
    plt.close()

# 예제 (실제로는 학습된 모델의 weights를 사용)
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
seq_len = len(tokens)
# 임의의 attention weights (예시)
example_weights = torch.softmax(torch.randn(1, 4, seq_len, seq_len), dim=-1)
visualize_attention_heads(example_weights, tokens)
```

### 5. Positional Encoding 상세 설명

#### 5.1. Positional Encoding의 필요성

트랜스포머는 RNN과 달리 순차적으로 처리하지 않습니다. 즉, "I love you"와 "you love I"를 구분할 수 없습니다. 따라서 위치 정보를 명시적으로 주입해야 합니다.

#### 5.2. 사인/코사인 Positional Encoding

원논문에서 제안한 방법은 사인과 코사인 함수를 사용합니다:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

여기서:
- $pos$: 토큰의 위치 (0, 1, 2, ...)
- $i$: 차원 인덱스 (0, 1, 2, ..., d_model/2 - 1)
- $d_{model}$: 모델 차원

**왜 사인/코사인인가?**

1. **주기성**: 각 차원마다 다른 주파수를 가지므로 다양한 스케일의 위치 관계를 포착
2. **외삽 가능**: 학습 시 보지 못한 긴 시퀀스도 처리 가능
3. **상대 위치 학습**: $PE_{pos+k}$는 $PE_{pos}$의 선형 함수로 표현 가능

#### 5.3. Positional Encoding 구현

```python
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional encoding 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 시각화
def visualize_positional_encoding():
    d_model = 128
    max_len = 100

    pe = PositionalEncoding(d_model, max_len)

    # Positional encoding 추출
    pos_enc = pe.pe.squeeze().numpy()

    plt.figure(figsize=(15, 6))
    plt.pcolormesh(pos_enc.T, cmap='RdBu')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.colorbar()
    plt.title('Positional Encoding Pattern')
    plt.savefig('positional_encoding.png', dpi=150)
    plt.close()

    # 특정 차원들의 패턴
    plt.figure(figsize=(15, 6))
    for i in [0, 1, 4, 8, 16, 32, 64]:
        plt.plot(pos_enc[:, i], label=f'dim {i}')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Positional Encoding for Different Dimensions')
    plt.savefig('positional_encoding_dims.png', dpi=150)
    plt.close()

visualize_positional_encoding()
```

#### 5.4. 학습 가능한 Positional Embedding

사인/코사인 대신 학습 가능한 임베딩을 사용할 수도 있습니다:

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_enc = self.pos_embedding(positions)
        x = x + pos_enc
        return self.dropout(x)
```

**비교:**

| 특성 | Sinusoidal | Learned |
|------|------------|---------|
| 파라미터 | 없음 | max_len × d_model |
| 외삽 | 가능 | 제한적 |
| 성능 | 비슷 | 비슷 |
| 사용 예 | 원본 Transformer | BERT, GPT |

### 6. Transformer 인코더/디코더 구조

#### 6.1. 전체 아키텍처 개요

트랜스포머는 인코더와 디코더 스택으로 구성됩니다:

**인코더:**
- N개의 동일한 레이어 (원논문: N=6)
- 각 레이어는 두 개의 서브레이어:
  1. Multi-Head Self-Attention
  2. Position-wise Feed-Forward Network
- 각 서브레이어 주변에 Residual Connection + Layer Normalization

**디코더:**
- N개의 동일한 레이어 (원논문: N=6)
- 각 레이어는 세 개의 서브레이어:
  1. Masked Multi-Head Self-Attention
  2. Multi-Head Cross-Attention (인코더 출력 참조)
  3. Position-wise Feed-Forward Network
- 각 서브레이어 주변에 Residual Connection + Layer Normalization

#### 6.2. 인코더 레이어 구현

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + Residual + Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # FFN + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

#### 6.3. 디코더 레이어 구현

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # Masked Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Multi-Head Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked Self-Attention + Residual + Norm
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Cross-Attention + Residual + Norm
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # FFN + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))

        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
```

#### 6.4. 완전한 Transformer 모델

```python
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 max_len=5000,
                 dropout=0.1):
        super(Transformer, self).__init__()

        # 임베딩 레이어
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 인코더와 디코더
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)

        # 출력 레이어
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        # 임베딩 + Positional Encoding
        src_emb = self.encoder_embedding(src) * math.sqrt(self.encoder_embedding.embedding_dim)
        src_emb = self.pos_encoding(src_emb)

        # 인코더 통과
        encoder_output = self.encoder(src_emb, src_mask)
        return encoder_output

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # 임베딩 + Positional Encoding
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.decoder_embedding.embedding_dim)
        tgt_emb = self.pos_encoding(tgt_emb)

        # 디코더 통과
        decoder_output = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
        return decoder_output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 인코딩
        encoder_output = self.encode(src, src_mask)

        # 디코딩
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # 출력 로짓
        output = self.output_layer(decoder_output)
        return output

# 모델 생성 예제
src_vocab_size = 10000
tgt_vocab_size = 10000

model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    dropout=0.1
)

# 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

### 7. PyTorch로 Transformer 훈련하기

#### 7.1. 데이터 준비

```python
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = self.src_vocab(self.src_texts[idx])
        tgt = self.tgt_vocab(self.tgt_texts[idx])
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    # 패딩
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0, batch_first=True)

    return src_batch, tgt_batch

# 예제 데이터 로더
# dataset = TranslationDataset(src_texts, tgt_texts, src_vocab, tgt_vocab)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
```

#### 7.2. 학습 루프

```python
import torch.optim as optim

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        # tgt_input: <start> + tgt[:-1]
        # tgt_output: tgt[1:] + <end>
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # 마스크 생성
        src_mask = create_padding_mask(src)
        tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)

        # Loss 계산
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = create_padding_mask(src)
            tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)

            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            total_loss += loss.item()

    return total_loss / len(dataloader)

# 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # 패딩 무시
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

num_epochs = 20
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
    val_loss = evaluate(model, val_dataloader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print("-" * 50)
```

#### 7.3. Learning Rate Scheduling

원논문에서는 warmup이 있는 특별한 learning rate schedule을 사용합니다:

$$
lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup\_steps^{-1.5})
$$

```python
class NoamOpt:
    def __init__(self, d_model, warmup_steps, optimizer):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()

    def get_lr(self):
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

# 사용
base_optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
optimizer = NoamOpt(d_model=512, warmup_steps=4000, optimizer=base_optimizer)
```

#### 7.4. Label Smoothing

과적합을 방지하기 위해 label smoothing을 사용할 수 있습니다:

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: (batch_size * seq_len, num_classes)
            target: (batch_size * seq_len,)
        """
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.ignore_index] = 0
            mask = torch.nonzero(target == self.ignore_index)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
```

### 8. 추론 (Inference)

#### 8.1. Greedy Decoding

```python
def greedy_decode(model, src, src_mask, max_len, start_token, end_token, device):
    """
    가장 확률이 높은 토큰을 선택하는 greedy decoding
    """
    model.eval()

    # 인코더 통과
    encoder_output = model.encode(src, src_mask)

    # 디코더 입력 초기화
    tgt = torch.ones(src.size(0), 1).fill_(start_token).long().to(device)

    for i in range(max_len - 1):
        tgt_mask = create_look_ahead_mask(tgt.size(1)).to(device)

        # 디코더 통과
        output = model.decode(tgt, encoder_output, src_mask, tgt_mask)

        # 다음 토큰 예측
        prob = model.output_layer(output[:, -1])
        next_token = prob.argmax(dim=-1).unsqueeze(1)

        # 종료 토큰이면 중단
        if next_token.item() == end_token:
            break

        # 다음 토큰 추가
        tgt = torch.cat([tgt, next_token], dim=1)

    return tgt

# 사용 예제
# src = torch.tensor([[1, 2, 3, 4, 5]]).to(device)
# src_mask = create_padding_mask(src)
# output = greedy_decode(model, src, src_mask, max_len=50, start_token=1, end_token=2, device=device)
```

#### 8.2. Beam Search

더 나은 결과를 위해 beam search를 사용할 수 있습니다:

```python
def beam_search(model, src, src_mask, max_len, start_token, end_token, beam_size, device):
    """
    Beam search decoding
    """
    model.eval()

    # 인코더 통과
    encoder_output = model.encode(src, src_mask)

    # 빔 초기화
    beams = [(torch.ones(1, 1).fill_(start_token).long().to(device), 0)]  # (sequence, score)

    for i in range(max_len - 1):
        all_candidates = []

        for seq, score in beams:
            if seq[0, -1].item() == end_token:
                all_candidates.append((seq, score))
                continue

            tgt_mask = create_look_ahead_mask(seq.size(1)).to(device)
            output = model.decode(seq, encoder_output, src_mask, tgt_mask)
            prob = F.log_softmax(model.output_layer(output[:, -1]), dim=-1)

            # Top-k 후보
            topk_prob, topk_idx = torch.topk(prob, beam_size)

            for k in range(beam_size):
                next_token = topk_idx[0, k].unsqueeze(0).unsqueeze(0)
                next_seq = torch.cat([seq, next_token], dim=1)
                next_score = score + topk_prob[0, k].item()
                all_candidates.append((next_seq, next_score))

        # 상위 beam_size개 선택
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        # 모든 빔이 종료되었으면 중단
        if all(seq[0, -1].item() == end_token for seq, _ in beams):
            break

    # 가장 좋은 시퀀스 반환
    best_seq, best_score = beams[0]
    return best_seq
```

### 9. 실제 활용 예시

#### 9.1. 기계 번역 (Machine Translation)

```python
class TranslationModel:
    def __init__(self, model, src_tokenizer, tgt_tokenizer, device):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device

    def translate(self, text, max_len=100, beam_size=5):
        # 소스 텍스트 토크나이징
        src_tokens = self.src_tokenizer.encode(text)
        src = torch.tensor([src_tokens]).to(self.device)
        src_mask = create_padding_mask(src)

        # Beam search로 번역
        output = beam_search(
            self.model, src, src_mask,
            max_len=max_len,
            start_token=self.tgt_tokenizer.start_token,
            end_token=self.tgt_tokenizer.end_token,
            beam_size=beam_size,
            device=self.device
        )

        # 토큰을 텍스트로 변환
        translated = self.tgt_tokenizer.decode(output[0].tolist())
        return translated

# 사용 예제
# translator = TranslationModel(model, src_tokenizer, tgt_tokenizer, device)
# result = translator.translate("Hello, how are you?")
# print(f"Translation: {result}")
```

#### 9.2. 텍스트 요약 (Text Summarization)

```python
class SummarizationModel:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def summarize(self, text, max_len=150):
        # 입력 텍스트 인코딩
        tokens = self.tokenizer.encode(text)
        src = torch.tensor([tokens]).to(self.device)
        src_mask = create_padding_mask(src)

        # 요약 생성
        summary = greedy_decode(
            self.model, src, src_mask,
            max_len=max_len,
            start_token=self.tokenizer.start_token,
            end_token=self.tokenizer.end_token,
            device=self.device
        )

        # 디코딩
        summary_text = self.tokenizer.decode(summary[0].tolist())
        return summary_text

# 사용 예제
# summarizer = SummarizationModel(model, tokenizer, device)
# long_text = "..." # 긴 문서
# summary = summarizer.summarize(long_text)
# print(f"Summary: {summary}")
```

#### 9.3. 질의응답 시스템

```python
class QASystem:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def answer(self, context, question, max_len=50):
        # 컨텍스트와 질문 결합
        input_text = f"{context} [SEP] {question}"
        tokens = self.tokenizer.encode(input_text)
        src = torch.tensor([tokens]).to(self.device)
        src_mask = create_padding_mask(src)

        # 답변 생성
        answer = greedy_decode(
            self.model, src, src_mask,
            max_len=max_len,
            start_token=self.tokenizer.start_token,
            end_token=self.tokenizer.end_token,
            device=self.device
        )

        answer_text = self.tokenizer.decode(answer[0].tolist())
        return answer_text

# 사용 예제
# qa_system = QASystem(model, tokenizer, device)
# context = "The Transformer is a deep learning model introduced in 2017."
# question = "When was the Transformer introduced?"
# answer = qa_system.answer(context, question)
# print(f"Answer: {answer}")
```

### 10. 성능 최적화 팁

#### 10.1. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_epoch_mixed_precision(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_padding_mask(src)
        tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

#### 10.2. Gradient Accumulation

배치 크기를 늘리지 않고도 효과적으로 학습할 수 있습니다:

```python
def train_with_gradient_accumulation(model, dataloader, optimizer, criterion, device,
                                     accumulation_steps=4):
    model.train()
    total_loss = 0

    optimizer.zero_grad()
    for i, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_padding_mask(src)
        tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)

        output = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

        # Normalize loss
        loss = loss / accumulation_steps
        loss.backward()

        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)
```

#### 10.3. 모델 체크포인팅

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

# 사용
# save_checkpoint(model, optimizer, epoch, loss, 'checkpoint.pt')
# epoch, loss = load_checkpoint(model, optimizer, 'checkpoint.pt')
```

### 11. Transformer의 변형과 발전

#### 11.1. BERT (Bidirectional Encoder Representations from Transformers)

- 인코더만 사용
- 양방향 컨텍스트 학습
- Masked Language Modeling (MLM) 사전학습

#### 11.2. GPT (Generative Pre-trained Transformer)

- 디코더만 사용
- 단방향 (왼쪽에서 오른쪽) 언어 모델
- Next Token Prediction 사전학습

#### 11.3. T5 (Text-to-Text Transfer Transformer)

- 모든 NLP 작업을 텍스트-투-텍스트로 통일
- 완전한 인코더-디코더 구조

#### 11.4. Vision Transformer (ViT)

- 이미지를 패치로 나누어 시퀀스로 처리
- 컴퓨터 비전에 Transformer 적용

## 결론

트랜스포머의 어텐션 메커니즘은 NLP 분야의 혁신을 이끌었습니다. 본 포스트에서는 그 핵심 원리와 작동 과정을 깊이 있게 살펴보았습니다.

**주요 내용 요약:**

1. **Attention의 역사**: Bahdanau Attention부터 Self-Attention까지의 발전 과정
2. **Self-Attention vs Cross-Attention**: 두 메커니즘의 차이와 사용 사례
3. **Scaled Dot-Product Attention**: 수식의 의미와 스케일링의 중요성
4. **Multi-Head Attention**: 여러 표현 부분공간에서의 학습
5. **Positional Encoding**: 위치 정보를 주입하는 방법
6. **인코더/디코더 구조**: 완전한 Transformer 아키텍처
7. **PyTorch 구현**: 실제 코드로 Transformer 구현하기
8. **실제 활용**: 번역, 요약, 질의응답 시스템

어텐션 메커니즘을 이해하는 것은 트랜스포머 모델을 효과적으로 활용하기 위한 첫 걸음입니다. 이를 바탕으로 BERT, GPT, T5 등 다양한 변형 모델들을 이해하고 활용할 수 있습니다.

### 추가 학습 자료

**논문:**
- "Attention is All You Need" (Vaswani et al., 2017): [링크](https://arxiv.org/abs/1706.03762)
- "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2014): [링크](https://arxiv.org/abs/1409.0473)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018): [링크](https://arxiv.org/abs/1810.04805)
- "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018) - GPT

**코드 및 튜토리얼:**
- PyTorch 공식 튜토리얼: [링크](https://pytorch.org/tutorials/)
- The Annotated Transformer: [링크](http://nlp.seas.harvard.edu/annotated-transformer/)
- Hugging Face Transformers 라이브러리: [링크](https://huggingface.co/transformers/)

**온라인 강의:**
- Stanford CS224N: Natural Language Processing with Deep Learning
- Fast.ai NLP Course
- DeepLearning.AI Natural Language Processing Specialization

어텐션 메커니즘을 통해 여러분의 프로젝트에 더 나은 성능을 가져올 수 있기를 바랍니다. Happy coding!
