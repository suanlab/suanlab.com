---
title: "[논문 리뷰] Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
date: "2026-01-01"
excerpt: "Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time a..."
category: "Paper Review"
tags: ["Paper Review","cs.LG","cs.AI","cs.LG"]
thumbnail: "/assets/images/blog/paper-review.jpg"
---

# [논문 리뷰] Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## TL;DR

최근 딥러닝에서 Transformer 아키텍처는 주의(attention) 메커니즘을 통해 복잡한 데이터 모델링을 가능케 하였으나, 긴 시퀀스 처리 시 계산 비용이 $O(n^2)$로 증가하는 단점이 있었습니다. 이를 해결하기 위해 선택적 상태 공간 모델(Selective State Space Models, SSMs)을 기반으로 한 Mamba 모델이 제안되었습니다. Mamba는 입력에 따라 동적으로 변하는 파라미터를 통해 정보를 선택적으로 처리하며, Transformer 대비 5배 높은 처리량과 선형적 시퀀스 확장성을 제공합니다. 다양한 모달리티에서 Transformer와 비교해 우수한 성능을 보이며, 특히 긴 시퀀스 처리에서 강점을 나타냅니다.

## 연구 배경 및 동기

Transformer 아키텍처는 주의 메커니즘을 통해 자연어 처리, 이미지 인식, 음성 처리 등 다양한 분야에서 혁신을 이끌어 왔습니다. 그러나, 이 모델은 시퀀스 길이에 따라 계산 복잡도가 $O(n^2)$로 증가하여 긴 시퀀스 처리 시 병목 현상이 발생합니다. 이러한 문제를 해결하기 위해 여러 하위 사분위 시간 복잡도(subquadratic-time complexity) 아키텍처가 제안되었지만, 이들은 언어와 같은 중요한 모달리티에서 Transformer만큼의 성능을 발휘하지 못했습니다. 이 연구는 기존 모델의 한계를 극복하고자 선택적 상태 공간 모델을 활용하여, 긴 시퀀스 처리 시 효율성과 성능을 동시에 개선하는 것을 목표로 합니다. 특히, 선택적 SSM이 입력에 따라 파라미터를 동적으로 조정하여 불필요한 정보를 필터링하고 중요한 정보를 기억하는 능력을 제공함으로써, Transformer의 계산 효율성 문제를 해결합니다.

## 관련 연구

1. **Transformer**: Vaswani et al. (2017)이 제안한 Transformer는 주의 메커니즘을 기반으로 하여 자연어 처리에서 획기적인 성능을 보였습니다. 그러나, 시퀀스 길이에 따라 계산 복잡도가 $O(n^2)$로 증가하는 단점이 있습니다.
   
2. **Reformer**: Kitaev et al. (2020)이 제안한 Reformer는 locality-sensitive hashing을 활용하여 attention 연산의 복잡도를 줄였습니다. 하지만, 여전히 긴 시퀀스 처리에 제한이 있습니다.

3. **Linformer**: Wang et al. (2020)은 low-rank factorization을 통해 attention 메커니즘의 효율성을 개선하였으나, 성능 저하가 발생할 수 있습니다.

4. **Longformer**: Beltagy et al. (2020)은 sparse attention을 사용하여 긴 시퀀스 처리의 효율성을 높였으나, 특정 도메인에 특화되어 있습니다.

5. **Performer**: Choromanski et al. (2020)은 kernel-based approximation을 통해 attention 연산을 가속화하였으나, 복잡한 데이터에 대한 성능이 제한적입니다.

| 연구명      | 기법                       | 장점                      | 단점                      |
|------------|---------------------------|---------------------------|---------------------------|
| Transformer| Attention                 | 강력한 성능               | $O(n^2)$ 복잡도           |
| Reformer   | Locality-sensitive hashing| 효율적인 attention       | 긴 시퀀스 제한            |
| Linformer  | Low-rank factorization    | 계산 효율성               | 성능 저하 가능            |
| Longformer | Sparse attention          | 긴 시퀀스 효율성          | 특정 도메인 특화          |
| Performer  | Kernel-based approximation| 연산 가속화               | 복잡한 데이터 성능 제한   |

## 핵심 기여

1. **선택적 상태 공간 모델 제안**: 입력에 따라 동적으로 파라미터를 조정하여 정보를 선택적으로 처리하는 선택적 SSM을 제안하였습니다. 이는 Transformer의 $O(n^2)$ 복잡도를 해결하면서도 긴 시퀀스에서 성능을 유지합니다.

2. **하드웨어 친화적 병렬 알고리즘 개발**: 효율적인 병렬 처리를 위한 알고리즘을 설계하여, Mamba 모델이 실제 하드웨어에서 빠르게 동작할 수 있도록 하였습니다.

3. **다양한 모달리티에서의 성능 검증**: 언어, 오디오, DNA 시퀀스 등 다양한 도메인에서 Mamba의 우수한 성능을 입증하였습니다.

4. **선택 메커니즘의 활용**: 선택 메커니즘을 통해 모델이 긴 시퀀스에서도 효율적으로 동작하도록 하였습니다. 이는 특히 긴 문맥 이해가 중요한 작업에서 성능을 향상시킵니다.

## 제안 방법론

Mamba 모델은 선택적 상태 공간 모델을 기반으로 하여, 입력에 따라 동적으로 변하는 파라미터를 통해 정보를 선택적으로 처리합니다. 이는 긴 시퀀스에서도 효율적으로 동작할 수 있는 구조를 제공합니다.

### 모델 아키텍처

Mamba는 attention 메커니즘 없이 선택적 상태 공간 모델을 사용하여 다양한 도메인에서 강력한 성능을 발휘합니다. 선택적 SSM은 입력에 따라 상태 전이 파라미터가 동적으로 변하는 것이 특징입니다.

### 핵심 수식

1. **상태 업데이트 방정식**:
   $$ h'(t) = A(x(t))h(t) + B(x(t))x(t) $$
   - $A(x(t))$, $B(x(t))$: 입력 $x(t)$에 의존하는 행렬
   - $h(t)$: 상태 벡터
   - $x(t)$: 입력 벡터

2. **출력 방정식**:
   $$ y(t) = C(x(t))h(t) + D(x(t))x(t) $$
   - $C(x(t))$, $D(x(t))$: 입력 $x(t)$에 의존하는 행렬
   - $y(t)$: 출력 벡터

3. **게이팅 메커니즘**:
   $$ g_t = \sigma(W_g x_t + b_g) $$
   - $W_g$: 가중치 행렬
   - $b_g$: 편향
   - $\sigma$: 시그모이드 함수

4. **상태 전이**:
   $$ h_{t+1} = g_t \odot f(h_t, x_t) + (1 - g_t) \odot h_t $$
   - $g_t$: 게이트 값
   - $f$: 상태 전이 함수
   - $\odot$: 요소별 곱셈

5. **학습률 스케줄링**:
   $$ \text{lr}(t) = \text{lr}_{\text{max}} \cdot \frac{1}{2} \left( 1 + \cos\left(\frac{t}{T} \pi\right) \right) $$
   - $\text{lr}(t)$: $t$번째 스텝에서의 학습률
   - $\text{lr}_{\text{max}}$: 최대 학습률
   - $T$: 총 학습 스텝 수

### Python/PyTorch 구현 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelectiveSSM, self).__init__()
        self.W_g = nn.Linear(input_dim, hidden_dim)
        self.b_g = nn.Parameter(torch.zeros(hidden_dim))
        self.A = nn.Linear(input_dim, hidden_dim)
        self.B = nn.Linear(input_dim, hidden_dim)
        self.C = nn.Linear(hidden_dim, input_dim)
        self.D = nn.Linear(input_dim, input_dim)

    def forward(self, x, h):
        # 게이팅 메커니즘
        g_t = torch.sigmoid(self.W_g(x) + self.b_g)
        
        # 상태 업데이트
        A_x = self.A(x)
        B_x = self.B(x)
        h_t_prime = A_x * h + B_x * x
        
        # 선택적 상태 전이
        h_t = g_t * h_t_prime + (1 - g_t) * h
        
        # 출력 계산
        C_h = self.C(h_t)
        D_x = self.D(x)
        y_t = C_h + D_x
        
        return y_t, h_t

# 모델 초기화
input_dim = 128
hidden_dim = 256
model = SelectiveSSM(input_dim, hidden_dim)

# 입력 및 초기 상태
x = torch.randn(10, input_dim)  # 10개의 시퀀스
h = torch.zeros(10, hidden_dim)  # 초기 상태

# 모델 실행
output, new_state = model(x, h)
```

## 실험 설정

Mamba 모델의 성능을 평가하기 위해 다양한 데이터셋과 평가 지표를 사용하였습니다. 주로 언어 모델링, DNA 시퀀스 분석, 오디오 처리에서 성능을 검증하였습니다.

### 데이터셋

- **언어 모델링**: WikiText-103 데이터셋
- **DNA 시퀀스 분석**: HG38 유전체 데이터셋
- **오디오 처리**: YouTubeMix 데이터셋

### 평가 지표

- **Perplexity**: 언어 모델링에서 사용
- **정확도**: DNA 시퀀스 분류에서 사용
- **신호 대 잡음비(SNR)**: 오디오 처리에서 사용

### 하이퍼파라미터

| 하이퍼파라미터   | 값         |
|----------------|-----------|
| 학습률          | 0.001     |
| 배치 크기       | 64        |
| 최대 시퀀스 길이 | 1024      |
| 옵티마이저      | AdamW     |
| 학습률 스케줄러 | 코사인 스케줄러 |

## 실험 결과 분석

### 주요 결과

Mamba 모델은 다양한 모달리티에서 Transformer 대비 우수한 성능을 보였습니다.

| 모델         | 언어 모델링 (Perplexity) | DNA 모델링 (정확도 %) | 오디오 모델링 (SNR) |
|-------------|--------------------------|----------------------|--------------------|
| Transformer | 25.5                     | 88.7                 | 15.2               |
| Mamba       | 24.0                     | 92.5                 | 16.8               |

### 성능 향상률

- **언어 모델링**: 5.9% 향상
- **DNA 모델링**: 4.3% 향상
- **오디오 모델링**: 10.5% 향상

### Ablation Study

선택적 상태 공간 모델의 각 구성 요소가 성능에 미치는 영향을 분석하였습니다. 선택적 파라미터 업데이트가 모델 성능에 가장 큰 기여를 하였습니다.

## 비판적 평가

### 강점

1. **효율성**: Mamba는 Transformer 대비 5배 높은 처리량을 제공하며, 긴 시퀀스 처리에서 우수한 성능을 보입니다.
2. **다양한 모달리티에서의 성능**: 언어, 오디오, DNA 시퀀스 등 다양한 도메인에서 뛰어난 성능을 발휘합니다.
3. **선택 메커니즘의 활용**: 긴 시퀀스에서도 중요한 정보를 선택적으로 처리하여 효율성을 높입니다.

### 한계점

1. **복잡한 구현**: 선택적 상태 공간 모델의 구현이 복잡할 수 있습니다.
2. **모델 해석 가능성**: 선택적 메커니즘의 해석 가능성을 높이기 위한 추가 연구가 필요합니다.

### 재현성 평가

논문에서 제시한 알고리즘과 실험 설정을 바탕으로, 재현성은 높은 것으로 평가됩니다. 다만, 선택적 메커니즘의 구현이 복잡하여 초기 구현 시 어려움이 있을 수 있습니다.

## 향후 연구 방향

1. **모델 해석 가능성 향상**: 선택적 메커니즘의 작동 원리를 더 명확히 이해하고 해석할 수 있는 방법론 개발이 필요합니다.
2. **다양한 응용 분야 확장**: 비전, 로보틱스 등 새로운 분야에 Mamba 모델의 적용 가능성을 탐색할 필요가 있습니다.
3. **하드웨어 최적화**: Mamba의 하드웨어 효율성을 더욱 높이기 위한 최적화 연구가 필요합니다.

## 실무 적용 가이드

- **모델 구현 시 고려사항**: 선택적 상태 공간 모델의 파라미터 업데이트가 복잡할 수 있으므로, 초기 구현 시 주의가 필요합니다.
- **팁**: PyTorch의 자동 미분 기능을 활용하여 파라미터 업데이트를 효율적으로 구현할 수 있습니다.

## 결론

Mamba 모델은 선택적 상태 공간 모델을 통해 긴 시퀀스 처리의 효율성을 획기적으로 개선하였으며, 다양한 모달리티에서 Transformer 대비 우수한 성능을 발휘합니다. 이는 Transformer의 계산 효율성 문제를 해결하고, 긴 시퀀스 데이터 처리에서 새로운 가능성을 열었습니다.

## 참고 자료

- [논문 링크](https://arxiv.org/abs/2312.00752)
- [코드 저장소](https://github.com/mamba-research/mamba)
- 관련 자료: Transformer, Reformer, Linformer, Longformer, Performer 논문
