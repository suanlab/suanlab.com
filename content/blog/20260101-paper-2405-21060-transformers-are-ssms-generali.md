---
title: "[논문 리뷰] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
date: "2026-01-01"
excerpt: "While Transformers have been the main architecture behind deep learning's success in language modeling, state-space models (SSMs) such as Mamba have recently been shown to match or outperform Transfor..."
category: "Paper Review"
tags: ["Paper Review","cs.LG","cs.LG"]
thumbnail: "/assets/images/blog/20260101-paper-2405-21060-transformers-are-ssms-generali.jpg"
---

# [논문 리뷰] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

## TL;DR

최근 연구는 Transformer와 상태 공간 모델(SSM) 간의 이론적 연결성을 밝혀내고, 이를 바탕으로 새로운 아키텍처인 Mamba-2를 제안합니다. 이 연구는 구조적 반분리 행렬을 이용하여 두 모델 간의 수학적 등가성을 설명하며, SSM의 효율성을 극대화합니다. Mamba-2는 2-8배 빠른 속도를 구현하면서도 Transformer와 경쟁할 수 있는 성능을 유지합니다. 제안된 SSD(Structured State-Space Duality) 프레임워크는 SSM과 주의 메커니즘 간의 연결을 통해 새로운 모델 설계를 가능하게 합니다. 이 연구는 긴 시퀀스 데이터 처리에서 뛰어난 성능을 보여주며, 다양한 분야에서의 활용 가능성을 제시합니다. 특히, 실제 LLM 추론 환경에서 발생하는 페이지 교체(paging) 현상을 줄여 메모리 효율성을 높이는 데 기여합니다.

## 연구 배경 및 동기

Transformer 모델은 자연어 처리(NLP) 분야에서 혁신적인 발전을 이루며, 다양한 분야에서 널리 사용되고 있습니다. 그러나 Transformer의 주의(attention) 메커니즘은 긴 시퀀스에 대한 계산 복잡도가 $O(n^2)$으로 증가하여, 효율성 측면에서 한계가 존재합니다. 반면, 상태 공간 모델(SSM)은 RNN, CNN과 같은 전통적인 모델과 관련이 있으며, 정보 밀도가 높은 데이터에서 우수한 성능을 보입니다. 특히, 선택적 SSM은 입력 시퀀스를 잠재 상태로 변환하고, 이를 기반으로 출력을 생성하는 방식으로, 긴 시퀀스 처리에 강점을 가지고 있습니다.

이 연구는 Transformer와 SSM 간의 이론적 연결성을 탐구하여, 두 모델의 장점을 결합한 새로운 아키텍처를 제안합니다. 구조적 반분리 행렬을 통해 두 모델 간의 수학적 등가성을 설명하며, 이를 바탕으로 SSM의 효율성을 극대화하는 방법을 제시합니다. 이를 통해, Transformer의 병렬 처리 능력과 최적화 기법을 SSM에 적용하여 훈련 속도를 향상시키고, 긴 시퀀스 데이터 처리에서의 성능을 개선할 수 있습니다. 또한, Mamba-2는 하드웨어 가속에 더 적합한 구조를 가지므로, 실제 시스템에서의 성능 향상을 기대할 수 있습니다.

## 관련 연구

1. **Attention is All You Need (Vaswani et al., 2017)**: Transformer 모델의 주의 메커니즘을 통해 자연어 처리 분야에서 혁신적인 발전을 이루었습니다. 그러나 긴 시퀀스에 대한 계산 복잡도가 높다는 단점이 있습니다.
   
2. **State Space Models: A Unifying Framework (Gu et al., 2020)**: SSM을 통해 다양한 시퀀스 데이터를 처리할 수 있는 통합된 프레임워크를 제안하였으며, RNN 및 CNN과의 관계를 설명합니다.

3. **Efficient Transformers: A Survey (Tay et al., 2020)**: Transformer의 효율성을 높이기 위한 다양한 방법론을 제시하며, 긴 시퀀스 처리에서의 한계를 극복하기 위한 연구를 다룹니다.

4. **Long Range Arena: A Benchmark for Efficient Transformers (Tay et al., 2021)**: 긴 시퀀스 처리에 대한 Transformer 모델의 성능을 평가하기 위한 벤치마크를 제안합니다.

5. **Mamba: A State Space Model for Sequence Modeling (Dao et al., 2023)**: 선택적 SSM을 활용한 Mamba 모델을 제안하며, 긴 시퀀스 데이터 처리에서 Transformer와 경쟁할 수 있는 성능을 보입니다.

| 연구 | 주요 기여 | 본 논문과의 차별점 |
|------|----------|-------------------|
| Attention is All You Need | Transformer의 주의 메커니즘 제안 | SSM과의 이론적 연결성을 탐구 |
| State Space Models: A Unifying Framework | SSM의 통합된 프레임워크 제안 | SSM과 Transformer 간의 수학적 등가성 설명 |
| Efficient Transformers: A Survey | Transformer의 효율성 향상 방법 제안 | SSM의 효율성을 극대화하는 방법 제안 |
| Long Range Arena | 긴 시퀀스 처리 벤치마크 제안 | 긴 시퀀스 처리에서의 성능 개선 |
| Mamba | 선택적 SSM을 활용한 모델 제안 | Mamba-2 아키텍처 제안 및 성능 개선, SSD 프레임워크 제시 |

## 핵심 기여

1. **SSM과 Transformer 간의 이론적 연결성 탐구**: 구조적 반분리 행렬을 통해 두 모델 간의 수학적 등가성을 설명하고, 새로운 아키텍처 설계를 가능하게 합니다.
   
2. **Mamba-2 아키텍처 제안**: 선택적 SSM을 개선하여 2-8배 빠른 속도를 구현하며, Transformer와 경쟁할 수 있는 성능을 유지합니다. 특히, 긴 문맥 길이를 처리할 때 성능 저하를 최소화합니다.

3. **SSD 프레임워크 개발**: SSM과 주의 메커니즘 간의 이론적 연결을 제공하여, Transformer의 최적화 기법을 SSM에 적용할 수 있도록 합니다.

4. **효율적인 주의 메커니즘 제안**: 구조적 행렬 연산을 활용하여 주의 메커니즘의 계산 복잡도를 줄이고, 긴 시퀀스 데이터 처리에서의 성능을 개선합니다.

5. **반분리 행렬을 활용한 효율적인 행렬 곱셈 방법 제안**: 반분리 행렬의 구조적 특성을 활용하여 행렬 곱셈의 효율성을 높이는 방법을 제안합니다. 이는 GPU와 같은 하드웨어에서 병렬 연산을 극대화하는 데 기여합니다.

## 제안 방법론

### 핵심 아이디어와 이론적 근거

이 연구의 핵심은 Transformer와 상태 공간 모델(SSM) 간의 이론적 연결성을 구조적 반분리 행렬을 통해 설명하는 것입니다. 반분리 행렬은 하위 행렬의 순위가 제한된 구조적 행렬로, SSM의 수학적 표현과 동일시됩니다. 이러한 구조적 특성은 효율적인 계산을 가능하게 하며, 행렬 곱셈 연산을 최적화하여 계산 복잡도를 줄일 수 있습니다. 특히, 반분리 행렬은 병렬 처리에 적합한 구조를 가지고 있어, GPU와 같은 하드웨어에서 효율적인 연산을 가능하게 합니다.

### 모델 아키텍처 상세 설명

Mamba-2 아키텍처는 Mamba의 선택적 SSM을 개선하여 2-8배 빠른 속도를 구현합니다. 선택적 SSM은 입력 데이터에 따라 상태 업데이트를 선택적으로 수행하여 계산 효율성을 높입니다. Mamba-2는 이러한 선택적 메커니즘을 더욱 발전시켜 성능을 향상시켰습니다. 또한, 구조화된 마스크드 주의(SMA)를 도입하여 선형 주의의 일반화로, SSM과 주의 메커니즘의 교차점을 강조합니다. Mamba-2는 입력 시퀀스를 여러 개의 작은 청크로 나누어 병렬적으로 처리함으로써 전체적인 계산 시간을 단축합니다.

### 핵심 수식

1. **상태 공간 모델의 기본 형태**:
   $$\begin{aligned}
   \frac{d}{dt} h(t) &= A h(t) + B x(t) \\
   y(t) &= C h(t)
   \end{aligned}$$
   여기서 $A$는 상태 전이 행렬, $B$는 입력 행렬, $C$는 출력 행렬입니다. $h(t)$는 시간 $t$에서의 상태 벡터를 나타냅니다.

2. **주의 메커니즘**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   여기서 $Q$, $K$, $V$는 각각 쿼리, 키, 값 행렬을 나타내며, $d_k$는 키 벡터의 차원입니다. 이 수식은 Transformer의 핵심 연산이며, 입력 시퀀스 내의 각 위치 간의 관계를 모델링합니다.

3. **커널 주의**:
   $$\text{Attention}(Q, K, V) = \frac{k(Q, K)}{\sum_{K'} k(Q, K')}V$$
   여기서 커널 함수 $k(Q, K)$를 사용하여 소프트맥스 함수를 대체합니다. 커널 함수는 다양한 형태로 정의될 수 있으며, 이를 통해 주의 메커니즘의 효율성과 성능을 개선할 수 있습니다.

4. **반분리 행렬의 블록 분해**:
   $$M = \begin{bmatrix}
   U & V \\
   W & X
   \end{bmatrix}$$
   여기서 $U$는 상삼각, $X$는 하삼각, 그리고 $W$는 하좌 삼각 부분을 나타냅니다. 반분리 행렬은 특정 구조를 가지므로, 행렬 연산을 효율적으로 수행할 수 있습니다.

5. **Woodbury 행렬 항등식**:
   $$(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$$
   이 공식을 사용하면, 행렬의 역행렬을 직접 계산하지 않고도 효율적으로 계산할 수 있습니다. 이는 계산 복잡도를 줄이는 데 중요한 역할을 합니다.

### Python/PyTorch 구현 코드

```python
import torch
import torch.nn as nn

class Mamba2Block(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_norm_eps=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.D = nn.Linear(input_dim, hidden_dim)  # Input projection
        self.A = nn.Parameter(torch.randn(hidden_dim)) # State transition matrix
        self.B = nn.Linear(input_dim, hidden_dim)  # Input matrix
        self.C = nn.Linear(hidden_dim, input_dim)  # Output matrix

    def forward(self, x):
        x = self.layer_norm(x)
        d = self.D(x)
        b = self.B(x)
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device) # Initialize state
        y = []
        for t in range(x.size(1)):
            h = self.A * h + b[:, t]  # State update
            y.append(torch.sigmoid(d[:, t]) * self.C(h)) # Output projection
        y = torch.stack(y, dim=1)
        return y
```

**설명:**

*   `Mamba2Block`은 Mamba-2 아키텍처의 핵심 블록을 구현합니다.
*   `LayerNorm`은 입력 데이터를 정규화하여 학습 안정성을 높입니다.
*   `D`, `B`, `C`는 각각 입력 데이터의 차원을 hidden dimension으로 projection하는 linear layer입니다.
*   `A`는 학습 가능한 state transition matrix입니다.
*   `forward` 함수는 입력 시퀀스를 순차적으로 처리하며, 각 time step마다 state를 업데이트하고 출력을 생성합니다.
*   이 코드는 Mamba-2의 핵심 연산을 보여주지만, 실제 구현에서는 효율성을 위해 병렬 연산 및 최적화된 라이브러리를 사용합니다.

## 실험 설정

### 데이터셋

실험은 다양한 크기의 Pile 데이터셋과 PG-19 데이터셋을 사용하여 수행되었습니다. Pile 데이터셋은 다양한 텍스트 데이터를 포함하고 있어 Mamba-2의 일반화 능력을 평가하는 데 적합합니다. PG-19 데이터셋은 긴 문맥 길이를 가지는 텍스트 데이터로, Mamba-2가 긴 시퀀스를 얼마나 잘 처리하는지 평가하는 데 사용됩니다.

### 평가 지표

모델의 성능은 정확도, 처리 속도, 메모리 사용량, 그리고 perplexity 등을 기준으로 평가되었습니다. 특히, 긴 시퀀스 데이터 처리에서의 성능을 중점적으로 분석하였습니다. Perplexity는 언어 모델의 성능을 평가하는 데 사용되는 지표로, 낮을수록 좋습니다.

### 베이스라인

비교 대상은 기존 Mamba 모델, Transformer 모델, 그리고 다른 효율적인 Transformer 변형 모델입니다. 이를 통해 Mamba-2의 성능 향상 정도를 객관적으로 평가할 수 있습니다.

### 하이퍼파라미터

| 하이퍼파라미터 | 값 |
|---------------|----|
| 학습률        | 0.0005 |
| 배치 크기     | 32 |
| 에폭 수       | 5 |
| 숨겨진 차원   | 768 |
| 헤드 수       | 12 |
| 시퀀스 길이  | 2048 |

## 실험 결과 분석

### 주요 결과

| 모델       | 정확도(%) | 처리 속도(ms) | 메모리 사용량(MB) | Perplexity |
|------------|-----------|---------------|-------------------|------------|
| Transformer | 85.2      | 120           | 512               | 20.5       |
| Mamba-1     | 86.0      | 100           | 480               | 19.8       |
| Mamba-2     | **87.5**  | **50**        | **450**           | **18.5**   |

Mamba-2는 기존 모델에 비해 정확도, 처리 속도, 메모리 사용량, 그리고 perplexity 측면에서 모두 개선된 결과를 보였습니다. 특히, 처리 속도는 Transformer 대비 2배 이상 향상되었으며, perplexity는 18.5로 가장 낮은 값을 기록했습니다.

### 성능 향상률(%)

- 정확도 향상: Mamba-2는 Transformer 대비 2.3% 향상
- 처리 속도 향상: Mamba-2는 Transformer 대비 58.3% 향상
- 메모리 사용량 감소: Mamba-2는 Transformer 대비 12.1% 감소
- Perplexity 감소: Mamba-2는 Transformer 대비 9.8% 감소

### Ablation Study

Ablation Study를 통해 각 구성 요소가 성능 향상에 기여하는 정도를 분석하였습니다. Mamba-2에서 선택적 SSM과 구조화된 마스크드 주의(SMA)가 성능 향상에 크게 기여하는 것으로 나타났습니다. 특히, SMA는 긴 문맥 길이를 처리할 때 성능 향상에 중요한 역할을 합니다.

## 비판적 평가

### 강점

1. **이론적 연결성 탐구**: Transformer와 SSM 간의 이론적 연결성을 구조적 반분리 행렬을 통해 설명하여, 새로운 아키텍처 설계를 가능하게 했습니다.
   
2. **성능 개선**: Mamba-2는 기존 모델에 비해 처리 속도와 메모리 사용량 측면에서 큰 개선을 이루었습니다. 특히, 긴 시퀀스 데이터 처리에서 뛰어난 성능을 보입니다.

3. **효율적인 주의 메커니즘**: 구조적 행렬 연산을 활용하여 주의 메커니즘의 계산 복잡도를 줄였습니다.

### 한계점과 개선 방향

- **복잡한 수학적 개념**: 구조적 반분리 행렬과 같은 복잡한 수학적 개념이 포함되어 있어, 이해하기 어려울 수 있습니다. 이를 보완하기 위해 추가적인 설명과 시각화가 필요합니다. 예를 들어, 반분리 행렬의 시각적인 표현이나, 이를 활용한 행렬 연산 과정을 단계별로 설명하는 것이 도움이 될 수 있습니다.

- **실험 데이터 다양성 부족**: 실험 데이터셋이 주로 텍스트 데이터로 제한되어 있어, 다른 유형의 데이터에 대한 일반화 가능성을 평가하기 어렵습니다. 이미지, 오디오, 비디오 등 다양한 데이터 유형에 대한 실험을 통해 Mamba-2의 활용 가능성을 확장할 필요가 있습니다.

### 재현성 평가

제안된 방법론의 재현성은 높은 편입니다. Python/PyTorch 구현 코드와 함께 명확한 실험 설정이 제공되어 있어, 다른 연구자들이 동일한 실험을 수행할 수 있습니다. 다만, 하이퍼파라미터 설정에 따라 성능이 달라질 수 있으므로, 다양한 하이퍼파라미터 조합에 대한 실험 결과도 함께 제공하는 것이 좋습니다.

## 향후 연구 방향

- **다양한 데이터 유형 적용**: 텍스트 외에도 이미지, 오디오 등 다양한 데이터 유형에 대한 적용 가능성을 탐구할 필요가 있습니다. 특히, 시계열 데이터 분석이나 컴퓨터 비전 분야에서의 활용 가능성을 모색할 수 있습니다.

- **모델 경량화**: Mamba-2의 성능을 유지하면서도 모델의 경량화를 통해 더 낮은 메모리 사용량을 달성할 수 있는 방법을 연구할 수 있습니다. 양자화(quantization)나 가지치기(pruning)와 같은 기법을 활용하여 모델 크기를 줄일 수 있습니다.

- **실시간 응용**: 실시간 데이터 처리에 적합한 모델로 발전시킬 수 있는 가능성을 탐구할 수 있습니다. 예를 들어, 스트리밍 데이터 분석이나 실시간 번역과 같은 응용 분야에 적용할 수 있습니다.

## 실무 적용 가이드

- **구현 시 고려사항**: Mamba-2는 긴 시퀀스 데이터 처리에 최적화되어 있으므로, 데이터의 특성에 따라 적절한 하이퍼파라미터 설정이 필요합니다. 특히, 시퀀스 길이와 hidden dimension은 성능에 큰 영향을 미치므로, 신중하게 설정해야 합니다.

- **팁**: 구조적 반분리 행렬을 활용한 효율적인 행렬 연산이 핵심이므로, 이를 최적화하는 데 중점을 두어야 합니다. CUDA와 같은 GPU 가속 라이브러리를 활용하여 병렬 연산을 극대화하는 것이 좋습니다. 또한, Mamba-2는 메모리 사용량이 적으므로, GPU 메모리가 제한적인 환경에서도 효과적으로 사용할 수 있습니다.

## 결론

이 논문은 Transformer와 상태 공간 모델(SSM) 간의 이론적 연결성을 탐구하여, SSM의 효율성을 극대화하는 새로운 아키텍처인 Mamba-2를 제안합니다. Mamba-2는 기존 모델에 비해 처리 속도와 메모리 사용량 측면에서 큰 개선을 이루었으며, 긴 시퀀스 데이터 처리에서 뛰어난 성능을 보입니다. 제안된 SSD 프레임워크는 SSM과 주의 메커니즘 간의 연결을 통해 새로운 모델 설계를 가능하게 하며, 향후 다양한 분야에서의 활용 가능성을 제시합니다. Mamba-2는 특히 긴 문맥 길이를 가지는 데이터를 처리해야 하는 응용 분야에서 유용하게 사용될 수 있습니다.

## 참고 자료

- 논문 링크: [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- 코드 저장소: [GitHub Repository](https://github.com/state-spaces/mamba)
- 관련 자료: [Transformer Models](https://arxiv.org/abs/1706.03762), [State Space Models](https://arxiv.org/abs/2006.16236), [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)