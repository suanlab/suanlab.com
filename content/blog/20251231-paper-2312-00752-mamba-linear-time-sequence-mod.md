---
title: "[논문 리뷰] Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
date: "2025-12-31"
excerpt: "Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time a..."
category: "Paper Review"
tags: ["Paper Review","cs.LG","cs.AI","cs.LG"]
thumbnail: "/assets/images/blog/20251231-paper-2312-00752-mamba-linear-time-sequence-mod.jpg"
---

# [논문 리뷰] Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## TL;DR
Mamba는 선택적 상태 공간 모델(Selective State Space Models, SSMs)을 활용하여 시퀀스 길이에 선형적으로 확장 가능한 효율적인 시퀀스 모델링 방법론을 제안합니다. Transformer 기반 모델의 대안으로, 긴 시퀀스 데이터 처리에서 우수한 성능을 발휘하며, 언어 모델링, DNA 시퀀스 분석, 오디오 생성 등 다양한 도메인에서 혁신적인 가능성을 보여줍니다.

## 연구 동기 및 문제 정의
Transformer는 많은 딥러닝 애플리케이션에서 필수적인 아키텍처로 자리 잡았지만, 긴 시퀀스를 처리할 때 computational inefficiency가 문제로 대두되었습니다. Transformer의 self-attention 메커니즘은 시퀀스 길이의 제곱에 비례하는 계산 복잡도를 가지며 ($O(n^2)$), 이로 인해 긴 시퀀스 처리에 비효율적입니다. 예를 들어, 10,000 토큰 길이의 시퀀스를 처리하는 경우, self-attention은 1억 번의 연산을 수행해야 합니다. 이에 따라 연구자들은 subquadratic-time 아키텍처를 개발했지만, 이러한 모델들은 Transformer의 성능에 미치지 못했습니다. 본 논문은 이러한 문제를 해결하기 위해 선택적 상태 공간 모델을 제안하며, 긴 시퀀스 처리에서의 효율성과 성능을 동시에 개선하고자 합니다.

## 제안하는 방법론
Mamba는 선택적 상태 공간 모델을 통해 복잡한 attention 메커니즘 없이도 고성능을 발휘할 수 있는 단순한 아키텍처를 제공합니다. 선택적 SSM은 입력에 따라 매개변수를 조정하여 시퀀스 내 정보의 선택적 전달 및 망각을 가능하게 합니다. 이는 모델이 현재 입력에 가장 관련성이 높은 정보에 집중하고 불필요한 정보를 걸러낼 수 있도록 합니다. 선택 메커니즘은 입력에 따라 모델이 특정 데이터를 선택적으로 집중하거나 무시할 수 있도록 하여, 시퀀스 모델의 효율성과 효과성을 높입니다. Mamba는 하드웨어 친화적인 알고리즘을 설계하여 병렬 처리를 가능하게 하고, FlashAttention과 유사한 메모리 최적화 기법을 활용하여 더 큰 모델과 더 긴 시퀀스를 처리할 수 있도록 설계되었습니다. 구체적으로, Mamba는 GPU의 병렬 처리 능력을 최대한 활용하기 위해, recurrent 연산을 convolution 연산으로 변환하는 방법을 사용합니다.

선택적 SSM의 핵심 수식은 다음과 같습니다:

$$
\begin{aligned}
\Delta, B, C &= f(x(t)) \\
h'(t) &= A(\Delta(t)) h(t) + B(\Delta(t)) x(t) \\
y(t) &= C(\Delta(t)) h'(t)
\end{aligned}
$$

여기서 $x(t)$는 입력, $h(t)$는 hidden state, $y(t)$는 출력, $\Delta$는 시간 스케일 파라미터, $A$, $B$, $C$는 학습 가능한 파라미터입니다. $\Delta$는 각 단계의 업데이트 속도를 제어하며, $B$와 $C$는 각각 입력에서 상태로, 상태에서 출력으로의 변환을 담당합니다. 중요한 점은 이러한 매개변수들($\Delta, B, C$)이 입력 $x(t)$에 따라 동적으로 조정된다는 것입니다. $A(\Delta(t))$는 시간 스케일에 따라 상태 전이 행렬 $A$를 조절하는 역할을 합니다.

```python
# Mamba의 간략화된 Python 구현 (실제 구현은 더 복잡함)
import torch
import torch.nn as nn

class MambaBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.input_proj = nn.Linear(dim, hidden_dim * 3)
        self.output_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        delta, B, C = torch.split(self.input_proj(x), x.shape[-1], dim=-1)
        # 실제 SSM 연산은 이 부분에서 수행됨 (생략)
        # ...
        output = self.output_proj(C)
        return output
```

## 주요 실험 결과
Mamba는 다양한 도메인에서 Transformer 기반 모델과 비교하여 경쟁력 있는 성능을 보였습니다. 특히, 긴 시퀀스 데이터 처리에서 우수한 성능을 발휘하였습니다.

- **언어 모델링**: Mamba-3B 모델은 같은 크기의 Transformer를 능가했으며, 두 배 크기의 Transformer와 성능이 유사했습니다. 예를 들어, Perplexity 지표에서 Mamba는 Transformer보다 낮은 값을 기록했습니다.
- **DNA 모델링**: 긴 시퀀스에서의 성능이 중요한 DNA 시퀀스 모델링에서도 Mamba는 유전자 간의 장거리 의존성을 더 잘 포착하여 예측 정확도를 향상시켰습니다. 구체적으로, Mamba는 염기 서열의 특정 패턴을 더 정확하게 식별했습니다.
- **오디오 모델링**: 오디오 파형 생성에서도 기존의 최첨단 모델보다 우수한 성능을 보이며, 긴 컨텍스트에서도 성능이 향상되었습니다. 예를 들어, Mamba는 더 긴 오디오 클립에서 일관성 있는 음악을 생성할 수 있었습니다.
- **효율성**: Mamba는 Transformer에 비해 5배 높은 추론 처리량을 제공하며, 이는 실시간 애플리케이션이나 대규모 데이터 처리 작업에 더 적합합니다. 이는 Mamba의 선형적인 계산 복잡도 덕분입니다.

## 한계점 및 향후 연구 방향
현재 Mamba는 긴 시퀀스 처리에 강점을 보이지만, 특정한 도메인에 최적화된 모델과 비교할 때 성능이 부족할 수 있습니다. 예를 들어, 특정 NLP 작업에서는 fine-tuning된 BERT 모델이 Mamba보다 더 나은 성능을 보일 수 있습니다. 향후 연구에서는 선택적 상태 공간 모델을 다양한 도메인에 더욱 적합하게 조정하는 방법을 탐구할 필요가 있습니다. 또한, 선택적 SSM의 효율성을 더욱 높이기 위한 하드웨어 최적화 연구도 필요합니다. Mamba의 병렬 처리 효율성을 높이기 위한 연구, 그리고 메모리 사용량을 더욱 줄이기 위한 양자화(quantization) 기법 연구 등이 진행될 수 있습니다.

## 결론 및 개인 의견
Mamba는 선택적 상태 공간 모델을 통해 Transformer의 한계를 극복하고, 다양한 도메인에서 혁신적인 가능성을 제시합니다. 특히 긴 시퀀스가 중요한 분야에서의 활용 가능성을 보여주며, 향후 시퀀스 모델링 연구의 중요한 방향을 제시합니다. 개인적으로, Mamba의 선택적 메커니즘이 다양한 응용 분야에서 혁신을 가져올 것으로 기대되며, 향후 연구를 통해 더 많은 발전이 이루어지기를 바랍니다. 특히, Mamba가 기존의 RNN이나 Transformer 기반 모델을 대체하고, 새로운 시퀀스 모델링 패러다임을 제시할 수 있을지 주목할 필요가 있습니다.
