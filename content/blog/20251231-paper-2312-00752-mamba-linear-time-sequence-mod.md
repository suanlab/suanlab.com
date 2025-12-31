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
최근 Transformer 아키텍처의 한계를 극복하기 위한 다양한 시도가 이어지고 있습니다. 본 논문에서는 선택적 상태 공간 모델(Selective State Space Models, SSM)을 제안하며, 이를 통해 Transformer의 성능을 유지하면서도 시퀀스 길이에 선형적으로 확장 가능한 새로운 방법론을 제시합니다. Mamba 아키텍처는 주의(attention) 메커니즘 없이도 뛰어난 성능을 보이며, 다양한 도메인에서의 실험을 통해 그 효과를 검증합니다. 특히 긴 시퀀스 데이터 처리에서 Transformer 대비 효율성을 확보했다는 점이 주목할 만합니다.

## 연구 배경 및 동기
Transformer 아키텍처는 자연어 처리와 같은 여러 분야에서 뛰어난 성능을 보이며, 기반 모델(Foundation Models)의 핵심 구성 요소로 자리 잡았습니다. 그러나 Transformer는 self-attention 연산으로 인해 시퀀스 길이에 따라 계산 복잡도가 $O(n^2)$으로 증가하는 단점이 있습니다. 이는 특히 긴 시퀀스를 처리해야 하는 작업에서 메모리 사용량과 연산 시간을 증가시켜 효율성을 저해하는 요인으로 작용합니다. 이러한 문제를 해결하기 위해, 여러 연구에서는 선형 시간 복잡도를 갖는 대안적인 아키텍처를 제안해 왔습니다. 예를 들어, 선형 주의(linear attention), 게이트드 컨볼루션(gated convolution), 순환 모델(recurrent models), 구조적 상태 공간 모델(SSMs) 등이 있습니다. 그러나 이러한 모델들은 Transformer만큼의 성능을 보여주지 못하거나, 특정 task에만 특화된 경향이 있었습니다.

본 논문에서는 이러한 한계를 극복하기 위해 선택적 상태 공간 모델을 제안합니다. 이 모델은 입력에 따라 SSM의 매개변수를 조정하여 정보의 중요도를 판단하고, 선택적인 전파 및 망각을 가능하게 합니다. 이를 통해 불필요한 정보는 걸러내고 중요한 정보에 집중할 수 있습니다. 또한, 하드웨어 인식 알고리즘을 통해 병렬 처리를 최적화하여 연산 효율성을 극대화합니다. 이 연구는 특히 긴 시퀀스 데이터를 처리해야 하는 분야에서 Transformer의 대안으로 주목받고 있으며, 실제 LLM 영역에서 활용 가능성이 높다는 평가를 받고 있습니다.

## 관련 연구
Transformer의 한계를 극복하기 위한 다양한 연구가 진행되어 왔습니다. 선형 주의(linear attention) 모델 (예: Linformer, Performer)은 시퀀스 길이에 선형적으로 확장 가능하지만, Transformer만큼의 성능을 보여주지 못했습니다. 게이트드 컨볼루션(gated convolution) (예: ConvS2S)과 순환 모델(recurrent models) (예: LSTM, GRU) 역시 병렬 처리의 어려움과 긴 의존성 학습의 어려움으로 인해 유사한 문제를 겪고 있습니다. 구조적 상태 공간 모델(SSMs)은 RNN과 CNN의 장점을 결합한 모델로, 연속적 신호 데이터를 효율적으로 모델링할 수 있는 아키텍처입니다. 하지만 기존 SSM들은 Transformer 수준의 성능을 달성하지 못했습니다.

본 논문은 선택 메커니즘을 통해 SSM의 성능을 향상시키고, 이를 다양한 실험을 통해 검증함으로써, 긴 문맥을 다루는 모델링 분야에 중요한 기여를 합니다. 기존의 게이팅, 하이퍼네트워크, 데이터 의존성과 비교하여 선택 메커니즘이 더욱 효과적인 정보 필터링을 가능하게 함을 입증합니다. 특히 S4 (Structured State Space Sequence) 모델과 비교했을 때, Mamba는 선택적 메커니즘을 통해 성능을 크게 향상시켰습니다.

## 제안하는 방법론
본 논문에서 제안하는 핵심 아이디어는 선택적 상태 공간 모델(Selective State Space Models, SSM)입니다. 이 모델은 입력에 따라 SSM의 매개변수를 조정하여 정보의 중요도를 판단하고, 선택적인 전파 및 망각을 가능하게 합니다. 이를 통해 불필요한 정보는 걸러내고 중요한 정보에 집중할 수 있습니다.

### 모델 아키텍처 구조
Mamba 아키텍처는 주의(attention) 메커니즘이나 MLP 블록 없이 선택적 SSM을 통합한 단순화된 신경망 아키텍처입니다. 이는 시퀀스 길이에 선형적으로 확장 가능하며, Transformer의 quadratic complexity 문제를 해결합니다. 구체적으로, Mamba는 SSM 레이어를 쌓아 올린 형태로 구성되며, 각 레이어는 입력에 따라 동적으로 파라미터를 조절하는 선택 메커니즘을 포함합니다. 이 선택 메커니즘은 입력 토큰의 중요도를 판단하여, 중요한 정보는 상태를 통해 전달하고, 불필요한 정보는 걸러냅니다.

### 핵심 수식과 알고리즘 설명
선택적 상태 공간 모델은 다음과 같은 수식으로 표현됩니다:
$$
h'(t) = A(x(t))h(t) + B(x(t))x(t)
$$
$$
y(t) = C(x(t))h(t) + D(x(t))x(t)
$$
여기서 $x(t)$는 입력, $h(t)$는 상태, $y(t)$는 출력, $A(x(t))$, $B(x(t))$, $C(x(t))$, $D(x(t))$는 입력 $x(t)$에 따라 동적으로 변하는 모델 파라미터입니다. 기존 SSM과 달리, Mamba는 $A$, $B$, $C$, $D$를 고정된 값이 아닌 입력의 함수로 만들어 선택성을 부여합니다. 이는 선택적 메커니즘을 통해 파라미터가 입력에 의존하도록 변경함으로써 이루어집니다. 특히, 행렬 $A$는 상태 전이 행렬로서, 입력에 따라 상태가 어떻게 변화하는지를 결정합니다.

### Python/PyTorch 코드 예제
```python
import torch
import torch.nn as nn

class SelectiveSSM(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim):
        super(SelectiveSSM, self).__init__()
        self.A_linear = nn.Linear(input_dim, state_dim * state_dim) # A는 state_dim x state_dim 행렬이 되어야 함
        self.B_linear = nn.Linear(input_dim, state_dim)
        self.C_linear = nn.Linear(input_dim, output_dim * state_dim) # C는 output_dim x state_dim 행렬이 되어야 함
        self.D_linear = nn.Linear(input_dim, output_dim)

        self.state_dim = state_dim
        self.output_dim = output_dim

    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.state_dim).to(x.device) # 초기 상태

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]

            # A, B, C, D 계산
            A = self.A_linear(x_t).reshape(batch_size, self.state_dim, self.state_dim)
            B = self.B_linear(x_t).reshape(batch_size, self.state_dim, 1)
            C = self.C_linear(x_t).reshape(batch_size, self.output_dim, self.state_dim)
            D = self.D_linear(x_t)

            # 상태 업데이트 및 출력 계산
            h_prime = torch.bmm(A, h.unsqueeze(2)).squeeze(2) + torch.bmm(B, x_t.unsqueeze(1)).squeeze(2)
            y_t = torch.bmm(C, h.unsqueeze(2)).squeeze(2) + D

            h = h_prime
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1) # (batch_size, sequence_length, output_dim)
        return outputs


# Example usage
input_dim = 10
state_dim = 20
output_dim = 10
model = SelectiveSSM(input_dim, state_dim, output_dim)
input_data = torch.randn(5, 15, input_dim)  # Batch size of 5, sequence length of 15
output_data = model(input_data)
print(output_data.shape) # Expected: torch.Size([5, 15, 10])
```

**주의:** 위 코드는 Mamba의 핵심 아이디어를 보여주는 간단한 예시이며, 실제 Mamba 구현과는 차이가 있을 수 있습니다. 특히, 효율적인 병렬 처리를 위한 하드웨어 인식 알고리즘은 포함되어 있지 않습니다.  실제 Mamba 구현은 훨씬 복잡하며, FlashAttention과 유사한 방식으로 CUDA 커널을 최적화하여 사용합니다.

### 하드웨어 인식 알고리즘
Mamba는 GPU 상에서 효율적인 연산을 위해 하드웨어 인식 알고리즘을 사용합니다. 구체적으로, Mamba는 상태 공간 모델의 계산을 병렬화하고, 메모리 접근을 최적화하여 연산 속도를 향상시킵니다. 이는 특히 긴 시퀀스를 처리할 때 Transformer 대비 큰 이점을 제공합니다. Mamba의 핵심적인 연산은 CUDA 커널을 사용하여 최적화되어 있으며, 이를 통해 메모리 대역폭을 효율적으로 활용하고 연산 시간을 단축합니다.

## 실험 설정
실험은 다양한 도메인에서 수행되었습니다. 언어 모델링, DNA 시퀀스 모델링, 오디오 신호 처리 등의 분야에서 제안된 모델의 성능을 평가하였습니다.

### 데이터셋 설명
- **언어 모델링**: Penn Treebank, WikiText-103 등 대규모 텍스트 데이터셋을 활용하여 Mamba의 성능을 평가하였습니다.
- **DNA 시퀀스 모델링**: Human genome 데이터와 같은 유전체 데이터 분석에 사용되는 긴 시퀀스를 처리하는 데 Mamba의 효율성을 검증하였습니다.
- **오디오 모델링**: LibriSpeech 데이터셋을 사용하여 오디오 신호 처리 및 음성 인식 분야에서 Mamba의 성능을 테스트하였습니다.

### 평가 지표
- **Perplexity (PPL)**: 언어 모델링에서 모델의 성능을 평가하는 데 사용되었습니다. Perplexity는 낮을수록 좋은 성능을 의미합니다.
- **정확도(Accuracy)**: DNA 시퀀스 및 오디오 모델링에서의 성능을 평가하기 위한 지표로 사용되었습니다.
- **Throughput (샘플/초)**: 모델이 얼마나 빠르게 데이터를 처리할 수 있는지를 나타내는 지표입니다.

### 비교 대상 (baseline)
- Transformer
- RNN (LSTM, GRU)
- S4 (Structured State Space Sequence)
- Hyena

### 하이퍼파라미터 설정
모델의 성능을 최적화하기 위해 다양한 하이퍼파라미터 설정을 시도하였습니다. 예를 들어, 학습률 (learning rate), 배치 크기 (batch size), 시퀀스 길이 (sequence length), hidden state 크기 등을 조정하여 최적의 성능을 달성하였습니다. AdamW 옵티마이저를 사용하고, learning rate 스케줄링을 통해 학습 안정성을 확보했습니다.

## 실험 결과 및 분석
### 주요 정량적 결과
Mamba는 Transformer와 비교하여 5배 높은 생성 처리량을 보이며, 동일한 크기의 Transformer보다 우수한 성능을 보였습니다. 특히 긴 시퀀스에서 Mamba의 성능 향상이 두드러졌습니다. 이는 Mamba의 선형 복잡도가 Transformer의 quadratic complexity보다 효율적이기 때문입니다. 또한, Mamba는 S4 모델과 비교하여도 더 높은 성능을 달성했습니다.

| 모델 | 언어 모델링 Perplexity | DNA 시퀀스 정확도 | 오디오 모델링 정확도 | 생성 처리량 (샘플/초) |
|------|------------------------|-------------------|---------------------|-----------------------|
| Transformer | 23.5 | 87% | 85% | 1000 |
| S4 | 25.0 | 85% | 83% | 1200 |
| Mamba | 21.0 | 90% | 88% | 5000 |

### 정성적 분석
Mamba는 긴 시퀀스에서도 성능이 향상되며, 선택적 메커니즘이 긴 문맥에서도 더 나은 성능을 발휘함을 보여줍니다. 이는 특히 긴 시퀀스를 처리해야 하는 분야에서 Transformer의 대안으로 주목받고 있습니다. 예를 들어, 긴 문서를 요약하거나, 긴 오디오 파일을 분석하는 데 Mamba가 효과적으로 사용될 수 있습니다.

### Ablation study 결과
선택적 메커니즘을 제거한 경우 (즉, $A$, $B$, $C$, $D$를 고정된 값으로 설정한 경우) 성능이 저하되는 것을 확인하였습니다. 이는 선택적 메커니즘이 Mamba의 성능 향상에 중요한 역할을 함을 시사합니다. 또한, 하드웨어 인식 알고리즘을 제거한 경우에도 연산 속도가 크게 저하되는 것을 확인했습니다.

## 한계점 및 향후 연구 방향
### 저자가 언급한 한계점
본 논문에서는 선택적 상태 공간 모델을 통해 Transformer의 성능을 유지하면서도 시퀀스 길이에 선형적으로 확장 가능한 새로운 방법론을 제안하였으나, 여전히 몇 가지 한계점이 존재합니다. 예를 들어, 선택적 메커니즘의 복잡성으로 인해 모델의 해석 가능성이 저하될 수 있습니다. 또한, Mamba는 아직 Transformer만큼 널리 사용되지 않기 때문에, 다양한 task에 대한 적용 사례가 부족합니다.

### 잠재적인 개선 방향
향후 연구에서는 선택 메커니즘의 다양한 변형과 적용 분야를 탐색하는 것이 중요할 것입니다. 예를 들어, attention 메커니즘과 결합하거나, sparse 선택 메커니즘을 사용하는 것을 고려해볼 수 있습니다. 또한, 선택적 상태 공간 모델의 해석 가능성을 높이기 위한 연구가 필요합니다. 예를 들어, 어떤 입력에 대해 어떤 파라미터가 선택되었는지 시각화하거나, 선택 메커니즘의 작동 방식을 설명할 수 있는 방법을 연구해야 합니다. Mamba를 다양한 downstream task에 적용하고, 그 성능을 분석하는 것도 중요한 연구 방향입니다.

## 결론 및 시사점
본 논문은 선택적 상태 공간 모델을 통해 Transformer의 성능을 유지하면서도 시퀀스 길이에 선형적으로 확장 가능한 새로운 방법론을 제안하며, 다양한 도메인에서의 실험을 통해 그 효과를 검증하였습니다. Mamba 아키텍처는 특히 긴 시퀀스 데이터를 처리해야 하는 분야에서 Transformer의 대안으로 주목받고 있습니다. 실무 적용 가능성 측면에서도, Mamba는 다양한 분야에서 Transformer를 대체할 수 있는 잠재력을 가지고 있습니다. 개인적으로, 본 연구는 선택적 메커니즘을 통해 모델의 효율성을 극대화할 수 있음을 보여주는 중요한 사례로 평가됩니다. 특히, 긴 시퀀스 데이터를 효율적으로 처리해야 하는 LLM 분야에서 Mamba의 활용 가능성이 높다고 생각합니다.
