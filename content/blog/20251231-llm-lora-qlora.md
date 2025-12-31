---
title: "LLM 파인튜닝: LoRA와 QLoRA 기법"
date: "2025-12-31"
excerpt: "대규모 언어 모델(LLM, Large Language Models)은 자연어 처리(NLP, Natural Language Processing) 분야에서 혁신을 이끌어왔습니다. 이러한 모델들은 방대한 양의 데이터를 활용하여 다양한 언어적 과제를 수행할 수 있습니다. 하지만 LLM은 막대한 컴퓨팅 자원을 요구하며, 특정 작업에 맞게 모델을 조정하는 파인튜닝 과..."
category: "NLP"
tags: []
thumbnail: "/assets/images/blog/20251231-llm-lora-qlora.jpg"
---

# LLM 파인튜닝: LoRA와 QLoRA 기법

## 도입부

대규모 언어 모델(LLM, Large Language Models)은 자연어 처리(NLP, Natural Language Processing) 분야에서 혁신을 이끌어왔습니다. 이러한 모델들은 방대한 양의 데이터를 활용하여 다양한 언어적 과제를 수행할 수 있습니다. 하지만 LLM은 막대한 컴퓨팅 자원을 요구하며, 특정 작업에 맞게 모델을 조정하는 파인튜닝 과정 또한 부담이 될 수 있습니다.

파인튜닝은 특정 작업에 대한 모델의 성능을 극대화하는 과정입니다. 최근에는 파인튜닝의 효율성을 높이기 위한 다양한 기법들이 개발되고 있으며, 그 중에서도 LoRA(Low-Rank Adaptation)와 QLoRA(Quantized Low-Rank Adaptation)는 파인튜닝의 비용과 시간을 줄이면서도 준수한 성능을 유지할 수 있는 기법으로 각광받고 있습니다. 이번 포스트에서는 LoRA와 QLoRA의 핵심 개념, 작동 원리, 그리고 간단한 구현 예제를 살펴보겠습니다.

## 본문

### LoRA(Low-Rank Adaptation)란?

LoRA는 모델의 모든 파라미터를 업데이트하는 대신, 파라미터 변경 행렬을 저랭크(low-rank) 행렬로 분해하여 학습 가능한 파라미터 수를 크게 줄이는 기법입니다.  일반적으로 LLM은 수백만에서 수십억 개의 파라미터를 가지고 있어, 전체 파라미터를 조정하는 것은 메모리 및 계산 비용 측면에서 비효율적입니다. LoRA는 이러한 문제를 해결하기 위해, 기존 모델의 가중치를 고정시키고, 추가적인 저랭크 행렬을 학습시켜 파인튜닝을 진행합니다.

#### LoRA의 작동 원리

LoRA는 기본적으로 다음과 같은 수식을 사용하여 파라미터를 조정합니다:

$$ W' = W_0 + \Delta W = W_0 + BA $$

여기서 $W_0$는 사전 학습된 원래의 파라미터 (frozen), $\Delta W$는 업데이트될 파라미터 변경 행렬, $B \in \mathbb{R}^{d \times r}$ 와 $A \in \mathbb{R}^{r \times k}$는 학습 가능한 저랭크 행렬입니다. ($r \ll min(d,k)$). $r$은 LoRA의 랭크 하이퍼파라미터이며, 이 값을 조정하여 학습 가능한 파라미터의 수를 조절할 수 있습니다.  $A$와 $B$의 차원이 낮기 때문에 $\Delta W$의 차원도 낮아지고, 이는 곧 파인튜닝의 효율성을 높입니다.  파인튜닝 시에는 $A$와 $B$만 업데이트되고, $W_0$는 고정됩니다.

#### Python 예제 코드 (PyTorch)

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # A와 B 행렬 초기화
        self.A = nn.Parameter(torch.randn(self.in_features, rank))
        self.B = nn.Parameter(torch.randn(rank, self.out_features))

        # A 행렬은 Kaiming He 초기화 사용 (일반적인 방법)
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        # B 행렬은 0으로 초기화
        nn.init.zeros_(self.B)

        # original_layer의 weight는 학습하지 않도록 설정
        self.original_layer.weight.requires_grad = False

    def forward(self, x):
        # original layer 통과
        original_output = self.original_layer(x)

        # LoRA adaptation 적용
        delta_W = torch.matmul(x, self.A)
        delta_W = torch.matmul(delta_W, self.B)

        return original_output + delta_W

# 예제: 기존 Linear 레이어에 LoRA 적용
original_layer = nn.Linear(768, 768)
lora_layer = LoRALayer(original_layer, rank=8) # rank 값 조정 가능

# 입력 데이터
x = torch.randn(1, 768)
output = lora_layer(x)

print(f"Output shape: {output.shape}")
```

**설명:**

*   `LoRALayer` 클래스는 LoRA를 적용할 레이어를 감싸는 역할을 합니다.
*   `rank`는 LoRA의 랭크 하이퍼파라미터입니다. 이 값을 조정하여 학습 가능한 파라미터 수를 조절할 수 있습니다.
*   `A`와 `B`는 학습 가능한 저랭크 행렬입니다.
*   `forward` 메서드에서는 입력 `x`를 original layer에 통과시킨 후, LoRA adaptation을 적용하여 결과를 반환합니다.
*   기존 레이어의 가중치(`original_layer.weight`)는 `requires_grad = False`로 설정하여 학습되지 않도록 합니다.

### QLoRA(Quantized Low-Rank Adaptation)란?

QLoRA는 LoRA의 아이디어를 더욱 발전시켜, 양자화(quantization)를 통해 메모리 효율성을 극대화하는 방법입니다. 양자화는 모델의 파라미터를 낮은 정밀도(예: 32비트 부동 소수점에서 4비트 정수)로 표현하여 모델 크기를 줄이는 기술입니다. QLoRA는 LoRA의 저랭크 행렬에 양자화를 적용하여, 파인튜닝에 필요한 메모리 공간을 획기적으로 줄입니다. 특히, 4비트 NormalFloat 양자화 방식을 사용하여 정보 손실을 최소화하면서 메모리 효율성을 높입니다.

#### QLoRA의 작동 원리

QLoRA는 LoRA에서 사용한 저차원 행렬 $A$와 $B$에 대해 양자화를 수행하는 것이 핵심입니다.  더 정확히는, 사전 학습된 모델의 가중치를 4비트로 양자화하여 고정시키고, LoRA에서 추가되는 저랭크 행렬 $A$와 $B$만 학습합니다.  이를 통해 메모리 사용량을 줄이면서도, LoRA의 파인튜닝 효과를 유지할 수 있습니다.  QLoRA는 또한 "paged optimizers"라는 기술을 사용하여 GPU 메모리가 부족한 경우 CPU 메모리를 활용하여 학습을 진행할 수 있도록 합니다.

#### Python 예제 코드

```python
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class QLoRALayer(nn.Module):
    def __init__(self, original_layer, rank):
        super(QLoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # QuantStub and DeQuantStub for quantization
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # A와 B 행렬 초기화
        self.A = nn.Parameter(torch.randn(self.in_features, rank))
        self.B = nn.Parameter(torch.randn(rank, self.out_features))

        # A 행렬은 Kaiming He 초기화 사용 (일반적인 방법)
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        # B 행렬은 0으로 초기화
        nn.init.zeros_(self.B)

        # original_layer의 weight는 학습하지 않도록 설정
        self.original_layer.weight.requires_grad = False

    def forward(self, x):
        # Quantize input
        x = self.quant(x)

        # original layer 통과
        original_output = self.original_layer(x)

        # LoRA adaptation 적용
        delta_W = torch.matmul(x, self.A)
        delta_W = torch.matmul(delta_W, self.B)

        # Dequantize output
        delta_W = self.dequant(delta_W)

        return original_output + delta_W

# 기존 모델에 QLoRA 적용
original_layer = nn.Linear(768, 768)
qlora_layer = QLoRALayer(original_layer, rank=8)

# 입력 데이터
x = torch.randn(1, 768)
output = qlora_layer(x)

print(f"Output shape: {output.shape}")
```

**주의:**

*   위 코드는 PyTorch의 `torch.quantization` 모듈을 사용하여 양자화를 시뮬레이션하는 예시입니다. 실제로 QLoRA를 구현하려면 더 복잡한 양자화 기법(예: 4비트 NormalFloat)을 사용해야 하며, 이는 일반적으로 허깅페이스 트랜스포머 라이브러리 등을 통해 제공됩니다.
*   위 코드는 단순화를 위해 모델 전체가 아닌 LoRA 레이어에만 양자화를 적용하는 것을 보여줍니다.  실제 QLoRA에서는 사전 학습된 모델의 가중치도 양자화됩니다.

### LoRA와 QLoRA의 비교

| 특징        | LoRA                                  | QLoRA                               |
| ----------- | ------------------------------------- | ------------------------------------- |
| 핵심 아이디어 | 저랭크 행렬을 사용하여 파라미터 효율적인 파인튜닝 | 양자화 + 저랭크 행렬을 사용하여 메모리 효율적인 파인튜닝 |
| 메모리 사용량 | 감소                                  | 대폭 감소                               |
| 파라미터 수   | 감소                                  | 감소 (양자화로 인한 효과)                   |
| 구현 복잡도   | 비교적 간단                            | 비교적 복잡 (양자화 관련)                |
| 장점        | 파인튜닝 비용 감소, 성능 유지                 | 메모리 효율성 극대화, 대규모 모델 파인튜닝 가능 |
| 단점        | QLoRA 대비 메모리 효율성 낮음               | 양자화로 인한 성능 저하 가능성 존재          |

## 결론

LoRA와 QLoRA는 대규모 언어 모델의 파인튜닝을 보다 효율적으로 수행할 수 있는 강력한 도구입니다. LoRA는 파라미터 변경 행렬을 저랭크 행렬로 분해하여 모델의 경량화를 도모하며, QLoRA는 여기에 양자화를 더해 메모리 및 계산 효율성을 극대화합니다. 이러한 기법들은 모델의 성능을 유지하면서도 자원 소모를 줄일 수 있어, NLP 분야에서 더욱 널리 사용될 것으로 기대됩니다. 특히, QLoRA는 제한된 자원을 가진 환경에서도 대규모 모델을 파인튜닝할 수 있도록 해주는 중요한 기술입니다.

**팁:**

*   LoRA의 `rank` 값을 조정하여 성능과 메모리 사용량 사이의 균형을 맞출 수 있습니다.
*   QLoRA를 사용할 때는 양자화로 인한 성능 저하를 최소화하기 위해 적절한 양자화 기법을 선택해야 합니다.

추가 학습을 원하는 독자들은 아래 참고 자료를 확인해보시기 바랍니다:

- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [QLoRA 논문](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT 라이브러리](https://github.com/huggingface/peft): LoRA 및 QLoRA를 포함한 다양한 파라미터 효율적인 파인튜닝 기법을 제공합니다.
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)

LoRA와 QLoRA를 활용하여 여러분의 프로젝트에서 더 나은 성능과 효율성을 경험해보세요!
