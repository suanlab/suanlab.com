---
title: "[논문 리뷰] An Information Theoretic Perspective on Agentic System Design"
date: "2025-12-31"
excerpt: "Agentic language model (LM) systems power modern applications like \"Deep Research\" and \"Claude Code,\" and leverage multi-LM architectures to overcome context limitations. Beneath their apparent divers..."
category: "Paper Review"
tags: ["Paper Review","cs.LG","cs.AI","cs.CL"]
thumbnail: "/assets/images/blog/20251231-paper-2512-21720-an-information-theoretic-persp.jpg"
---

```markdown
# [논문 리뷰] An Information Theoretic Perspective on Agentic System Design

## TL;DR
이 논문은 에이전트 언어 모델(LM) 시스템의 설계에 정보 이론적 관점을 도입하여, 압축기와 예측기의 조합이 시스템 성능에 미치는 영향을 분석합니다. 상호 정보를 활용하여 압축 품질을 평가하고, 이를 통해 다양한 데이터셋에서의 성능을 예측합니다. 결과적으로, 큰 압축기는 더 높은 정확도와 효율성을 제공하며, 이는 예측기의 성능을 향상시키는 데 효과적임을 보여줍니다. 특히, 정보 병목(Information Bottleneck) 이론과의 연관성을 제시합니다.

## 연구 배경 및 동기
현대의 에이전트 언어 모델 시스템은 "Deep Research"나 "Claude Code"와 같은 응용 프로그램에서 긴 문맥을 처리하기 위해 다수의 LM을 활용합니다. 이러한 시스템은 작은 "압축기" 모델이 원시 문맥을 요약하고, 큰 "예측기" 모델이 이를 소비하는 구조로 설계되어 있습니다. 그러나, 이러한 구조의 설계는 주로 경험적으로 이루어지며, 압축기와 예측기의 선택이 시스템 성능에 미치는 영향을 체계적으로 이해하기 어렵습니다. 예를 들어, 특정 데이터셋에 적합한 압축기를 선택하거나, 압축률을 최적화하여 예측기의 성능을 극대화하는 것은 여전히 도전 과제로 남아 있습니다.

이 연구는 이러한 문제를 해결하기 위해 정보 이론적 접근을 제안합니다. 상호 정보를 활용하여 압축 품질을 평가하고, 이를 통해 다양한 데이터셋과 모델 패밀리에서의 성능을 예측하는 방법론을 제시합니다. 이는 시스템 설계의 효율성을 높이고, 다양한 응용 분야에서의 성능을 개선하는 데 기여할 수 있습니다. 특히, 압축기와 예측기의 최적의 조합을 찾는 데 있어 정보 이론적 접근은 유용한 도구가 될 수 있습니다. 또한, 정보 병목(Information Bottleneck) 이론을 활용하여 압축 과정에서 필요한 정보만을 추출하고 불필요한 정보를 제거함으로써 예측기의 성능을 향상시키는 것을 목표로 합니다.

## 관련 연구
기존 연구에서는 에이전트 시스템의 설계에 있어 다양한 접근 방식을 제안해 왔습니다. 예를 들어, Retrieval-Augmented Generation (RAG) 시스템에서는 외부 지식 베이스를 활용하여 문맥을 확장하는 방법이 연구되었습니다.  RAG는 검색된 정보를 LM의 입력에 추가하여 생성 품질을 향상시키는 방법입니다. 또한, 긴 문서를 요약하여 모델의 입력으로 사용하는 방법도 제안되었습니다. 이러한 연구들은 주로 경험적 접근에 의존하며, 시스템 설계의 체계적인 이해를 제공하지는 못했습니다.

이 논문은 기존 연구와 달리 정보 이론적 관점을 도입하여, 압축기와 예측기의 조합이 시스템 성능에 미치는 영향을 체계적으로 분석합니다. 상호 정보를 활용하여 압축 품질을 평가하고, 이를 통해 다양한 데이터셋에서의 성능을 예측하는 방법론을 제시합니다. 이는 기존 연구와 차별화되는 접근 방식으로, 에이전트 시스템의 설계에 있어 새로운 가능성을 제시합니다. 특히, 정보 이론적 관점은 압축 과정에서 정보 손실을 최소화하고, 예측기에 필요한 핵심 정보만을 전달하는 데 초점을 맞춥니다.

## 제안하는 방법론
이 논문은 에이전트 언어 모델 시스템의 설계에 있어 정보 이론적 접근을 제안합니다. 핵심 아이디어는 압축기를 잡음 채널로 보고, 문맥과 압축된 정보 간의 상호 정보를 추정하여 압축 품질을 평가하는 것입니다. 상호 정보는 특정 작업에 의존하지 않고 압축 품질을 예측할 수 있는 지표로 사용됩니다. 이를 통해 다양한 데이터셋과 모델 패밀리에서 압축기와 예측기의 성능을 분석할 수 있습니다.

### 모델 아키텍처 구조
에이전트 시스템은 작은 "압축기" 모델과 큰 "예측기" 모델로 구성됩니다. 압축기 모델은 원시 문맥을 요약하여, 예측기 모델이 이를 소비할 수 있도록 합니다. 이러한 구조는 긴 문맥을 효과적으로 처리하고 계산 비용을 줄이는 데 도움이 됩니다. 예를 들어, 압축기는 BERT와 같은 모델을 사용할 수 있으며, 예측기는 GPT와 같은 생성 모델을 사용할 수 있습니다.

### 핵심 수식과 알고리즘 설명
상호 정보 $I(X;Z)$를 추정하기 위해 KL 발산을 사용합니다. 이는 문맥 $X$와 압축된 정보 $Z$ 간의 정보량을 측정합니다. 수식은 다음과 같습니다:

$$ I(X;Z) = D_{KL}(p(x,z) \parallel p(x)p(z)) = \mathbb{E}_{x,z \sim p(x,z)} \left[ \log \frac{p(z|x)}{p(z)} \right] $$

여기서 $p(x,z)$는 $X$와 $Z$의 결합 확률 분포이고, $p(x)$와 $p(z)$는 각각 $X$와 $Z$의 주변 확률 분포입니다. $p(z|x)$는 문맥 $X$가 주어졌을 때 압축된 정보 $Z$의 조건부 확률 분포를 나타냅니다. KL 발산은 두 확률 분포의 차이를 측정하는 데 사용되며, 상호 정보는 이 차이를 기대값으로 나타낸 것입니다.  상호 정보가 높을수록 압축된 정보가 원본 문맥을 잘 보존하고 있다는 의미입니다.

### Python/PyTorch 코드 예제
다음은 상호 정보를 추정하는 Python 코드 예제입니다.  이 예제는 이산적인 데이터에 대한 상호 정보 추정 방법을 보여줍니다.

```python
import numpy as np
from scipy.stats import entropy

def mutual_information(X, Z, bins=10):
    """
    문맥 X와 압축된 정보 Z 사이의 상호 정보를 추정합니다.

    Args:
        X: 원본 문맥 데이터 (numpy array).
        Z: 압축된 문맥 데이터 (numpy array).
        bins: 히스토그램을 계산할 때 사용할 bin의 개수.

    Returns:
        상호 정보 값.
    """
    # 결합 히스토그램 계산
    joint_histogram, _, _ = np.histogram2d(X, Z, bins=bins)
    joint_histogram /= np.sum(joint_histogram)  # 정규화

    # 주변 히스토그램 계산
    marginal_X = np.sum(joint_histogram, axis=1)
    marginal_Z = np.sum(joint_histogram, axis=0)

    # 상호 정보 계산
    mutual_info = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_histogram[i, j] > 0 and marginal_X[i] > 0 and marginal_Z[j] > 0:
                mutual_info += joint_histogram[i, j] * np.log2(
                    joint_histogram[i, j] / (marginal_X[i] * marginal_Z[j])
                )

    return mutual_info

# 예시 데이터 생성
X = np.random.rand(1000)
Z = X + np.random.normal(0, 0.1, 1000)  # X에 약간의 노이즈를 추가하여 Z 생성

# 상호 정보 계산
mi = mutual_information(X, Z)
print(f"Estimated Mutual Information: {mi}")
```

**참고:** 실제 언어 모델에서는 데이터가 연속적이므로, 위 코드는 직접적으로 적용하기 어렵습니다.  실제 LM에서는 확률 분포를 추정하기 위해 커널 밀도 추정(Kernel Density Estimation, KDE)과 같은 방법을 사용하거나, 신경망을 사용하여 $p(z|x)$를 직접 모델링하는 방법을 사용할 수 있습니다.  PyTorch를 사용하여 신경망을 학습시키는 예제는 다음과 같습니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ConditionalProbabilityModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ConditionalProbabilityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1) # 확률 분포를 만들기 위해 softmax 사용

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 예시 데이터
X = torch.randn(100, 10) # 100개의 문맥 데이터, 각 데이터는 10차원
Z = torch.randn(100, 5)  # 100개의 압축된 데이터, 각 데이터는 5차원

# 모델 초기화
input_size = 10
hidden_size = 20
output_size = 5
model = ConditionalProbabilityModel(input_size, hidden_size, output_size)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss() # 예시로 CrossEntropyLoss 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    # Z를 one-hot encoding으로 변환 (CrossEntropyLoss 사용을 위해)
    Z_indices = torch.argmax(Z, dim=1) # Z에서 가장 큰 값의 인덱스를 가져옴
    loss = criterion(outputs, Z_indices)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 학습된 모델을 사용하여 p(z|x)를 추정하고, 이를 바탕으로 상호 정보 계산
# (실제 상호 정보 계산은 추가적인 구현이 필요)
```

## 실험 설정
### 데이터셋 설명
실험은 LongHealth, FinanceBench, QASPER, WildChat, FineWeb 등 5개의 데이터셋을 사용하여 수행되었습니다. 이러한 데이터셋은 다양한 길이와 특성을 가진 텍스트 데이터를 포함하고 있어, 에이전트 시스템의 성능을 종합적으로 평가하는 데 적합합니다. 예를 들어, LongHealth는 긴 의료 기록 데이터를 포함하고 있으며, FinanceBench는 금융 관련 텍스트 데이터를 포함합니다. QASPER는 학술 논문 QA 데이터셋이며, WildChat은 다양한 주제의 대화 데이터셋입니다. FineWeb은 웹 문서 데이터셋입니다.

### 평가 지표
압축기와 예측기의 성능을 평가하기 위해 상호 정보와 다운스트림 질의응답(QA) 정확도를 주요 지표로 사용했습니다. 상호 정보는 압축 품질을 평가하는 데 사용되며, QA 정확도는 시스템의 실제 성능을 평가하는 데 사용됩니다. 또한, 토큰 효율성(압축률)도 함께 평가하여, 압축 성능과 QA 정확도 간의 균형을 분석했습니다.

### 비교 대상 (baseline)
기존의 압축기-예측기 구조와의 비교를 통해 제안된 방법론의 효과를 평가했습니다. 특히, 서로 다른 크기의 모델을 사용하여 압축을 수행하고, 각 압축 결과에 대한 성능을 비교했습니다.  예를 들어, 압축기를 사용하지 않는 경우(원시 문맥을 그대로 사용하는 경우)와, 간단한 요약 모델을 사용하는 경우를 비교했습니다.

### 하이퍼파라미터 설정
모든 실험에서 하이퍼파라미터는 동일하게 설정하여, 모델의 크기 변화에 따른 성능 변화를 분석했습니다. 압축기의 크기는 1.5B에서 7B까지 다양하게 설정되었으며, 예측기의 크기는 고정되었습니다.  학습률, 배치 크기, 에폭 수 등의 하이퍼파라미터는 모든 실험에서 동일하게 유지되었습니다.

## 실험 결과 및 분석
### 주요 정량적 결과
큰 압축기는 더 높은 정확도와 토큰 효율성을 보였으며, 압축기 크기를 키우는 것이 예측기 크기를 키우는 것보다 성능 향상에 더 효과적임을 발견했습니다. 예를 들어, Qwen-2.5 압축기를 1.5B에서 7B로 확장할 경우 정확도가 60% 향상되었습니다. 이는 더 큰 모델이 더 많은 정보를 효과적으로 압축하고, 예측기가 더 나은 성능을 발휘할 수 있도록 돕는다는 것을 의미합니다. 또한, 상호 정보 값이 높은 압축기가 더 높은 QA 정확도를 보이는 경향을 확인했습니다.

### 정성적 분석
상호 정보율은 다운스트림 성능과 강한 상관관계를 보였으며, 이는 시스템 성능을 예측하는 실용적인 지표로 사용될 수 있습니다. 상호 정보 값이 높을수록 압축된 정보가 원본 문맥을 잘 반영하고 있다는 의미이며, 이는 예측기의 성능 향상으로 이어질 수 있습니다. 예를 들어, 특정 데이터셋에서 상호 정보 값이 낮은 압축기는 원본 문맥의 중요한 정보를 누락시키는 경향이 있었습니다.

### Ablation study 결과
압축기의 크기와 예측기의 크기를 독립적으로 변화시켜 성능을 분석한 결과, 압축기의 크기를 늘리는 것이 예측기의 크기를 늘리는 것보다 더 큰 성능 향상을 가져오는 것으로 나타났습니다. 이는 압축기의 설계가 시스템 성능에 중요한 영향을 미친다는 것을 시사합니다. 특히, 압축기의 크기가 작을 경우, 예측기의 크기를 아무리 늘려도 성능 향상이 제한적이었습니다.

## 한계점 및 향후 연구 방향
이 연구는 압축기와 예측기의 조합이 시스템 성능에 미치는 영향을 체계적으로 분석했지만, 여전히 몇 가지 한계점이 존재합니다. 첫째, 압축기의 실패 모드를 완전히 해결하지 못했습니다. 부정확한 답변이나 정보 누락 등의 문제는 여전히 존재합니다. 둘째, 다양한 유형의 연구 질문에 대한 방법론의 일반화 가능성을 탐구할 필요가 있습니다. 셋째, 상호 정보 추정 방법의 정확도를 높이는 연구가 필요합니다.

향후 연구에서는 압축 모델의 실패 모드를 완화하고, 다양한 유형의 연구 질문에 대한 방법론의 일반화 가능성을 탐구하는 데 초점을 맞출 수 있습니다. 또한, 상호 정보 기반의 최적화 방법을 개발하여 에이전트 시스템의 성능을 더욱 향상시키는 데 집중할 필요가 있습니다.  예를 들어, 강화 학습을 사용하여 상호 정보를 최대화하는 압축 정책을 학습할 수 있습니다.  또한, 압축 과정에서 정보 손실을 최소화하기 위해 contrastive learning과 같은 방법을 적용할 수 있습니다.

## 결론 및 시사점
이 연구는 압축기-예측기 시스템의 설계에 있어 정보 이론적 접근이 유용함을 보여주며, 에이전트 시스템의 효율성을 높이는 데 기여할 수 있는 방법론을 제시합니다. 상호 정보는 압축 품질을 평가하고 시스템 성능을 예측하는 데 유용한 지표로 활용될 수 있으며, 다양한 데이터셋과 모델 패밀리에서 압축기와 예측기의 성능을 분석하는 데 도움이 될 수 있습니다. 향후 연구에서는 다양한 압축 알고리즘과 모델 아키텍처를 탐색하고, 상호 정보 기반의 최적화 방법을 개발하여 에이전트 시스템의 성능을 더욱 향상시키는 데 집중할 필요가 있습니다.

이 연구는 실무 적용 가능성이 높으며, 다양한 응용 분야에서의 성능을 개선하는 데 기여할 수 있습니다. 특히, 압축기와 예측기의 최적의 조합을 찾는 데 있어 정보 이론적 접근은 유용한 도구가 될 수 있습니다. 개인적으로, 이 연구는 에이전트 시스템의 설계에 있어 새로운 가능성을 제시하며, 향후 연구의 방향성을 제시하는 데 중요한 기여를 했다고 생각합니다. 정보 이론적 접근은 에이전트 시스템의 설계 원리를 이해하고, 성능을 최적화하는 데 필수적인 도구가 될 것입니다.
```