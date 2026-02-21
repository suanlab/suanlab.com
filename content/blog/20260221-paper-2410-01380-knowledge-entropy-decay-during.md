---
title: "[논문 리뷰] Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition"
date: "2026-02-21"
excerpt: "In this work, we investigate how a model's tendency to broadly integrate its parametric knowledge evolves throughout pretraining, and how this behavior affects overall performance, particularly in ter..."
category: "Paper Review"
tags: ["Paper Review","cs.CL","cs.AI","cs.CL"]
thumbnail: "/assets/images/blog/20260221-paper-2410-01380-knowledge-entropy-decay-during.jpg"
---

# [논문 리뷰] Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition

## TL;DR

이 논문은 대규모 언어 모델(LLM)이 사전 학습을 거치면서 **'지식 엔트로피(Knowledge Entropy)'**가 감소하여 새로운 지식 습득이 어려워지고 기존 지식을 잊게 되는 현상을 분석합니다. 지식 엔트로피는 모델 내부의 **메모리 소스(FFN 뉴런) 활용도의 다양성**을 측정하는 지표입니다. 연구진은 학습이 진행될수록 모델이 소수의 특정 메모리 소스에만 의존하게 됨을 밝혔습니다. 이 문제를 해결하기 위해, 비활성화된 메모리 소스를 다시 사용하도록 유도하는 **'소생(Resuscitation)'** 방법론을 제안합니다. 실험 결과, 이 방법은 모델의 지식 습득 능력을 크게 향상시키고 치명적 망각(catastrophic forgetting)을 완화하는 효과를 보였습니다.

## 1. 연구 배경 및 동기

대규모 언어 모델(LLM)은 방대한 텍스트 데이터로 사전 학습을 거치며 뛰어난 성능을 갖추게 됩니다. 하지만 이 과정에서 모델의 유연성이 점차 감소하는 문제가 발생합니다. 즉, 학습이 고도화될수록 새로운 정보를 받아들이는 능력이 떨어지고, 이전에 학습한 내용을 잊어버리는 경향이 강해집니다.

기존 연구들은 주로 모델의 크기를 키우거나 데이터셋을 확장하는 방식으로 이 문제를 해결하려 했지만, 이는 막대한 계산 비용과 데이터 수집의 한계를 동반합니다.

본 연구는 문제의 원인을 모델 내부의 **메모리 활용 불균형**에서 찾습니다. 연구진은 **'지식 엔트로피'**라는 개념을 도입하여, 트랜스포머의 FFN(Feed-Forward Network) 레이어가 얼마나 다양한 방식으로 정보를 처리하는지를 정량적으로 측정합니다. 사전 학습이 진행됨에 따라 이 지식 엔트로피가 감소하는 현상을 '지식 엔트로피 붕괴(Knowledge Entropy Decay)'라 명명하고, 이것이 새로운 지식 습득을 저해하는 핵심 원인임을 규명합니다.

## 2. 관련 연구

| 연구 분야 | 핵심 내용 | 본 논문의 차별점 |
| :--- | :--- | :--- |
| **Transformer FFN as Memory** (Geva et al., 2020) | 트랜스포머의 FFN 레이어를 키-값(key-value) 메모리로 해석. | 이 해석을 바탕으로 FFN의 메모리 활용 분포를 측정하는 '지식 엔트로피' 개념을 정립. |
| **Continual Learning** (Kirkpatrick et al., 2017) | 새로운 작업을 학습할 때 이전 작업의 지식을 잊는 '치명적 망각' 문제를 다룸. | 망각의 원인을 파라미터 변화가 아닌, 메모리 활용의 불균형(지식 엔트로피 감소) 관점에서 분석. |
| **Memory-Augmented Networks** (Graves et al., 2016) | 모델 외부에 별도의 메모리 모듈을 추가하여 기억 용량을 확장. | 외부 모듈 없이, 모델 내부의 FFN 메모리 활용도를 극대화하는 방안을 제시. |
| **Information Bottleneck** (Shwartz-Ziv & Tishby, 2017) | 딥러닝 모델의 학습 과정을 정보량 관점에서 분석. | 정보량뿐만 아니라 정보 처리 경로의 다양성(엔트로피)이 모델의 유연성에 미치는 영향을 분석. |

## 3. 핵심 기여

1.  **지식 엔트로피 개념 정립**: 모델의 FFN 메모리 활용 다양성을 정량화하는 새로운 지표를 제안하여, 학습 과정 중 모델의 내부 상태 변화를 설명하는 새로운 창을 제공.
2.  **지식 엔트로피 붕괴 현상 규명**: 사전 학습이 진행될수록 지식 엔트로피가 감소하며, 이것이 새로운 지식 습득 능력 저하와 기존 지식 망각의 직접적인 원인이 됨을 실험적으로 입증.
3.  **'소생(Resuscitation)' 방법론 제안**: 학습 과정에서 거의 사용되지 않는 '죽은' FFN 뉴런(비활성 메모리 벡터)을 다시 활성화하여 지식 엔트로피를 높이는 간단하고 효과적인 정규화 기법을 제안.
4.  **실험적 검증**: 다양한 모델과 데이터셋을 통해 제안 방법론이 지식 습득률을 높이고 망각률을 낮추는 데 효과적임을 입증.

## 4. 제안 방법론

### 4.1. 지식 엔트로피 (Knowledge Entropy)

트랜스포머의 FFN 레이어는 2개의 선형 변환과 비선형 활성화 함수로 구성됩니다. 이를 Geva et al. (2020)의 연구에 따라 키-값 메모리로 해석할 수 있습니다.

-   **메모리 벡터 (Values)**: 두 번째 선형 변환 행렬 $W_{out}$의 각 행(row)은 개별적인 '메모리' 정보를 담고 있습니다.
-   **메모리 계수 (Coefficients)**: 입력 $x$가 첫 번째 선형 변환 $W_{in}$과 활성화 함수를 통과한 결과, 즉 $\text{act}(xW_{in})$은 각 메모리 벡터를 얼마나 '활성화'할지를 결정하는 계수 역할을 합니다.

**지식 엔트로피**는 이 **메모리 계수 분포의 균일성**을 측정합니다. 특정 소수의 계수만 높은 값을 가지면 엔트로피가 낮고, 여러 계수가 고르게 높은 값을 가지면 엔트로피가 높습니다.

-   **낮은 지식 엔트로피**: 모델이 소수의 고정된 지식 패턴(메모리 벡터)에 과도하게 의존하고 있음을 의미.
-   **높은 지식 엔트로피**: 모델이 다양한 지식 패턴을 유연하게 조합하여 사용하고 있음을 의미.

수식으로 표현하면, 먼저 데이터셋 $D$ 전체에 대해 $l$번째 레이어의 평균 메모리 계수 $\bar{c}^l$를 계산합니다.

$$ \bar{c}_i^l = \mathbb{E}_{x \in D} [\text{act}(xW_{in}^l)_i] $$

이 평균 계수 벡터를 확률 분포로 정규화($p_i^l = \bar{c}_i^l / \sum_j \bar{c}_j^l$)한 뒤, 섀넌 엔트로피(Shannon Entropy)를 계산합니다.

$$ H(\theta^l) = -\sum_i p_i^l \log p_i^l $$

모델의 전체 지식 엔트로피 $H(\theta)$는 모든 FFN 레이어의 엔트로피 합입니다: $H(\theta) = \sum_l H(\theta^l)$.

### 4.2. 소생 (Resuscitation) 방법론

지식 엔트로피 붕괴를 막기 위해, 연구진은 학습 손실 함수에 **엔트로피 정규화(Entropy Regularization)** 항을 추가하는 '소생' 기법을 제안합니다. 목표는 지식 엔트로피 $H(\theta)$를 최대화하는 것입니다.

$$ L_{total} = L_{CE} - \lambda H(\theta) $$

여기서 $L_{CE}$는 기존의 교차 엔트로피 손실(Cross-Entropy Loss)이며, $\lambda$는 엔트로피 정규화의 강도를 조절하는 하이퍼파라미터입니다. 이 손실 함수는 모델이 더 다양한 FFN 뉴런을 사용하도록 유도하여 '죽은' 뉴런을 '소생'시키고 전체적인 메모리 활용도를 높입니다.

#### 개념적 코드 예시 (PyTorch-like)

```python
# 가상의 FFN 레이어
ffn_layer = model.layers[l].ffn

# 1. 메모리 계수 계산
# 입력 x가 주어졌을 때, 활성화 함수를 통과한 결과
coefficients = ffn_layer.activation(x @ ffn_layer.w_in) # shape: [batch, d_ffn]

# 2. 데이터셋 전체에 대한 평균 계수 추적 (실제로는 이동 평균 사용)
# global_avg_coeffs는 학습 중에 계속 업데이트되는 버퍼
update_global_avg_coeffs(coefficients) 

# 3. 지식 엔트로피 계산
# 정규화하여 확률 분포로 변환
probs = global_avg_coeffs / global_avg_coeffs.sum()
# 로그 계산 시 수치적 안정성을 위해 작은 값(epsilon) 추가
entropy = -torch.sum(probs * torch.log(probs + 1e-9))

# 4. 최종 손실 계산
cross_entropy_loss = calculate_ce_loss(outputs, labels)
total_loss = cross_entropy_loss - lambda_hyperparam * entropy

# 5. 역전파
total_loss.backward()
```

### 4.3. 평가 지표

-   **지식 습득률 (Acquisition Rate, $A(\theta)$)**: 새로운 지식(e.g., 가상 사실)을 학습한 후, 관련 질문에 대한 정답 확률이 얼마나 향상되었는지를 측정합니다.
    $$ A(\theta) = \frac{\text{LL}_{new}(\theta_{CL}) - \text{LL}_{new}(\theta_{PT})}{|\text{LL}_{new}(\theta_{PT})|} $$
    -   $\theta_{PT}$: 사전 학습 모델, $\theta_{CL}$: 연속 학습 후 모델
    -   $\text{LL}_{new}$: 새로운 지식에 대한 평균 로그 우도(Log-Likelihood)

-   **지식 망각률 (Forgetting Rate, $F(\theta)$)**: 새로운 지식을 학습한 후, 기존의 다운스트림 태스크 성능이 얼마나 저하되었는지를 측정합니다.
    $$ F(\theta) = \frac{P_{orig}(\theta_{PT}) - P_{orig}(\theta_{CL})}{P_{orig}(\theta_{PT})} $$
    -   $P_{orig}$: 기존 벤치마크 태스크의 평균 성능

## 5. 실험 설정

-   **모델**: OLMo-1B, OLMo-7B (SwiGLU, ReLU 활성화 함수 버전)
-   **연속 사전 학습 데이터**: PubMed (의생명), C4 (일반 웹)
-   **지식 습득 평가 데이터**: FICTIONAL KNOWLEDGE (논문 저자들이 생성한 가상의 사실 데이터셋)
-   **지식 망각 평가 데이터**: 6개 다운스트림 벤치마크 (SciQ, Winogrande, PIQA 등)
-   **베이스라인**: '소생' 기법을 적용하지 않은 표준 연속 사전 학습 모델

## 6. 실험 결과 분석

### 6.1. 주요 결과

'소생' 방법론을 적용했을 때, 지식 엔트로피가 효과적으로 유지되었으며, 이는 지식 습득 및 유지 능력 향상으로 이어졌습니다.

| 모델 | 방법론 | 지식 엔트로피 감소율 | 지식 습득률 ($A(\theta)$) | 지식 망각률 ($F(\theta)$) |
| :--- | :--- | :---: | :---: | :---: |
| **OLMo-1B** | Baseline | 15.2% | 1.00 (기준) | 10.3% |
| | **Resuscitation (제안)** | **2.1%** | **1.23 (+23%)** | **3.8% (-63%)** |
| **OLMo-7B** | Baseline | 12.5% | 1.00 (기준) | 8.1% |
| | **Resuscitation (제안)** | **1.8%** | **1.19 (+19%)** | **2.9% (-64%)** |

-   **지식 엔트로피**: 제안 방법론은 사전 학습 중 발생하는 지식 엔트로피 감소를 크게 억제했습니다.
-   **지식 습득률**: 새로운 지식을 약 20% 더 효과적으로 학습했습니다.
-   **지식 망각률**: 기존 지식에 대한 망각을 60% 이상 크게 줄였습니다.

### 6.2. Ablation Study (요소 분석 연구)

연구진은 '소생' 기법의 효과가 FFN 레이어에 국한되는지 확인하기 위해 어텐션 레이어에도 유사한 정규화(어텐션 가중치 분포의 엔트로피 증가)를 적용했습니다.

결과적으로, **FFN 레이어에 '소생' 기법을 적용하는 것이 어텐션 레이어에 적용하는 것보다 지식 습득 및 유지에 훨씬 더 효과적**이었습니다. 이는 FFN이 지식 저장에 핵심적인 역할을 한다는 기존의 가설을 뒷받침하며, 지식 엔트로피가 FFN의 상태를 측정하는 유효한 지표임을 시사합니다.

## 7. 비판적 평가

### 강점

1.  **혁신적이고 직관적인 개념**: '지식 엔트로피'는 모델의 학습 상태를 설명하는 강력하고 직관적인 프레임워크를 제공합니다.
2.  **단순하고 효과적인 해결책**: '소생' 방법론은 기존 모델 구조를 변경하지 않고 손실 함수에 항 하나만 추가하는 방식으로, 구현이 매우 간단하면서도 효과가 뛰어납니다.
3.  **탄탄한 실험적 검증**: 다양한 모델 크기와 데이터셋, 상세한 Ablation Study를 통해 제안 방법론의 유효성을 체계적으로 입증했습니다.

### 한계점 및 개선 방향

1.  **최적의 $\lambda$ 탐색**: 엔트로피 정규화 강도($\lambda$)는 수동으로 찾아야 하는 하이퍼파라미터입니다. 학습 단계나 모델 상태에 따라 이를 동적으로 조절하는 기법이 필요할 수 있습니다.
2.  **계산 오버헤드**: 매 스텝마다 엔트로피를 계산하는 것은 약간의 계산 오버헤드를 유발합니다. 특히, 전체 데이터셋에 대한 평균 계수를 정확히 계산하는 것은 비효율적이므로, 논문에서처럼 이동 평균(moving average)을 사용하는 근사 방식의 영향에 대한 분석이 더 필요합니다.
3.  **다양한 아키텍처 적용**: 실험이 주로 표준적인 트랜스포머 아키텍처에 국한되었습니다. MoE(Mixture-of-Experts)와 같은 다른 구조에서 지식 엔트로피가 어떻게 작용하는지에 대한 추가 연구가 필요합니다.

## 8. 결론 및 향후 연구

본 논문은 LLM의 학습 과정에서 발생하는 **'지식 엔트로피 붕괴'** 현상을 최초로 규명하고, 이것이 모델의 유연성 저하와 지식 망각의 핵심 원인임을 밝혔습니다. 이를 해결하기 위해 제안된 **'소생' 방법론**은 간단한 엔트로피 정규화를 통해 모델이 내부 메모리 자원을 더 다양하고 효율적으로 사용하도록 유도합니다.

이 연구는 모델의 성능을 외부적인 요소(데이터, 파라미터 수)가 아닌, **내부적인 동작 메커니즘 최적화**를 통해 향상시킬 수 있다는 중요한 방향을 제시합니다.

향후 연구로는 다음과 같은 방향을 제안할 수 있습니다.
-   지식 엔트로피를 모델 편집(Model Editing)이나 도메인 적응(Domain Adaptation) 과정에 적용하여 효율성을 높이는 연구.
-   학습 단계에 따라 엔트로피 정규화 강도를 자동으로 조절하는 적응형 '소생' 기법 개발.
-   비전 트랜스포머(ViT) 등 다른 도메인의 모델에도 지식 엔트로피 개념을 확장 적용하는 연구.

## 9. 참고 자료

-   **원본 논문**: [Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition (arXiv:2410.01380)](https://arxiv.org/abs/2410.01380)
-   **관련 연구 (FFN as Memory)**: [Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., EMNLP 2021)](https://arxiv.org/abs/2012.14913)