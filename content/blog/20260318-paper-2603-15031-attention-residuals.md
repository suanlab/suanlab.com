---
title: "[논문 리뷰] Attention Residuals"
date: "2026-03-18"
excerpt: "Residual connections with PreNorm are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with dept..."
category: "Paper Review"
tags: ["Paper Review","cs.CL","cs.CL"]
thumbnail: "/assets/images/blog/20260318-paper-2603-15031-attention-residuals.jpg"
---

# [논문 리뷰] Attention Residuals: 잔차 연결을 넘어서 동적 정보 흐름으로

## TL;DR

현대의 대형 언어 모델(LLM)은 수많은 레이어를 쌓기 위해 잔차 연결(Residual Connection)에 의존하지만, 이는 모든 이전 정보가 동일한 가중치로 더해져 중요한 초기 정보가 희석되는 '정보 희석(Information Dilution)' 문제를 야기합니다. "Attention Residuals (AttnRes)"는 이 문제를 해결하기 위해, 고정된 덧셈을 동적인 어텐션 메커니즘으로 대체하는 새로운 아키텍처를 제안합니다. 각 레이어는 이전 모든 레이어의 출력에 어텐션을 적용하여 현재 계산에 가장 유용한 정보를 가중합으로 가져옵니다. 또한, 메모리 효율성을 높인 'Block AttnRes' 변형을 통해 대규모 모델에도 실용적으로 적용할 수 있습니다. 실험 결과, AttnRes는 모든 모델 크기에서 성능을 일관되게 향상시켰으며, 특히 여러 단계의 추론이 필요한 수학 문제 풀이 작업에서 뛰어난 성능을 보였습니다.

## 연구 배경 및 동기

딥러닝 모델, 특히 LLM은 수백, 수천 개의 레이어를 쌓아 깊이를 확보함으로써 성능을 높여왔습니다. 이러한 깊은 구조의 학습을 가능하게 한 핵심 기술은 바로 **잔차 연결(Residual Connection)**입니다. 잔차 연결은 각 레이어의 출력을 이전 레이어의 입력에 더해줌으로써($x_{l+1} = x_l + F(x_l)$), 기울기 소실(Vanishing Gradient) 문제를 완화하고 정보의 흐름을 원활하게 만듭니다.

하지만 모델이 극도로 깊어지면서 잔차 연결의 근본적인 한계가 드러납니다. 모든 레이어의 출력이 동일한 가중치(1)로 계속 더해지기 때문에, 초기 레이어의 중요한 정보(예: 문제의 핵심 조건)가 후반 레이어로 갈수록 다른 정보들에 섞여 희석되는 **정보 희석(Information Dilution)** 문제가 발생합니다. 마치 여러 사람이 계속해서 같은 물감 통에 각자 다른 색을 한 방울씩 떨어뜨리는 것과 같습니다. 처음의 중요한 색은 점차 그 정체성을 잃게 됩니다.

이러한 문제를 해결하기 위해, 각 레이어가 과거의 정보를 무비판적으로 더하는 대신, 필요한 정보를 선별적으로, 그리고 중요도에 따라 가중치를 두어 가져올 수 있는 새로운 메커니즘이 필요합니다. "Attention Residuals (AttnRes)"는 바로 이 지점에서 출발합니다.

## 관련 연구

잔차 연결은 He et al.의 ResNet에서 처음 제안된 이후, 딥러닝 아키텍처의 표준으로 자리 잡았습니다. 트랜스포머(Transformer) 역시 셀프 어텐션과 피드포워드 네트워크 블록마다 잔차 연결을 사용하여 깊은 모델의 학습을 안정화했습니다.

그러나 깊이 방향의 정보 흐름을 개선하려는 시도는 계속되어 왔습니다. 일부 연구는 가중치를 학습하여 잔차 연결을 조절하거나(Weighted Residuals), 여러 경로를 통해 정보를 전달하는 DenseNet과 같은 구조를 제안했습니다. 하지만 이러한 방법들도 깊이가 매우 깊어질 때 정보 흐름을 동적으로 최적화하는 데에는 한계가 있었습니다.

AttnRes는 깊이 방향의 정보 흐름 자체에 **어텐션 메커니즘**을 적용했다는 점에서 기존 연구들과 차별화됩니다. 이는 각 레이어가 전체 계산 히스토리에서 가장 관련성 높은 정보를 직접 참조할 수 있게 하여, 훨씬 더 유연하고 강력한 정보 전송 경로를 구축합니다. 특히, 제안된 Block AttnRes는 계산 효율성을 확보하여 이 아이디어를 수십억 파라미터 규모의 LLM에 실용적으로 적용할 수 있게 만들었다는 점에서 중요한 의의를 가집니다.

## 핵심 기여

1.  **정보 희석 문제 해결**: 고정된 덧셈 기반의 잔차 연결을 동적 가중합 기반의 어텐션으로 대체하여, 깊은 네트워크에서의 정보 희석 문제를 근본적으로 해결합니다.
2.  **계산 효율성을 갖춘 아키텍처**: 전체 레이어에 어텐션을 적용하는 Full AttnRes의 개념을 제시하고, 이를 블록 단위로 근사하여 계산량과 메모리 사용량을 크게 줄인 **Block AttnRes**를 제안함으로써 대규모 모델에 즉시 적용 가능한 실용성을 확보했습니다.
3.  **다단계 추론 능력의显著한 향상**: AttnRes는 특히 GSM8K와 같은 수학 문제 풀이 벤치마크에서 큰 성능 향상을 보였습니다. 이는 여러 단계에 걸쳐 논리적 추론을 수행하고 초기 정보를 유지해야 하는 작업에서 AttnRes의 강점이 발휘됨을 시사합니다.
4.  **새로운 LLM 아키텍처의 가능성 제시**: 잔차 연결이라는 기존 패러다임을 넘어, 깊이 방향의 정보 흐름을 최적화하는 새로운 방향을 제시하여 향후 LLM 아키텍처 연구에 중요한 영감을 줍니다.

## 제안 방법론

AttnRes의 핵심은 기존의 잔차 연결을 깊이 방향의 어텐션(depth-wise attention)으로 대체하는 것입니다.

*   **기존 잔차 연결**: $(l+1)$번째 레이어의 입력은 이전 레이어의 출력 $x_l$에 현재 레이어의 연산 결과 $F(x_l)$을 더한 것입니다.
    $$x_{l+1} = x_l + F(x_l)$$
*   **AttnRes**: $(l+1)$번째 레이어의 입력은 $0$번째부터 $l$번째까지의 모든 이전 레이어 출력 $(x_0, x_1, \dots, x_l)$을 동적으로 가중합하여 만듭니다.
    $$x'_{l+1} = \sum_{i=0}^{l} \alpha_{i \to l+1} \cdot x_i$$
    $$x_{l+1} = F(x'_{l+1})$$
    여기서 $F$는 트랜스포머 블록(Self-Attention + FFN)이며, $\alpha$는 어텐션 가중치입니다. 즉, 덧셈 기반의 스킵 커넥션이 사라지고, 어텐션을 통해 계산된 새로운 입력이 바로 다음 블록으로 전달됩니다.

### 모델 아키텍처 상세 설명

AttnRes는 두 가지 변형으로 제안됩니다.

1.  **Full Attention Residuals**: 가장 직관적인 방식으로, 각 레이어($l$)는 자신보다 앞선 모든 레이어($0$부터 $l-1$까지)의 출력($x_0, x_1, \dots, x_{l-1}$)에 대해 직접 어텐션을 수행합니다. 모든 과거 정보에 직접 접근하여 최적의 정보 조합을 찾을 수 있지만, 레이어 깊이($L$)에 대해 메모리 및 계산량이 $O(L^2)$로 증가하여 매우 깊은 모델에는 비효율적입니다.

2.  **Block Attention Residuals**: 계산 효율성을 높이기 위한 실용적인 근사 방식입니다. 전체 $L$개의 레이어를 크기가 $B$인 여러 개의 블록으로 나눕니다. 어텐션은 두 단계로 이루어집니다.
    *   **블록 내(Intra-block)**: 각 블록 내에서는 표준 잔차 연결(덧셈)을 통해 정보를 집계하여 블록의 마지막에 하나의 **대표 표현(representative representation)**을 만듭니다.
    *   **블록 간(Inter-block)**: 현재 레이어는 이전 블록들의 대표 표현들과 초기 토큰 임베딩($x_0$)에 대해서만 어텐션을 수행합니다.
    이를 통해 어텐션 대상의 수를 $L$에서 $L/B$ 수준으로 크게 줄여 메모리 사용량과 계산량을 절감하면서도, Full AttnRes의 핵심 장점인 장거리 정보 접근 능력을 대부분 유지합니다.

### 핵심 수식과 개념

AttnRes의 어텐션 가중치 $\alpha_{i \to l}$는 어떻게 계산될까요? 일반적인 어텐션과 유사하지만, 쿼리(Query)를 생성하는 방식에 차이가 있습니다. $l$번째 레이어의 출력이 아직 계산되지 않았기 때문에, 이를 쿼리로 사용할 수 없습니다. 대신, 각 레이어는 자신만의 학습 가능한 **의사 쿼리(pseudo-query)** 벡터 $w_l$을 가집니다. 이 $w_l$은 "$l$번째 레이어가 이전 정보들로부터 무엇을 필요로 하는가?"를 나타내는 '의도' 벡터로 학습됩니다.

$$
e_{i \to l} = \frac{(x_i W_K^l) (w_l W_Q^l)^T}{\sqrt{d}}
$$

$$
\alpha_{\cdot \to l} = \text{softmax}(e_{0 \to l}, e_{1 \to l}, \dots, e_{l-1 \to l})
$$

-   $w_l$: $l$번째 레이어를 위한 학습 가능한 쿼리 벡터.
-   $x_i$: $i$번째 레이어의 출력 (키와 값으로 사용됨).
-   $W_K^l, W_Q^l$: 키와 쿼리를 위한 프로젝션 행렬.
-   $d$: 헤드의 차원 수.

### 개념적 코드 비교

```python
# --- Standard Residual Connection in a Transformer Block ---
def standard_block(x_prev):
    # x_prev is output from layer l-1
    residual = x_prev
    x_processed = ffn(self_attention(x_prev))
    return residual + x_processed # Additive skip connection

# --- AttnRes in a Transformer Block (Conceptual) ---
def attnres_block(history, pseudo_query_l):
    # history is a list of all previous outputs [x_0, ..., x_{l-1}]
    
    # 1. Dynamically compute the input for this layer
    # Keys and Values are projections of history
    keys = key_projection(history)
    values = values_projection(history)
    
    # Query is a projection of the learned pseudo_query
    query = query_projection(pseudo_query_l)
    
    # Calculate attention weights and the new input
    scores = torch.matmul(query, keys.transpose(-1, -2))
    weights = torch.softmax(scores, dim=-1)
    x_input_l = torch.matmul(weights, values)
    
    # 2. Process this new input through the transformation
    # Note: No additive skip connection around the block
    return ffn(self_attention(x_input_l))
```

## 실험 설정

AttnRes의 성능을 검증하기 위해 125M부터 1.3B 파라미터 크기의 Pre-Norm 기반 디코더-온리 트랜스포머 모델에서 실험이 진행되었습니다. 비교 대상은 표준 잔차 연결을 사용하는 **Baseline**, **Full AttnRes**, 그리고 **Block AttnRes**입니다.

-   **데이터셋**: 대규모 텍스트 데이터셋(RedPajama)을 사용하여 언어 모델링 성능을 평가하고, GSM8K(수학 문제), MMLU(종합 지식) 등 다양한 다운스트림 벤치마크에서 추론 능력을 평가했습니다.
-   **평가 지표**: 모델의 일반화 성능은 주로 검증 손실(validation loss)로 측정되었으며, 다운스트림 작업에서는 정확도(accuracy)를 사용했습니다.

## 실험 결과 분석

실험 결과, AttnRes는 모든 모델 크기에서 Baseline보다 일관되게 더 낮은 검증 손실을 기록했습니다. 이는 AttnRes가 정보를 더 효율적으로 활용하여 언어 모델링 자체의 성능을 향상시킨다는 것을 의미합니다.

특히 주목할 점은 다음과 같습니다.

-   **Block AttnRes의 효율성**: Block AttnRes는 Full AttnRes와 거의 동등한 성능을 보이면서도 메모리 사용량과 계산 시간을 크게 줄여, 대규모 모델 학습에 매우 효과적임을 입증했습니다.
-   **다단계 추론에서의 강점**: GSM8K와 같은 수학 문제 풀이 벤치마크에서 성능 향상이 두드러졌습니다. 이는 문제의 초기 조건이나 중간 계산 결과를 마지막 단계까지 정확하게 전달하는 능력이 향상되었기 때문으로 분석됩니다. AttnRes가 필요한 정보를 동적으로 '불러오는' 능력이 복잡한 추론 과정에서 빛을 발한 것입니다.

### Ablation Study 분석

-   **어텐션 함수**: 가중치를 계산할 때 `softmax`가 `sigmoid`보다 더 나은 성능을 보였습니다. 이는 여러 정보 소스 간의 경쟁을 유도하여 더 명확한 정보 선택을 가능하게 하기 때문입니다.
-   **정규화의 중요성**: 키(key)에 `RMSNorm`을 적용하는 것이 학습 안정성과 성능에 매우 중요한 요소임이 밝혀졌습니다.
-   **멀티헤드 어텐션**: 채널(차원) 그룹별로 깊이 어텐션을 다르게 적용하는 멀티헤드 방식은 오히려 성능을 저하시켰습니다. 이는 최적의 깊이 방향 정보 흐름은 채널 전반에 걸쳐 일관되게 혼합되는 것이 더 효과적임을 시사합니다.

## 비판적 평가

### 강점

1.  **혁신적인 아이디어**: 잔차 연결이라는 당연시되던 구조에 의문을 제기하고, 동적 정보 흐름이라는 새로운 패러다임을 성공적으로 제시했습니다.
2.  **뛰어난 실용성**: Block AttnRes를 통해 아이디어를 계산적으로 효율적인 아키텍처로 구현하여, 실제 대규모 모델에 바로 적용할 수 있도록 했습니다.
3.  **명확한 성능 향상**: 특히 복잡한 추론 작업에서 뚜렷한 성능 향상을 보여, 모델의 지능을 한 단계 끌어올릴 잠재력을 입증했습니다.

### 한계점과 개선 방향

-   **파라미터 증가**: AttnRes는 의사 쿼리 벡터와 프로젝션 행렬 등 추가적인 파라미터를 필요로 합니다. 비록 전체 모델 크기에 비하면 미미하지만, 오버헤드가 존재합니다.
-   **Full AttnRes의 확장성**: Full AttnRes는 여전히 계산 복잡도가 높아 매우 깊은 모델(수천 레이어 이상)에서는 비효율적입니다. 더 효율적인 장거리 어텐션 메커니즘(예: 선형 어텐션)을 깊이 방향에 적용하는 연구가 필요할 수 있습니다.
-   **블록 대표 표현**: Block AttnRes에서 블록의 정보를 단순히 마지막 레이어의 출력으로 요약하는데, 더 정교한 요약(예: 풀링, 가중합) 방법을 탐구해볼 수 있습니다.

## 향후 연구 방향

1.  **다양한 아키텍처와의 결합**: AttnRes를 Mixture-of-Experts (MoE)나 다른 효율적인 트랜스포머 변형 아키텍처에 적용하여 시너지 효과를 탐색할 수 있습니다.
2.  **어텐션 메커니즘 고도화**: 깊이 방향 어텐션을 위한 더 효율적이고 강력한 메커니즘을 개발하여 계산 복잡도를 줄이고 성능을 극대화하는 연구가 가능합니다.
3.  **시각 및 멀티모달 모델로의 확장**: 정보 희석 문제는 언어 모델뿐만 아니라 매우 깊은 비전 모델이나 멀티모달 모델에서도 발생할 수 있습니다. AttnRes를 이러한 도메인에 적용하는 연구도 흥미로울 것입니다.

## 실무 적용 가이드

AttnRes를 실제 프로젝트에 적용할 때 고려할 점은 다음과 같습니다.

1.  **Block AttnRes부터 시작**: 대부분의 경우, 계산 효율성과 성능의 균형이 좋은 Block AttnRes를 우선적으로 고려하는 것이 좋습니다. 기존 트랜스포머 아키텍처의 잔차 연결을 대체하는 '드롭인(drop-in)' 방식으로 적용할 수 있습니다.
2.  **블록 크기 튜닝**: Block AttnRes의 블록 크기($B$)는 중요한 하이퍼파라미터입니다. 메모리 제약과 성능 사이의 트레이드오프를 고려하여 적절한 크기를 선택해야 합니다.
3.  **다단계 추론 작업에 우선 적용**: 만약 개발 중인 모델이 챗봇의 긴 대화 맥락 이해, 복잡한 문서 요약, 코드 생성, 수학 문제 풀이 등 여러 단계의 추론을 필요로 한다면, AttnRes 도입으로 인한 성능 향상 효과를 더 크게 기대할 수 있습니다.

## 결론

"Attention Residuals (AttnRes)"는 LLM 아키텍처의 핵심 요소인 잔차 연결을 혁신적으로 개선하여, 모델이 깊어질수록 발생하는 정보 희석 문제를 해결하는 강력한 대안을 제시합니다. 고정된 덧셈을 동적인 어텐션 기반의 정보 취합으로 대체함으로써, 모델은 각 계층에서 필요한 정보를 선별적으로 활용할 수 있게 되었습니다. 특히 Block AttnRes는 실용성까지 갖추어, 향후 LLM 아키텍처 설계에 중요한 영향을 미칠 것으로 기대됩니다. 이는 단순히 성능을 개선하는 것을 넘어, 모델이 정보를 처리하는 방식을 근본적으로 더 지능적으로 만드는 중요한 진일보입니다.

## 참고 자료

-   [논문 링크 (arXiv:2403.15031)](https://arxiv.org/abs/2403.15031)
-   [관련 블로그 포스트 및 자료](https://taog.github.io/project/attnres/)