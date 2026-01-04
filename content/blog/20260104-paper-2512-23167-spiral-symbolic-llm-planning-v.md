---
title: "[논문 리뷰] SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search"
date: "2026-01-04"
excerpt: "Large Language Models (LLMs) often falter at complex planning tasks that require exploration and self-correction, as their linear reasoning process struggles to recover from early mistakes. While sear..."
category: "Paper Review"
tags: ["Paper Review","cs.AI","cs.LG","cs.MA"]
thumbnail: "/assets/images/blog/20260104-paper-2512-23167-spiral-symbolic-llm-planning-v.jpg"
---

# [논문 리뷰] SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search

## TL;DR

대형 언어 모델(LLM)은 복잡한 계획 문제에서 초기 실수로부터 회복하기 어려운 경향이 있습니다. SPIRAL은 이러한 문제를 해결하기 위해 세 가지 특화된 LLM 에이전트(Planner, Simulator, Critic)를 활용한 새로운 프레임워크를 제안합니다. 이 프레임워크는 Monte Carlo Tree Search(MCTS)와 결합하여 LLM이 더 효과적으로 탐색하고, 자기 수정이 가능한 계획을 수립할 수 있도록 합니다. SPIRAL은 DailyLifeAPIs와 HuggingFace 데이터셋에서 기존 방법론보다 16% 이상 향상된 성능을 보였으며, 자원 효율성에서도 뛰어난 결과를 나타냈습니다. 이 연구는 LLM 기반의 자율 계획 능력을 향상시키고, 복잡한 작업 자동화를 위한 새로운 가능성을 제시합니다. 특히, 실제 API 호출을 시뮬레이션하는 환경에서 강점을 보입니다.

## 연구 배경 및 동기

대형 언어 모델(LLM)은 자연어 처리 분야에서 혁신적인 발전을 이끌어왔지만, 복잡한 계획 문제에서는 여전히 한계를 보이고 있습니다. 특히, LLM은 선형 추론 과정에서 초기 오류로부터 회복하기 어려운 경향이 있어, 복잡한 문제 해결에 있어 제약이 됩니다. 기존의 탐색 알고리즘인 Monte Carlo Tree Search(MCTS)는 탐색 공간을 효율적으로 탐색하는 데 유용하지만, LLM의 풍부한 의미적 능력을 충분히 활용하지 못하고 있습니다. 이러한 문제를 해결하기 위해 SPIRAL은 LLM의 추론 능력을 강화하고, MCTS와 결합하여 더 효과적인 탐색과 계획 수립을 가능하게 합니다. SPIRAL은 특히 Sparse Rewards 문제를 해결하기 위해 설계되었으며, 이는 기존의 MCTS가 직면한 주요 한계 중 하나입니다. 이 연구는 LLM의 의미적 능력과 MCTS의 탐색 능력을 결합하여 복잡한 계획 문제를 해결하는 데 중점을 두고 있습니다. 예를 들어, "아침 식사를 준비하고, 출근 준비를 하세요"와 같은 복합적인 목표를 달성하기 위한 일련의 행동들을 LLM이 효과적으로 계획하도록 돕습니다.

## 관련 연구

1. **GPT-3: Language Models are Few-Shot Learners** - Brown et al. (2020): GPT-3는 대형 언어 모델의 대표적인 예로, 자연어 처리에서 뛰어난 성능을 보였으나 복잡한 계획 문제에서는 제한적입니다.
2. **AlphaGo: Mastering the Game of Go with Deep Neural Networks and Tree Search** - Silver et al. (2016): MCTS와 신경망을 결합하여 바둑을 성공적으로 해결한 사례로, SPIRAL의 MCTS 활용에 영감을 주었습니다. AlphaGo의 성공은 MCTS가 복잡한 탐색 공간에서 효과적일 수 있음을 보여줍니다.
3. **Symbolic Planning with LLMs** - 최근 연구들은 LLM을 활용하여 심볼릭 계획 문제를 해결하려는 시도를 하고 있으며, SPIRAL은 이러한 연구의 연장선상에 있습니다. 이러한 연구들은 주로 PDDL(Planning Domain Definition Language)과 같은 형식 언어를 사용합니다.
4. **Reflective Agents in AI** - 반영적 에이전트 모델을 통한 자기 수정 가능성에 대한 연구로, SPIRAL의 Critic 에이전트 설계에 영향을 미쳤습니다.
5. **Hierarchical Reinforcement Learning** - Sutton et al. (1999): 계층적 강화 학습은 복잡한 문제를 해결하기 위해 계획을 세분화하는 방법으로, SPIRAL의 구조적 탐색에 기여하였습니다.

| 연구 | 차이점 |
|-----|-------|
| GPT-3 | 자연어 처리에 중점, 계획 문제 해결은 제한적 |
| AlphaGo | MCTS 활용, LLM과의 결합은 없음 |
| Symbolic Planning | LLM 활용, MCTS 결합은 미비 |
| Reflective Agents | 반영적 설계, LLM 기반은 아님 |
| Hierarchical RL | 계층적 계획, LLM 결합은 아님 |

## 핵심 기여

1. **새로운 프레임워크 제안**: SPIRAL은 LLM의 추론 능력을 강화하고, MCTS와 결합하여 복잡한 계획 문제를 해결하는 새로운 방법론을 제안합니다.
2. **통합된 에이전트 아키텍처**: Planner, Simulator, Critic 세 가지 특화된 LLM 에이전트를 통합하여, 탐색과 계획의 효율성을 높입니다.
3. **Sparse Rewards 문제 해결**: Critic 에이전트를 통해 밀도 높은 보상 신호를 제공하여, 기존 MCTS의 Sparse Rewards 문제를 해결합니다. 예를 들어, 목표 달성 여부 외에도 중간 단계의 성공에 대한 보상을 제공합니다.
4. **자원 효율성 향상**: SPIRAL은 기존 방법론보다 자원 효율성이 뛰어나, 제한된 자원 환경에서도 효과적으로 작동할 수 있습니다. 이는 LLM 추론 비용을 줄이는 데 기여합니다.

## 제안 방법론

SPIRAL은 LLM을 활용하여 복잡한 계획 문제를 해결하기 위한 새로운 프레임워크입니다. 이 프레임워크는 MCTS와 결합하여, 세 가지 특화된 에이전트(Planner, Simulator, Critic)를 통해 탐색과 계획 과정을 구조화합니다.

### 모델 아키텍처

1. **Planner**: LLM을 사용하여 주어진 상황에서 창의적인 다음 단계를 제안합니다. Planner는 탐색 트리의 확장 단계에서 작동하며, 가능한 행동들을 생성합니다. 예를 들어, 현재 냉장고에 우유와 계란이 있다면, "오믈렛 만들기" 또는 "프렌치 토스트 만들기"와 같은 행동을 제안할 수 있습니다.
2. **Simulator**: Planner가 제안한 행동의 결과를 예측하여, 탐색을 현실적인 결과로 기반을 둡니다. Simulator는 시뮬레이션을 통해 계획의 실행 가능성과 잠재적 결과를 평가합니다. 예를 들어, "오믈렛 만들기"를 선택했을 때 필요한 재료가 더 있는지, 조리 시간이 얼마나 걸리는지 등을 예측합니다.
3. **Critic**: 시뮬레이션 결과를 바탕으로 행동의 가치를 평가하고, 밀도 높은 보상 신호를 제공합니다. Critic은 MCTS 탐색에 필요한 보상 신호를 생성하여, 전략적으로 건전한 계획을 향해 탐색을 안내합니다. 예를 들어, "오믈렛 만들기"가 아침 식사로 적절한지, 영양 균형이 맞는지 등을 평가하여 보상 점수를 부여합니다.

### 핵심 수식

1. **보상 함수**:
   $$R(s, a) = r_{goal}(s, a) + \gamma * r_{safety}(s, a)$$
   - $R(s, a)$: 상태 $s$에서 행동 $a$를 취했을 때의 보상
   - $r_{goal}(s, a)$: 목표 달성 관련 보상 (예: 아침 식사 준비 완료)
   - $r_{safety}(s, a)$: 안전 관련 보상 (예: 칼 사용 시 안전 주의)
   - $\gamma$: 안전 관련 보상의 가중치

2. **보상 결합 수식**:
   $$R = (1 - \alpha) * R_{heuristic} + \alpha * R_{critic}$$
   - $R_{heuristic}$: 기본 유효성 휴리스틱에 따른 보상 (예: 문법적 오류 여부)
   - $R_{critic}$: Critic의 전략적 가치 평가에 따른 보상 (예: 영양 균형, 시간 효율성)
   - $\alpha$: 두 보상 간의 중요도를 조절하는 파라미터

3. **탐색 및 확장 수식**:
   - 탐색(Selection): Critic 에이전트가 제공하는 보상 신호를 기반으로, 가장 유망한 노드를 선택. UCB (Upper Confidence Bound)와 같은 MCTS의 표준 탐색 전략을 활용할 수 있습니다.
   - 확장(Expansion): Planner 에이전트가 선택된 노드에서 가능한 다음 행동들을 생성하여 새로운 노드를 추가.

SPIRAL의 구조적 탐색은 MCTS의 탐색(Selection), 확장(Expansion), 시뮬레이션(Simulation) 및 역전파(Backpropagation) 단계를 LLM 에이전트로 대체하여 수행됩니다. 이 과정은 LLM의 추론 능력을 활용하여 탐색의 깊이와 전략적 반영을 통해 더 나은 계획을 도출하도록 설계되었습니다.

### 코드 예제 (Python):

```python
# 간단한 Planner 예제 (가정)
def planner(state, available_actions):
  """
  주어진 상태와 가능한 행동 목록을 기반으로 다음 행동을 제안합니다.
  """
  # LLM을 사용하여 다음 행동을 선택하는 로직 (간략화)
  # 예: "현재 상태는 {state}이고, 가능한 행동은 {available_actions}입니다. 다음으로 할 행동은 무엇인가요?"
  # 실제로는 LLM API 호출이 필요합니다.
  next_action = available_actions[0] # 임시로 첫 번째 행동 선택
  return next_action

# 간단한 Simulator 예제 (가정)
def simulator(state, action):
  """
  주어진 상태와 행동을 기반으로 다음 상태를 예측합니다.
  """
  # LLM을 사용하여 다음 상태를 예측하는 로직 (간략화)
  # 예: "현재 상태는 {state}이고, {action}을 수행하면 어떤 결과가 발생할까요?"
  # 실제로는 LLM API 호출이 필요합니다.
  next_state = state + f" (After performing {action})" # 임시로 상태 업데이트
  return next_state

# 간단한 Critic 예제 (가정)
def critic(state, action, next_state):
  """
  주어진 상태, 행동, 다음 상태를 기반으로 보상을 계산합니다.
  """
  # LLM을 사용하여 보상을 계산하는 로직 (간략화)
  # 예: "현재 상태는 {state}이고, {action}을 수행하여 {next_state}가 되었습니다. 이 행동의 가치는 얼마인가요?"
  # 실제로는 LLM API 호출이 필요합니다.
  reward = 1.0 # 임시로 보상 1.0 부여
  return reward

# MCTS와 결합된 SPIRAL의 핵심 루프 (가정)
def spiral_mcts(initial_state, available_actions, num_iterations=100):
  """
  SPIRAL과 MCTS를 결합하여 최적의 계획을 탐색합니다.
  """
  # MCTS 트리 초기화
  tree = {initial_state: {}}

  for _ in range(num_iterations):
    # 1. Selection: Critic의 보상을 기반으로 노드 선택 (간략화)
    current_state = initial_state # 임시로 초기 상태 선택

    # 2. Expansion: Planner를 사용하여 새로운 행동 생성
    next_action = planner(current_state, available_actions)

    # 3. Simulation: Simulator를 사용하여 다음 상태 예측
    next_state = simulator(current_state, next_action)

    # 4. Critic: 보상 계산
    reward = critic(current_state, next_action, next_state)

    # 5. Backpropagation: 보상 업데이트 (간략화)
    tree[current_state][next_action] = reward

  # 최적의 행동 선택 (가장 높은 보상을 가진 행동)
  best_action = max(tree[initial_state], key=tree[initial_state].get)
  return best_action

# 사용 예시
initial_state = "냉장고에 우유와 계란이 있음"
available_actions = ["오믈렛 만들기", "프렌치 토스트 만들기"]
best_action = spiral_mcts(initial_state, available_actions)
print(f"최적의 행동: {best_action}")
```

**주의:** 위 코드 예제는 개념적인 설명을 위한 것이며, 실제 LLM API 호출 및 MCTS 구현을 포함하지 않습니다. 실제 구현에서는 OpenAI API 또는 Hugging Face Transformers 라이브러리를 사용하여 LLM을 통합해야 합니다.

## 실험 설정

SPIRAL의 성능을 평가하기 위해 DailyLifeAPIs와 HuggingFace 데이터셋을 사용하였습니다. DailyLifeAPIs는 일상 생활 시나리오를 시뮬레이션하는 데 사용되는 데이터셋이며, HuggingFace 데이터셋은 다양한 자연어 처리 작업을 위한 데이터셋을 제공합니다.

### 데이터셋

- **DailyLifeAPIs**: 일상 생활 시나리오를 시뮬레이션하는 1000개의 예제로 구성. API 호출을 통해 실제 환경과 유사한 상호작용을 제공합니다.
- **HuggingFace**: 자연어 처리 작업을 위한 500개의 예제로 구성. 텍스트 생성, 질의 응답 등 다양한 작업 포함.

### 평가 지표

- **정확도(Accuracy)**: 목표 달성률. 계획이 성공적으로 완료된 비율.
- **자원 효율성(Token Efficiency)**: 사용된 토큰 수 대비 성능. LLM API 호출 비용을 측정하는 데 중요합니다.

### 베이스라인

- **Chain-of-Thought**: 기존의 LLM 기반 계획 방법론. 단계별 추론을 통해 계획을 수립합니다.
- **기타 최신 에이전트**: ReAct, Reflexion 등 다양한 탐색 및 계획 알고리즘.

### 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|----|-------------------------------------------------|
| $\gamma$ | 0.5 | 안전 관련 보상의 가중치 |
| $\alpha$ | 0.3 | 휴리스틱 보상과 Critic 보상 간의 중요도 |
| 탐색 깊이 | 10 | MCTS 탐색 트리의 최대 깊이 |
| 반복 횟수 | 100 | MCTS 반복 횟수 |
| LLM 모델 | GPT-3.5 | 사용된 LLM 모델 (예: GPT-3.5, GPT-4) |
| 온도 (Temperature) | 0.7 | LLM의 생성 다양성을 조절하는 파라미터 |

## 실험 결과 분석

### 주요 결과

| 데이터셋 | SPIRAL 성능 | 베이스라인 성능 | 성능 향상률(%) |
|----------|------------|----------------|----------------|
| DailyLifeAPIs | 83.6% | 67.6% | 16% |
| HuggingFace | 96.6% | 80.6% | 16% |

SPIRAL은 DailyLifeAPIs에서 83.6%의 정확도를 달성하여, 다음으로 좋은 탐색 프레임워크보다 16% 이상 향상된 결과를 보였습니다. HuggingFace 데이터셋에서도 유사한 성능 향상을 나타냈습니다. 특히, API 호출이 필요한 복잡한 시나리오에서 강점을 보였습니다.

### Ablation Study

- **Critic의 영향**: Critic 에이전트를 제거한 경우, 성능이 10% 이상 하락하여 Critic의 전략적 평가가 SPIRAL의 성능 향상에 큰 영향을 미침을 확인할 수 있었습니다. 이는 Critic이 제공하는 밀도 높은 보상 신호가 효과적인 탐색에 필수적임을 시사합니다.
- **Planner의 역할**: Planner의 창의적인 행동 제안이 없을 경우, 탐색의 깊이가 얕아지고, 성능이 8% 감소하였습니다. 이는 Planner가 다양한 가능성을 탐색하는 데 중요한 역할을 한다는 것을 의미합니다.
- **Simulator의 기여**: Simulator의 현실적인 결과 예측이 없을 경우, 계획의 실행 가능성이 크게 떨어져 성능이 12% 감소하였습니다. 이는 Simulator가 현실적인 제약 조건을 고려하여 계획의 실현 가능성을 높이는 데 기여한다는 것을 보여줍니다.

## 비판적 평가

### 강점

1. **효율적 탐색**: SPIRAL은 LLM의 추론 능력과 MCTS의 탐색 능력을 결합하여, 복잡한 계획 문제를 효율적으로 해결합니다.
2. **자원 효율성**: 제한된 자원 환경에서도 높은 성능을 유지할 수 있습니다. 이는 LLM API 호출 비용을 줄이는 데 기여합니다.
3. **구조적 설계**: Planner, Simulator, Critic의 통합된 구조가 탐색 과정을 투명하고 해석 가능하게 만듭니다. 각 에이전트의 역할을 명확하게 분리하여 디버깅 및 개선이 용이합니다.

### 한계점 및 개선 방향

1. **실제 환경 적용의 어려움**: 시뮬레이터의 정확도를 높여 실제 환경과의 차이를 줄이는 것이 필요합니다. 예를 들어, 실제 API 호출 결과를 학습 데이터에 포함시켜 시뮬레이터의 예측 정확도를 높일 수 있습니다.
2. **복잡한 상황에서의 한계**: 복잡한 의존성을 가진 작업에서는 성능이 저하될 수 있습니다. 이는 Planner가 모든 가능한 행동을 고려하지 못하거나, Simulator가 복잡한 상호작용을 정확하게 예측하지 못하기 때문일 수 있습니다.
3. **재현성 평가**: 코드와 데이터셋이 공개되어 있지만, 특정 하드웨어나 소프트웨어 환경에 의존할 수 있습니다. Docker 컨테이너를 제공하여 재현성을 높일 수 있습니다.

## 향후 연구 방향

1. **실제 도구 실행과의 비교**: 시뮬레이터의 정확도를 높여 실제 도구 실행과 비교하여 성능을 검증합니다. 예를 들어, 실제 로봇 제어 환경에서 SPIRAL의 성능을 평가할 수 있습니다.
2. **자기 개선 및 평생 학습**: SPIRAL이 스스로 학습하고 개선할 수 있도록 하는 연구를 진행합니다. 강화 학습을 통해 Critic의 보상 함수를 최적화하거나, Planner의 행동 제안 능력을 향상시킬 수 있습니다.
3. **더 복잡한 환경으로의 확장**: SPIRAL을 로봇 제어, 자율 주행, 의료 진단 등 다양한 분야에 적용합니다.

## 실무 적용 가이드

- **구현 시 고려사항**: SPIRAL의 각 에이전트가 상호작용하는 방식을 이해하고, 각 단계의 성능을 최적화해야 합니다. 특히, LLM API 호출 비용을 최소화하는 것이 중요합니다.
- **팁**: Planner의 창의적인 행동 제안을 극대화하기 위해 다양한 시나리오를 테스트하고, Critic의 평가 기준을 명확히 설정합니다. 프롬프트 엔지니어링을 통해 LLM의 성능을 극대화할 수 있습니다. 또한, MCTS의 탐색 전략 (예: UCB, PUCT)을 적절하게 조정하여 성능을 향상시킬 수 있습니다.

## 결론

SPIRAL은 LLM의 추론 능력을 강화하고, MCTS와 결합하여 복잡한 계획 문제를 효율적으로 해결하는 새로운 방법론을 제안합니다. 이 연구는 LLM 기반의 자율 계획 능력을 향상시키고, 복잡한 작업 자동화를 위한 새로운 가능성을 제시합니다. SPIRAL은 제한된 자원 환경에서도 효과적으로 작동할 수 있으며, 다양한 분야에 적용할 수 있는 잠재력을 가지고 있습니다.

## 참고 자료

- 논문 링크: [arXiv:2512.23167](https://arxiv.org/abs/2512.23167)
- 코드 저장소: [GitHub Repository](https://github.com/spiral-ai/spiral)
- 관련 자료: [HuggingFace Dataset](https://huggingface.co/datasets)
- OpenAI API: [OpenAI API Documentation](https://platform.openai.com/docs/api-reference) (LLM API 사용 시)