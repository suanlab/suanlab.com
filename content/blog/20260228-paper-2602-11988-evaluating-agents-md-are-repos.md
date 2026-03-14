---
title: "[논문 리뷰] Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?"
date: "2026-02-28"
excerpt: "A widespread practice in software development is to tailor coding agents to repositories using context files, such as AGENTS.md, by either manually or automatically generating them. Although this prac..."
category: "Paper Review"
tags: ["Paper Review","cs.SE","cs.AI","cs.SE"]
thumbnail: "/assets/images/blog/20260228-paper-2602-11988-evaluating-agents-md-are-repos.jpg"
---

# [논문 리뷰] Evaluating AGENTS.md: 저장소 수준 컨텍스트 파일은 코딩 에이전트에 도움이 되는가?

## TL;DR

이 논문은 코딩 에이전트(coding agent)가 소프트웨어 개발 작업을 수행할 때, 저장소(repository) 수준의 컨텍스트 파일(예: `AGENTS.md`)이 실제로 작업 성공률에 도움이 되는지를 평가합니다. 연구 결과, 컨텍스트 파일은 에이전트의 행동을 변화시키지만 **최종 작업 성공률에는 거의 영향을 미치지 않는 것**으로 나타났습니다. 특히, LLM이 생성한 장황한 파일은 비용만 증가시키고 오히려 성능을 저하시켰습니다. 결론적으로, 컨텍스트 파일은 저장소별 특수 도구 사용법 등 **필수적인 최소 요구사항만 간결하게 기술**하는 것이 가장 바람직합니다.

## 연구 배경 및 동기

최근 소프트웨어 개발에서 코딩 에이전트의 활용이 폭발적으로 증가하고 있습니다. 코딩 에이전트란 대규모 언어 모델(LLM)을 기반으로, 사람의 개입 없이 자율적으로 버그 수정, 기능 추가, 코드 리팩토링 등의 개발 작업을 수행하는 AI 프로그램을 의미합니다. 이러한 에이전트가 저장소의 구조와 맥락을 더 잘 이해하고 효율적으로 작업하도록 돕기 위해, `AGENTS.md`와 같은 컨텍스트 파일이 제안되었습니다. 그러나 이 파일이 정말로 에이전트의 성능 향상에 기여하는지에 대한 체계적인 연구는 부족했습니다.

본 연구는 이러한 간극을 메우고자 하며, 컨텍스트 파일의 유용성을 다음 핵심 질문을 통해 엄밀하게 평가합니다: **컨텍스트 파일이 코딩 에이전트의 작업 성공률에 실질적인 영향을 미치는가?**

## 관련 연구

선행 연구에서는 코딩 에이전트의 성능 향상을 위해 다양한 접근법이 시도되었습니다. LLM을 활용한 코드 자동 생성 및 수정, 자연어 처리 기반의 버그 탐지 등이 대표적입니다. 그러나 대부분의 연구는 개별 코드 스니펫이나 파일 단위의 작업에 초점을 맞추고 있으며, 저장소 전체의 맥락을 고려한 연구는 드물었습니다. 본 논문은 저장소 수준의 컨텍스트 파일이 에이전트의 작업 수행에 어떤 영향을 미치는지 직접적으로 평가했다는 점에서 기존 연구와 차별화됩니다.

| 연구 구분 | 주요 접근법 | 본 논문과의 차별점 |
| :--- | :--- | :--- |
| 기존 연구 A | LLM 기반 코드 생성 | 개별 코드 스니펫에 집중, 저장소 맥락 미고려 |
| 기존 연구 B | 자연어 처리 기반 버그 탐지 | 저장소의 개발 환경이나 규칙 등 고차원적 맥락 미고려 |
| 기존 연구 C | 코드 수정 및 테스트 자동화 | 맥락 정보 파일의 효과를 직접적으로 평가하지 않음 |
| **본 논문** | **컨텍스트 파일 효과 평가** | **저장소 수준의 맥락 정보가 에이전트 성능에 미치는 영향 정량적 분석** |

## 핵심 기여

1.  **컨텍스트 파일의 효과 정량적 평가**: 저장소 수준의 컨텍스트 파일이 코딩 에이전트의 성공률, 비용, 행동 패턴에 미치는 영향을 체계적으로 분석했습니다.
2.  **AGENTBENCH 벤치마크 제안**: 실제 GitHub 저장소의 이슈와 개발자가 직접 작성한 컨텍스트 파일을 기반으로 한 새로운 벤치마크 `AGENTBENCH`를 구축했습니다.
3.  **LLM 생성 파일의 한계 규명**: LLM이 자동 생성한 컨텍스트 파일은 유용하기보다 오히려 성능을 저하시키고 비용을 증가시킨다는 것을 실험적으로 입증했습니다.
4.  **행동 변화 분석**: 컨텍스트 파일이 에이전트의 행동(예: 파일 탐색, 테스트 실행)을 유의미하게 변화시키지만, 이것이 반드시 최종적인 작업 성공으로 이어지지는 않는다는 점을 밝혔습니다.

## 제안 방법론

본 논문은 컨텍스트 파일의 효과를 체계적으로 평가하기 위한 프레임워크를 제안합니다. 핵심 아이디어는 `AGENTBENCH` 벤치마크를 활용하여, 컨텍스트 파일의 유무와 종류에 따른 에이전트의 성능 변화를 실험적으로 검증하는 것입니다.

### 1. AGENTBENCH 벤치마크 구축

-   **데이터 수집**: 개발자가 직접 작성한 컨텍스트 파일(`AGENTS.md`, `CONTRIBUTING.md` 등)이 포함된 실제 Python GitHub 저장소를 수집합니다.
-   **과업 추출**: 실제 이슈(Issue)와 이를 해결한 커밋(Commit) 및 Pull Request(PR)를 연결하여 현실적인 개발 과업을 추출합니다.
-   **과업 표준화**: LLM 에이전트를 활용해 불명확한 과업 설명을 표준화하고, 테스트가 부족한 경우 과업의 성공 여부를 판별할 단위 테스트를 자동으로 생성하여 벤치마크의 신뢰도를 높입니다.

### 2. 비교 평가 파이프라인

동일한 코딩 과업에 대해 아래 세 가지 시나리오를 구성하여 에이전트의 성능을 비교 분석합니다.

-   **None**: 컨텍스트 파일을 제공하지 않는 기본(baseline) 환경.
-   **LLM**: LLM(GPT-4)을 이용해 저장소 구조를 분석하고 컨텍스트 파일을 자동으로 생성하여 제공하는 환경.
-   **Human**: 개발자가 직접 작성한 원본 컨텍스트 파일을 제공하는 환경 (`AGENTBENCH` 데이터셋에만 해당).

각 시나리오에서 에이전트의 과업 성공률, 추론 비용, 작업 단계 수, 행동 패턴(trace) 등을 수집하고 비교 분석합니다.

### 핵심 평가지표

1.  **성공률 (Success Rate)**: 에이전트가 제출한 코드 수정안이 주어진 모든 테스트 케이스를 통과하는 비율.
    $$
    SuccessRate = \frac{\text{Number of Successful Tasks}}{\text{Total Number of Tasks}}
    $$

2.  **추론 비용 (Inference Cost)**: 에이전트가 작업을 수행하는 데 사용된 LLM의 입출력 토큰 수에 기반한 비용(USD).
    $$
    InferenceCost = \text{Cost per Token} \times \text{Number of Tokens}
    $$

3.  **작업 단계 수 (Number of Steps)**: 에이전트가 과업을 완료하기 위해 수행한 명령어(파일 읽기, 코드 수정, 테스트 실행 등)의 총 개수.

## 실험 설정

실험은 외부 정답 유출을 막기 위해 인터넷 접근은 허용하되 git commit 기록은 제거된 격리된 Docker 컨테이너 환경에서 수행되었습니다.

-   **에이전트**: OpenAI의 GPT-4o, Anthropic의 Claude 3 Opus 등 최신 고성능 LLM을 기반으로 한 자율 에이전트 사용.
-   **데이터셋**: 기존의 `SWE-BENCH LITE`와 본 논문에서 제안한 `AGENTBENCH`.
-   **평가 시나리오**: 컨텍스트 파일이 없는 경우(None), LLM 생성 파일(LLM), 개발자 작성 파일(Human)의 세 가지 조건.
-   **평가 지표**: 과업 성공률, 추론 비용(USD), 작업 단계 수.

| 항목 | 설정값 |
| :--- | :--- |
| **LLM 모델** | GPT-4o, Claude 3 Opus |
| **데이터셋** | SWE-BENCH LITE, AGENTBENCH |
| **평가 지표** | 성공률, 작업 단계 수, 추론 비용 |

## 실험 결과 분석

실험 결과는 컨텍스트 파일의 유용성에 대한 일반적인 통념에 도전합니다.

-   **LLM 생성 파일은 해롭다**: LLM이 생성한 컨텍스트 파일은 대부분의 경우 **과업 성공률을 소폭 감소**시켰으며, **추론 비용은 평균 20% 이상 증가**시켰습니다. 이는 LLM이 생성한 장황하고 일반적인 정보가 에이전트를 핵심 과업에서 벗어나게 만드는 '주의 분산' 효과를 유발하기 때문으로 분석됩니다.
-   **사람이 작성한 파일도 효과는 미미**: 개발자가 직접 작성한 파일은 성공률을 평균 2~4%p 정도 소폭 향상시켰지만, 이 역시 작업 단계 수와 비용을 증가시키는 경향을 보였습니다.
-   **행동은 바뀌지만 결과는 그대로**: 컨텍스트 파일이 제공되면 에이전트는 파일의 지시를 따르려는 경향을 보였습니다. 예를 들어, "테스트를 철저히 하라"는 내용이 있으면 불필요하게 많은 테스트를 실행하거나, "전체 구조를 파악하라"는 내용에 따라 더 넓은 범위의 파일을 탐색하는 등 **과도한 탐색(over-exploration)**을 수행했습니다. 이러한 행동 변화가 최종 성공률 향상으로 직결되지는 않았습니다.

| 조건 | 성공률(%) (상대적 변화) | 추론 비용(USD) (상대적 변화) | 작업 단계 수 (상대적 변화) |
| :--- | :--- | :--- | :--- |
| **None (기준)** | 60.0% | \$1.00 | 100 |
| **LLM** | 58.2% (-1.8%p) | \$1.22 (+22%) | 125 (+25%) |
| **Human** | 62.5% (+2.5%p) | \$1.15 (+15%) | 118 (+18%) |
*(위 수치는 논문의 경향성을 보여주기 위한 예시입니다.)*

## 비판적 평가

-   **강점**: 본 연구는 업계의 막연한 기대를 실험적으로 검증하여 컨텍스트 파일의 실제 유용성에 대한 중요한 실증적 증거를 제시했습니다. 또한, `AGENTBENCH`라는 현실적인 벤치마크를 제안하여 후속 연구의 기틀을 마련했습니다.
-   **한계점**: 연구가 주로 Python 저장소에 한정되어 있어 다른 프로그래밍 언어나 복잡한 빌드 시스템을 가진 프로젝트에서도 동일한 결과가 나올지는 추가 검증이 필요합니다. 또한, 컨텍스트 파일의 '품질'을 정량화하는 지표가 없어, 잘 작성된 파일과 그렇지 않은 파일의 효과를 더 세밀하게 분석하지 못한 점은 아쉬움으로 남습니다.
-   **재현성**: 제안된 벤치마크와 실험 설정이 명확히 기술되어 있어 연구의 재현 가능성은 높다고 판단됩니다.

## 향후 연구 방향

향후 연구에서는 다양한 프로그래밍 언어와 프로젝트 유형을 포괄하는 데이터셋을 구축하여 일반화 가능성을 높일 필요가 있습니다. 또한, 에이전트가 작업에 필요한 최소한의 정보를 동적으로 선택하고 활용하도록 하는 '적응형 컨텍스트(adaptive context)' 방법론을 개발하여, 불필요한 정보로 인한 비효율을 줄이는 연구가 유망할 것입니다.

## 실무 적용 가이드

실무에서 코딩 에이전트를 위해 컨텍스트 파일을 작성할 때는 다음 원칙을 따르는 것이 좋습니다.

-   **최소주의 원칙**: 모든 것을 설명하려 하지 말고, 저장소별 특수 도구 사용법이나 반드시 따라야 하는 설정 방법 등 **에이전트가 스스로 파악하기 어려운 필수 정보**만 간결하게 기술하세요.
-   **LLM 자동 생성 주의**: LLM에게 "이 저장소에 대한 `AGENTS.md`를 만들어줘"라고 요청하는 것은 비용 낭비일 가능성이 높습니다. 생성된 내용을 반드시 검토하고 핵심 정보만 남겨야 합니다.
-   **중복 정보 제거**: `README.md`나 다른 문서에 이미 있는 내용은 반복하지 마세요. 에이전트의 작업 비용만 증가시킵니다.

### 좋은 예시 vs. 나쁜 예시

#### 👎 나쁜 예시 (LLM이 생성한 장황한 파일)

```markdown
# AGENTS.md for Project-X

## Welcome to Project-X!
This document provides a guide for AI agents working on this repository.

## Overview
Project-X is a powerful data processing library written in Python. Our goal is to create efficient and readable code. Please adhere to best practices.

## Getting Started
1. Clone the repository.
2. Set up a virtual environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Run tests to ensure everything is working.

## How to Contribute
- Please write clean, well-documented code.
- Add unit tests for any new features.
- Make sure all tests pass before submitting a change.
... (이하 일반적인 내용 반복)
```

#### 👍 좋은 예시 (핵심 정보만 담은 간결한 파일)

```markdown
# AGENTS.md for Project-X

## Setup
- Use Poetry for dependency management: `poetry install`
- The custom data processing engine must be initialized before running tests. Use the helper script: `bash scripts/init_engine.sh`

## Testing
- All tests are located in the `tests/` directory.
- Run all tests with: `poetry run pytest`
- For integration tests involving the database, use the `--run-db-tests` flag.
```

## 결론

본 논문은 저장소 수준의 컨텍스트 파일이 코딩 에이전트의 성능에 미치는 영향을 심층적으로 분석하여, **컨텍스트 파일이 만능 해결책이 아님**을 명확히 보여주었습니다. 특히 LLM이 자동으로 생성한 파일은 비용만 증가시키고 성공률을 저해할 수 있으므로 사용을 지양해야 합니다. 사람이 직접 작성하더라도, 저장소의 특수한 설정이나 도구 사용법 등 **필수적인 최소 요구사항만 간결하게 기술**하는 것이 가장 효과적이라는 실용적인 결론을 제시합니다.

## 참고 자료

-   [논문 원문 링크 (가상)](https://arxiv.org/abs/2602.11988)
-   [코드 저장소 (가상)](https://github.com/some-research-group/agent-benchmark)
-   관련 자료: SWE-bench, AGENTBENCH Dataset