---
title: "[논문 리뷰] Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty"
date: "2026-01-03"
excerpt: "The honesty of Large Language Models (LLMs) is increasingly important for safe deployment in high-stakes domains. However, this crucial trait is severely undermined by supervised fine-tuning (SFT), a ..."
category: "Paper Review"
tags: ["Paper Review","cs.CL","cs.CL"]
thumbnail: "/assets/images/blog/20260103-paper-2511-12991-fine-tuned-llms-know-they-don-.jpg"
---

# [논문 리뷰] Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty

## 1. TL;DR

본 논문은 파인튜닝된 대규모 언어 모델(LLM)의 부정직함이 모델의 지식 경계에 대한 인식이 손상된 것이 아니라, 그 인식을 표현하는 능력이 억압된 결과라는 것을 밝혀냈습니다. 기존의 부정직함 복구 방법들은 데이터 집약적인 전역 파라미터 조정을 통해 모델의 지식 경계를 근본적으로 손상되었다고 가정했지만, 본 연구에서는 파인튜닝된 LLM이 여전히 자신의 무지를 인지하고 있음을 관찰했습니다. 이러한 관찰을 바탕으로, 본 논문에서는 **Honesty-Critical Neurons Restoration (HCNR)**이라는 새로운 접근법을 제안합니다. HCNR은 정직함 표현을 담당하는 핵심 뉴런을 식별하여 사전 학습된 상태로 복원하고, 헤시안 가이드 보상을 통해 이를 작업 지향적인 뉴런과 조화시킵니다. 다양한 질문응답(QA) 작업과 LLM 모델군에 대한 실험 결과, HCNR은 기존 방법 대비 10배 적은 데이터와 2.23배 빠른 속도로 훼손된 정직함의 33.25%를 효과적으로 복구하여 신뢰할 수 있는 LLM 배포를 위한 실용적인 솔루션을 제공합니다.

## 2. 연구 배경 및 동기

대규모 언어 모델(LLM)이 의료, 법률, 교육 등 고위험(high-stakes) 영역에 점점 더 많이 통합되면서, LLM의 신뢰성은 단순한 기능이 아닌 필수적인 요소가 되었습니다. LLM의 신뢰성을 구성하는 핵심 요소 중 하나는 **정직함(honesty)**이며, 이는 모델이 자신의 지식 경계를 인식하고, 아는 것과 모르는 것을 구별할 수 있는 능력(self-knowledge)과, 이러한 인식을 바탕으로 솔직하게 자기 표현을 할 수 있는 능력(faithful self-expression)으로 구성됩니다. LLM이 자신 있게 허위 사실을 만들어내거나 잘못된 치료법을 추천하는 경우, 사용자 신뢰를 훼손하고 심각한 해를 끼칠 수 있기 때문에 정직함은 매우 중요합니다.

일반적으로 LLM의 정직함은 인간 피드백 기반 강화 학습(RLHF)과 같은 정렬(alignment) 단계를 통해 주입됩니다. RLHF는 모델이 부적절한 질문이나 지식 범위를 벗어나는 질문에 대해 거부하도록 훈련시킵니다. 그러나 이러한 방식으로 획득한 정직함은 고정불변한 것이 아닙니다. 최근 연구에 따르면 지도 학습 기반 파인튜닝(SFT)은 법률 QA, 의료 진단, 교육 콘텐츠 생성 등 다양한 분야에서 LLM의 정직함을 크게 손상시킬 수 있습니다.

SFT 후 LLM의 정직함을 복구하기 위해 기존 방법들은 광범위한 데이터셋을 사용하여 전역 파라미터를 크게 변경합니다. 이러한 접근 방식은 모델의 지식 경계가 심각하게 손상되었고, 자기 인식 능력을 상실했다는 가정에 기반합니다. 예를 들어, RAIT(Refusal-Aware Instruction Tuning)는 모델이 "모르겠습니다"라고 응답하도록 명시적으로 가르치는 방식으로, IDK(I Don't Know) 데이터셋을 사용하여 SFT를 수행합니다. 다른 방법으로는 Rehearsal, DPO(Direct Preference Optimization), ORPO(Odds Ratio Preference Optimization) 등이 있습니다.

하지만 본 논문에서는 파인튜닝된 LLM의 부정직함의 근본 원인이 자기 인식 능력의 손실이 아닌, 자기 표현 능력의 실패라는 점을 지적합니다. 즉, 모델은 여전히 자신의 무지를 인지하고 있지만, 이를 솔직하게 표현하지 못하는 것입니다. 이러한 관찰에 따라 본 논문은 전역적인 개입이 불필요할 수 있다고 주장하며, **Honesty-Critical Neurons Restoration (HCNR)**이라는 목표 지향적이고 파라미터 효율적인 솔루션을 제안합니다.

본 연구가 해결하는 gap은 다음과 같습니다.

*   기존 방법들은 LLM의 부정직함의 원인을 자기 인식 능력의 손실로 잘못 진단하고, 데이터 집약적인 전역 파라미터 조정에 의존합니다.
*   기존 방법들은 파인튜닝된 LLM이 여전히 자신의 무지를 인지하고 있다는 사실을 간과합니다.
*   기존 방법들은 종종 다운스트림 작업 성능 저하를 초래합니다.

이러한 gap을 해결하기 위해 본 연구는 다음과 같은 연구 질문을 제시합니다.

*   파인튜닝된 LLM의 부정직함의 근본 원인은 무엇인가?
*   자기 표현 능력을 선택적으로 복원하여 LLM의 정직함을 효율적으로 복구할 수 있는가?
*   제안하는 방법이 다운스트림 작업 성능에 미치는 영향은 무엇인가?

## 3. 관련 연구

본 논문과 관련된 선행 연구들을 분석하고, 본 논문과의 차별점을 표로 정리하면 다음과 같습니다.

| 연구                                  | 주요 내용