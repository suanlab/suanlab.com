---
title: "[논문 리뷰] Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration"
date: "2026-01-03"
excerpt: "Hyperparameter tuning can dramatically impact training stability and final performance of large-scale models. Recent works on neural network parameterisations, such as $μ$P, have enabled transfer of o..."
category: "Paper Review"
tags: ["Paper Review","cs.LG","cs.AI","stat.ML"]
thumbnail: "/assets/images/blog/20260103-paper-2512-22382-completed-hyperparameter-trans.jpg"
---

# [논문 리뷰] Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration

## TL;DR

본 논문은 대규모 모델 학습 시 하이퍼파라미터 튜닝의 중요성을 강조하며, 모델 크기, 너비, 깊이, 배치 크기, 학습 기간 등 다양한 스케일링 축에 걸쳐 하이퍼파라미터를 효율적으로 전이할 수 있는 새로운 파라미터화 방법인 Complete$^{(d)}$ Parameterisation (Complete$^{(d)}$P)를 제안한다. Complete$^{(d)}$P는 기존 방법론의 한계를 극복하고, 모듈별 하이퍼파라미터 최적화 및 전이를 가능하게 하여 전체 모델의 성능을 극대화한다. 실험 결과, Complete$^{(d)}$P를 통해 최적화된 모듈별 하이퍼파라미터는 대규모 모델 학습 속도를 크게 향상시켰으며, 이는 실제 대규모 모델 학습에 효과적으로 적용될 수 있음을 시사한다. 본 연구는 대규모 모델의 하이퍼파라미터 최적화 문제를 해결하기 위한 새로운 접근법을 제시하고, 학습 효율성을 크게 향상시킴으로써 심층 학습 모델의 효율적인 개발과 배포에 기여할 것으로 기대된다. 특히, 모듈별 하이퍼파라미터 전이를 통해 학습 속도를 최대 27%까지 향상시키는 결과를 보여주며, 이는 대규모 언어 모델(LLM) 학습에 큰 영향을 미칠 수 있다.

## 연구 배경 및 동기

최근 딥러닝 모델의 규모가 기하급수적으로 증가하면서 하이퍼파라미터 튜닝의 중요성이 더욱 부각되고 있다. 대규모 모델은 수백만 개에서 수십억 개의 파라미터를 가지며, 이러한 모델의 학습은 막대한 계산 자원과 시간을 요구한다. 따라서, 하이퍼파라미터 설정이 조금만 잘못되어도 학습이 불안정해지거나, 과적합이 발생하거나, 최종 성능이 저하되는 등 심각한 문제가 발생할 수 있다. 특히, 대규모 언어 모델(LLM)과 같은 복잡한 모델의 경우, 하이퍼파라미터 공간이 매우 넓고 복잡하여 최적의 설정을 찾기가 매우 어렵다.

기존의 하이퍼파라미터 튜닝 방법은 대부분 경험적인 시행착오(trial and error)에 의존하거나, Grid Search, Random Search와 같은 단순 탐색 방법을 사용한다. 하지만 이러한 방법들은 계산 비용이 매우 높고, 대규모 모델에는 적용하기 어렵다는 한계가 있다. Bayesian Optimization이나 Reinforcement Learning과 같은 보다 발전된 방법들도 있지만, 여전히 하이퍼파라미터 공간의 복잡성으로 인해 최적의 설정을 찾는데 어려움을 겪고 있다.

최근에는 Neural Architecture Search (NAS) 연구가 활발하게 진행되고 있지만, NAS는 주로 모델 구조 자체를 탐색하는 데 초점을 맞추고 있으며, 하이퍼파라미터 튜닝 문제는 상대적으로 덜 다루고 있다. 또한, NAS는 계산 비용이 매우 높아 대규모 모델에는 적용하기 어렵다는 단점이 있다.

이러한 배경 하에, 본 연구는 대규모 모델의 하이퍼파라미터 튜닝 문제를 해결하기 위해 하이퍼파라미터 전이(hyperparameter transfer)라는 새로운 접근법을 제시한다. 하이퍼파라미터 전이는 작은 모델에서 최적화된 하이퍼파라미터 설정을 대규모 모델로 전이하여 하이퍼파라미터 탐색 비용을 줄이는 방법이다. 특히, 본 연구는 모델의 너비, 깊이, 배치 크기, 학습 기간 등 다양한 스케일링 축에 걸쳐 하이퍼파라미터를 전이할 수 있는 새로운 파라미터화 방법인 Complete$^{(d)}$P를 제안한다. 또한, 모듈별 하이퍼파라미터 최적화 및 전이를 통해 전체 모델의 성능을 극대화하는 방법을 제시한다.

본 연구는 다음과 같은 연구 질문에 답하고자 한다.

1.  Complete$^{(d)}$P는 다양한 스케일링 축에 걸쳐 하이퍼파라미터를 효과적으로 전이할 수 있는가?
2.  모듈별 하이퍼파라미터 최적화 및 전이는 대규모 모델의 학습 속도와 성능을 향상시킬 수 있는가?
3.  Complete$^{(d)}$P는 기존의 하이퍼파라미터 튜닝 방법보다 효율적인가?

## 관련 연구

본 연구는 하이퍼파라미터 최적화 및 전이에 대한 기존 연구를 기반으로 한다. 다음은 본 연구와 관련된 주요 선행 연구들이다.

1.  **μP (Micro-Parameterization)**: μP는 모델의 크기가 변하더라도 학습률과 같은 하이퍼파라미터를 유지할 수 있도록 하는 파라미터화 방법이다. μP는 모델의 너비를 늘릴 때 학습률을 적절히 조정하여 학습 안정성을 유지하고 성능을 향상시키는 데 기여한다. (Yang et al., 2021)

2.  **CompleteP**: CompleteP는 μP를 확장하여 모델의 깊이를 늘릴 때도 하이퍼파라미터를 전이할 수 있도록 하는 파라미터화 방법이다. CompleteP는 모델의 너비와 깊이를 동시에 스케일링할 때 하이퍼파라미터를 적절히 조정하여 학습 효율성을 높이는 데 기여한다. (De et al., 2023)

3.  **Transfer Learning for Hyperparameter Optimization**: 이 연구는 작은 데이터셋에서 최적화된 하이퍼파라미터를 큰 데이터셋으로 전이하는 방법을 제안한다. Transfer Learning for Hyperparameter Optimization은 데이터셋 크기가 변하더라도 하이퍼파라미터 탐색 비용을 줄이고 성능을 향상시키는 데 기여한다. (Yogatama et al., 2016)

4.  **Meta-Learning for Hyperparameter Optimization**: 이 연구는 다양한 작업(task)에서 학습된 하이퍼파라미터를 새로운 작업에 적용하는 방법을 제안한다. Meta-Learning for Hyperparameter Optimization은 새로운 작업에 대한 하이퍼파라미터 탐색 비용을 줄이고 초기 성능을 향상시키는 데 기여한다. (Finn et al., 2017)

5.  **Auto-sklearn**: Auto-sklearn은 Bayesian Optimization과 Meta-Learning을 결합하여 하이퍼파라미터 최적화를 자동화하는 프레임워크이다. Auto-sklearn은 다양한 머신러닝 모델에 대한 하이퍼파라미터 탐색을 자동화하고 최적의 모델을 선택하는 데 기여한다. (Feurer et al., 2015)

| 연구                                      | 주요 특징                                                                                                                               | 본 논문과의 차별점