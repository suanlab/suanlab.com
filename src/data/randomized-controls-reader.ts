import type { LearningGuide, LearningUnit } from './interactive-book';

export const randomizedControlsId = 'ablations-and-randomized-negative-controls' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const randomizedControlsUnit: LearningUnit = {
  id: randomizedControlsId,
  title: t('제거 실험과 무작위 음성 대조', 'Ablations and Randomized Negative Controls'),
  subtitle: t('가중치를 고정한 특성 교란과 재학습한 대조의 결과를 비교합니다.', 'Compare perturbations of a frozen predictor with refitted controls.'),
  prerequisites: t('독립 변수의 분산과 선형 예측을 복습하세요.', 'Review variances of independent variables and linear prediction.'),
  prerequisiteIds: ['baselines-and-matched-compute-budgets'],
  eyebrow: 'ABLATIONS AND CONTROLS',
  formula: 'Y = 0.7X + cZ + 0.1N',
  accent: 'block-order',
  questions: [
    {
      id: 'controls-frozen-refitted',
      question: t('특성을 무작위화하는 대조에서 반드시 밝혀야 하는 것은?', 'What must be specified for a feature-randomization control?'),
      options: [t('가중치를 고정했는지 재학습했는지', 'Whether weights were frozen or refitted'), t('그림 색상만', 'Only the plot color'), t('최종 지표의 반올림 자릿수만', 'Only the rounded metric precision')],
      answer: 0,
      explanation: t('고정 대조와 재학습 대조는 서로 다른 질문을 검사합니다.', 'Frozen and refitted controls test different questions.'),
    },
    {
      id: 'controls-preserve-marginal',
      question: t('독립 Z′로 바꾸면 어떤 성질이 유지될 수 있나요?', 'What can remain unchanged when Z is replaced by independent Z′?'),
      options: [t('특성의 주변 분포', 'The feature’s marginal distribution'), t('목표와의 상관', 'Its target correlation'), t('모든 표본의 동일한 예측', 'Identical predictions for every sample')],
      answer: 0,
      explanation: t('같은 주변 분포를 사용하면서 목표와의 관계를 제거할 수 있습니다.', 'A randomization can preserve the marginal distribution while removing its target relationship.'),
    },
  ],
};

export const randomizedControlsGuide: LearningGuide = {
  intuition: t('관계를 끊는 교란은 모형이 어떤 정보에 의존하는지 확인하게 해 줍니다.', 'Disrupting a relationship probes which information a model relies on.'),
  steps: [{
    title: t('같은 특성 분포, 다른 관계', 'The same marginal with a different relationship'),
    explanation: t('독립 특성으로 바꾼 고정 모형에는 틀린 특성 신호도 더해집니다.', 'A frozen model adds an irrelevant feature signal after independent replacement.'),
    formula: 'MSE_frozen = 2c² + 0.01',
    diagram: t('독립 교란에서 제거·고정·재학습의 오차를 비교합니다.', 'Compare removal, frozen randomization, and refitting under independent perturbation.'),
  }],
  experiments: [
    {
      title: t('독립 무작위화', 'Independent randomization'),
      predict: t('c=0.3에서 고정 모형과 재학습 모형의 오차는 같을까요?', 'At c=0.3, do frozen and refitted randomization give the same error?'),
      action: t('상관을 0으로 두고 두 오차를 비교하세요.', 'Set correlation to zero and compare the errors.'),
      explain: t('고정은 0.19, 재학습은 0.10입니다.', 'The frozen error is 0.19 and the refitted error is 0.10.'),
      preset: 0,
    },
    {
      title: t('정보가 남는 교란', 'A perturbation retaining information'),
      predict: t('대체 특성과 원 특성의 상관이 커지면 재학습 오차는 어떻게 될까요?', 'How does retained feature correlation affect refitted error?'),
      action: t('상관을 0에서 1로 올리세요.', 'Raise correlation from zero to one.'),
      explain: t('상관이 1이면 원 특성 정보를 복원해 오차가 0.01입니다.', 'At correlation one the feature information is recovered and error is 0.01.'),
      preset: 1,
    },
  ],
};
