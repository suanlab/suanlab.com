import type { LearningGuide, LearningUnit } from './interactive-book';

export const evidenceRolesId = 'theorems-evidence-and-working-hypotheses' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const evidenceRolesUnit: LearningUnit = {
  id: evidenceRolesId,
  title: t('정리와 경험적 근거의 역할', 'Theorems, Evidence and Working Hypotheses'),
  subtitle: t('같은 관측값을 둘러싼 실측 요약과 가정에 따른 보장을 구분합니다.', 'Distinguish an empirical summary from an assumption-dependent guarantee.'),
  prerequisites: t('유계 손실, 확률 상계와 합집합 상계를 복습하세요.', 'Review bounded loss, probability bounds, and the union bound.'),
  prerequisiteIds: ['dimensions-limits-and-convergence-checks'],
  eyebrow: 'ROLES OF EVIDENCE',
  formula: 'ε = √[ln(2M/δ)/(2m)]',
  accent: 'block-order',
  questions: [
    {
      id: 'evidence-model-selection',
      question: t('평가와 독립적으로 정한 후보 수 M이 커지면 동시 반지름은?', 'What happens to the simultaneous radius as the fixed candidate count M grows?'),
      options: [t('커집니다.', 'It increases.'), t('작아집니다.', 'It decreases.'), t('항상 0입니다.', 'It is always zero.')],
      answer: 0,
      explanation: t('여러 후보에 대한 동시 주장은 더 큰 범위를 보장합니다.', 'A simultaneous claim must cover more candidate models.'),
    },
    {
      id: 'evidence-proof-assumptions',
      question: t('이 Hoeffding 보장을 적용하기 전에 무엇을 확인해야 하나요?', 'What should be checked before applying this Hoeffding guarantee?'),
      options: [t('독립 표본, 유계 손실, 평가와 독립적인 후보 구성', 'Independent observations, bounded loss, and candidates fixed independently of evaluation'), t('평균 손실이 작다는 사실만', 'Only that the mean loss is small'), t('시각화가 매끄러운지만', 'Only whether the plot is smooth')],
      answer: 0,
      explanation: t('주어진 부등식은 명시한 가정에 의존합니다.', 'The displayed inequality depends on its stated assumptions.'),
    },
  ],
};

export const evidenceRolesGuide: LearningGuide = {
  intuition: t('측정값은 그대로여도 보장하려는 대상과 가정을 바꾸면 결론의 범위가 달라집니다.', 'The same measurement can support different guarantees when the scope and assumptions change.'),
  steps: [{
    title: t('관측과 보장의 차이', 'Measurement and guarantee'),
    explanation: t('관측 평균 주변에 가정 아래의 확률 상계에서 얻은 반지름을 둡니다.', 'Place a radius around the observed mean using an assumption-dependent probability bound.'),
    formula: 'Pr[∃j: |R̂ⱼ − Rⱼ| ≥ ε] ≤ 2M exp(−2mε²)',
    diagram: t('같은 표본 수에서 후보 수에 따른 두 반지름을 비교합니다.', 'Compare the one-model and multiple-model radii at the same sample size.'),
  }],
  experiments: [
    {
      title: t('주장하는 후보 수', 'Number of models covered'),
      predict: t('100표본에서 후보가 1개와 100개이면 반지름은 얼마나 다를까요?', 'How do radii differ for one and 100 candidates at 100 samples?'),
      action: t('M을 1에서 100으로 올리세요.', 'Raise M from one to 100.'),
      explain: t('반지름이 약 0.136에서 약 0.204로 증가합니다.', 'The radius increases from about 0.136 to about 0.204.'),
      preset: 0,
    },
    {
      title: t('독립 표본 늘리기', 'Add independent observations'),
      predict: t('표본 수를 네 배로 하면 반지름은 어떻게 될까요?', 'What happens to the radius if the sample count quadruples?'),
      action: t('m을 100에서 400으로 올리세요.', 'Raise m from 100 to 400.'),
      explain: t('같은 M과 δ에서 반지름이 절반이 됩니다.', 'The radius halves at fixed M and δ.'),
      preset: 1,
    },
  ],
};
