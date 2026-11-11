import type { LearningGuide, LearningUnit } from './interactive-book';

export const modelDiscrepancyId = 'regime-dependent-model-discrepancy' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const modelDiscrepancyUnit: LearningUnit = {
  id: modelDiscrepancyId,
  title: t('조건에 따른 모형 불일치', 'Regime-Dependent Model Discrepancy'),
  subtitle: t('평형값을 생략한 모형이 추정과 외삽을 어떻게 바꾸는지 살펴봅니다.', 'Explore how an omitted equilibrium affects calibration and extrapolation.'),
  prerequisites: t('지수 감쇠, 매개변수 교정과 관측 잔차를 복습하세요.', 'Review exponential decay, parameter calibration, and observation residuals.'),
  prerequisiteIds: ['observation-and-dynamics-residuals'],
  eyebrow: 'MODEL DISCREPANCY',
  formula: 'y = m(θ) + δ + ε',
  accent: 'block-order',
  questions: [
    {
      id: 'discrepancy-absorbed-rate',
      question: t('양의 평형값을 생략하고 한 시각의 관측을 맞추면 추정 감쇠율은?', 'How does omitting a positive equilibrium affect the one-reading fitted decay rate?'),
      options: [t('실제보다 작습니다.', 'It is smaller than the true rate.'), t('실제보다 큽니다.', 'It is larger than the true rate.'), t('언제나 같습니다.', 'It always equals the true rate.')],
      answer: 0,
      explanation: t('모형은 더 높은 관측값에 도달하기 위해 더 천천히 감쇠합니다.', 'The simplified model decays more slowly to reach the higher observation.'),
    },
    {
      id: 'discrepancy-follow-up',
      question: t('두 모형이 교정 관측을 모두 맞추면 다음 관측을 어디에서 하는 것이 좋을까요?', 'If both models match the calibration readings, where should a follow-up measurement be taken?'),
      options: [t('잡음에 비해 두 예측 차이가 큰 시각', 'At a time when predictions differ substantially relative to noise'), t('두 예측이 같은 교정 시각만', 'Only at the calibration time where predictions agree'), t('시각과 무관합니다.', 'Time makes no difference.')],
      answer: 0,
      explanation: t('예측이 갈라지는 곳에서 가능한 설명을 구분할 수 있습니다.', 'A reading where predictions diverge can distinguish the explanations.'),
    },
  ],
};

export const modelDiscrepancyGuide: LearningGuide = {
  intuition: t('빠진 물리 효과를 매개변수가 대신하면 교정 관측은 맞아도 새 조건에서 오차가 생길 수 있습니다.', 'A fitted parameter can absorb missing physics at calibration and leave errors under new conditions.'),
  steps: [{
    title: t('빠진 평형값', 'The omitted equilibrium'),
    explanation: t('평형값이 0인 모형은 한 측정점을 맞추기 위해 감쇠율을 바꿉니다.', 'The zero-equilibrium model adjusts its rate to match a calibration reading.'),
    formula: 'k̂ = −ln[y(1)]',
    diagram: t('교정 시각을 공유하는 두 곡선의 후기 예측을 비교합니다.', 'Compare late predictions of curves that share a calibration point.'),
  }],
  experiments: [
    {
      title: t('교정점과 검증점', 'Calibration and validation points'),
      predict: t('t=1에서 두 모형이 일치하면 t=3에서도 일치할까요?', 'If the models agree at t=1, will they agree at t=3?'),
      action: t('평형값을 0.25로 두고 t=1과 t=3의 차를 비교하세요.', 'With equilibrium 0.25, compare the gaps at t=1 and t=3.'),
      explain: t('교정점의 차는 0이고 t=3의 차는 약 0.142입니다.', 'The calibration gap is zero and the gap at t=3 is about 0.142.'),
      preset: 0,
    },
    {
      title: t('모형의 가정 만족시키기', 'Satisfy the model assumption'),
      predict: t('실제 평형값이 0이면 두 감쇠율이 같아질까요?', 'Do the two decay rates coincide when the true equilibrium is zero?'),
      action: t('평형값을 0으로 줄이세요.', 'Reduce the equilibrium to zero.'),
      explain: t('두 모형이 같은 식이 되어 추정 감쇠율이 1로 돌아옵니다.', 'The models coincide and the fitted rate returns to one.'),
      preset: 1,
    },
  ],
};
