import type { LearningGuide, LearningUnit } from './interactive-book';

export const uncertaintyPropagationId = 'correlation-in-measurement-uncertainty' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const uncertaintyPropagationUnit: LearningUnit = {
  id: uncertaintyPropagationId,
  title: t('측정 불확실성과 상관', 'Correlation in Measurement Uncertainty'),
  subtitle: t('공유 오차가 합과 차의 불확실성을 어떻게 바꾸는지 살펴봅니다.', 'Explore how shared errors change uncertainty in sums and differences.'),
  prerequisites: t('분산과 공분산, 선형 결합을 복습하세요.', 'Review variance, covariance, and linear combinations.'),
  prerequisiteIds: ['optimal-sensor-allocation'],
  eyebrow: 'UNCERTAINTY PROPAGATION',
  formula: 'Var(wᵀx) = wᵀΣw',
  accent: 'block-order',
  questions: [
    {
      id: 'propagation-positive-correlation',
      question: t('양의 상관을 추가하면 합과 차의 분산은 어떻게 변하나요?', 'How does positive correlation change the two variances?'),
      options: [t('합의 분산은 커지고 차의 분산은 작아집니다.', 'Sum variance increases and difference variance decreases.'), t('두 분산 모두 커집니다.', 'Both variances increase.'), t('두 분산 모두 같습니다.', 'Both variances stay fixed.')],
      answer: 0,
      explanation: t('공분산 항은 합에 더해지고 차에서 빠집니다.', 'The covariance term is added for the sum and subtracted for the difference.'),
    },
    {
      id: 'propagation-zero-gradient',
      question: t('입력 평균에서 출력 함수의 기울기가 0이면 어떤 계산이 필요할 수 있나요?', 'If the output function has zero gradient at the input mean, what may be needed?'),
      options: [t('곡률이나 전체 분포를 반영한 전파', 'Propagation accounting for curvature or the full distribution'), t('출력 분산을 항상 0으로 설정', 'Always set output variance to zero'), t('공분산을 항상 양수로 설정', 'Always make covariance positive')],
      answer: 0,
      explanation: t('제곱 함수처럼 접선에 보이지 않는 곡률이 출력 변동을 만들 수 있습니다.', 'Curvature can generate variation that the tangent misses, as with a square function.'),
    },
  ],
};

export const uncertaintyPropagationGuide: LearningGuide = {
  intuition: t('같이 움직이는 오차는 두 값을 더할 때 남고 두 값을 뺄 때 서로 지워질 수 있습니다.', 'Shared errors reinforce in sums and can cancel in differences.'),
  steps: [{
    title: t('상관을 분산으로 전달하기', 'Propagate correlation into variance'),
    explanation: t('공분산에 붙는 부호가 합과 차에서 달라집니다.', 'The covariance term has opposite signs for the sum and difference.'),
    formula: 'Var(x₁ ± x₂) = σ₁² + σ₂² ± 2ρσ₁σ₂',
    diagram: t('공분산 타원과 투영 방향을 비교합니다.', 'Compare the covariance ellipse with the projection directions.'),
  }],
  experiments: [
    {
      title: t('공유 오차 제거', 'Cancel a shared error'),
      predict: t('표준편차가 같고 상관이 1이면 차의 분산은 얼마일까요?', 'What is the difference variance when the scales match and correlation is one?'),
      action: t('σ₂를 1로 유지하고 ρ를 1로 올리세요.', 'Keep σ₂ at one and raise ρ to one.'),
      explain: t('두 센서가 같은 오차만큼 움직여 차에서 상쇄됩니다.', 'Both sensors shift by the same error, which cancels in their difference.'),
      preset: 0,
    },
    {
      title: t('오차 척도가 다를 때', 'Unequal error scales'),
      predict: t('완전 상관에서 σ₂가 2이면 차에 오차가 남을까요?', 'With perfect correlation and σ₂=2, does any difference error remain?'),
      action: t('ρ를 1로 두고 σ₂를 2로 올리세요.', 'Keep ρ at one and raise σ₂ to two.'),
      explain: t('오차가 함께 움직여도 크기가 달라 차의 분산이 1입니다.', 'The errors move together at different scales, leaving a difference variance of one.'),
      preset: 1,
    },
  ],
};
