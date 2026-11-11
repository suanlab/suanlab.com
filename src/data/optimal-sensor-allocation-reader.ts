import type { LearningGuide, LearningUnit } from './interactive-book';

export const optimalSensorAllocationId = 'optimal-sensor-allocation' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const optimalSensorAllocationUnit: LearningUnit = {
  id: optimalSensorAllocationId,
  title: t('최적 센서 관측 배분', 'Optimal Sensor Allocation'),
  subtitle: t('한정된 관측 예산에서 분산의 합과 불확실성 타원의 넓이를 비교합니다.', 'Compare total variance and uncertainty ellipse area under a fixed measurement budget.'),
  prerequisites: t('표본평균의 분산과 공분산 행렬을 복습하세요.', 'Review variance of the sample mean and covariance matrices.'),
  prerequisiteIds: ['parameter-identifiability-and-sensor-design'],
  eyebrow: 'EXPERIMENT DESIGN',
  formula: 'Σ = diag(σₐ²/nₐ, σᵦ²/nᵦ)',
  accent: 'block-order',
  questions: [
    {
      id: 'allocation-trace-ratio',
      question: t('연속 A-최적 배분에서 센서 B의 표준편차가 A의 세 배이면 관측 횟수 비는?', 'If sensor B has three times the noise standard deviation of A, what is the continuous A-optimal count ratio?'),
      options: [t('A:B = 1:3', 'A:B = 1:3'), t('A:B = 1:9', 'A:B = 1:9'), t('A:B = 1:1', 'A:B = 1:1')],
      answer: 0,
      explanation: t('분산 합을 미분하면 횟수의 비가 표준편차의 비와 같습니다.', 'Differentiating total variance makes the count ratio equal to the standard deviation ratio.'),
    },
    {
      id: 'allocation-determinant-balance',
      question: t('이 모형에서 전체 횟수를 고정하면 어떤 배분이 공분산 행렬식을 최소화하나요?', 'At fixed total count, what allocation minimizes the covariance determinant in this model?'),
      options: [t('가능한 한 균등한 배분', 'The most balanced allocation'), t('모두 센서 A에 배분', 'All readings on sensor A'), t('잡음 분산에 비례한 배분', 'Allocation proportional to noise variances')],
      answer: 0,
      explanation: t('두 센서 횟수의 곱이 최대일 때 공분산 행렬식이 최소입니다.', 'The determinant is smallest when the product of the two counts is largest.'),
    },
  ],
};

export const optimalSensorAllocationGuide: LearningGuide = {
  intuition: t('같은 관측 예산에서도 목표에 따라 더 좋은 배분이 달라집니다.', 'The objective changes which allocation best uses the same measurement budget.'),
  steps: [{
    title: t('공분산에 적용하는 두 기준', 'Two criteria on a covariance matrix'),
    explanation: t('대각합은 각 분산을 더하고, 행렬식은 이 모형에서 각 분산을 곱합니다.', 'The trace sums the variances, and the determinant multiplies them in this model.'),
    formula: 'tr(Σ) = Vₐ + Vᵦ,  det(Σ) = VₐVᵦ',
    diagram: t('같은 횟수 축에서 선택한 기준의 최솟값을 표시합니다.', 'The selected criterion is plotted over the same measurement counts.'),
  }],
  experiments: [
    {
      title: t('두 목적함수 비교', 'Compare the two objectives'),
      predict: t('σᵦ=3에서 대각합과 행렬식이 같은 배분을 고를까요?', 'With σᵦ=3, do trace and determinant select the same allocation?'),
      action: t('관측 12회에서 최적 배분 버튼을 누르고 기준을 바꾸세요.', 'At a budget of 12, apply the optimum and switch between criteria.'),
      explain: t('대각합은 (3,9), 행렬식은 (6,6)에서 가장 작습니다.', 'Trace is minimized at (3,9) and determinant at (6,6).'),
      preset: 0,
    },
    {
      title: t('같은 잡음 크기', 'Equal noise levels'),
      predict: t('두 센서의 잡음이 같으면 두 기준의 최적 배분은 어떻게 될까요?', 'What happens to both optimal allocations when the sensors have equal noise?'),
      action: t('σᵦ를 1로 줄이고 두 기준의 최적 배분을 비교하세요.', 'Set σᵦ to one and compare the two optima.'),
      explain: t('12회에서는 두 기준 모두 (6,6)을 고릅니다.', 'At a budget of 12 both criteria choose (6,6).'),
      preset: 1,
    },
  ],
};
