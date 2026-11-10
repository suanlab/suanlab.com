import type { LearningGuide, LearningUnit } from './interactive-book';
import { scalarGibbsUnitId } from '@/lib/learning-scalar-gibbs-input';

const t = (ko: string, en: string) => ({ ko, en });

export const scalarGibbsPresetLabels = [
  t('대칭 결합', 'Symmetric coupling'),
  t('독립 격자점', 'Independent sites'),
  t('서로 다른 외부장', 'Unequal sources'),
  t('순수 사차 퍼텐셜', 'Pure quartic potential'),
];

export const scalarGibbsUnit: LearningUnit = {
  id: scalarGibbsUnitId,
  title: t('사차 스칼라 장의 깁스 분포', 'Gibbs Distributions of Quartic Scalar Fields'),
  subtitle: t('한두 격자점의 분포를 그려 보고 정규화와 공분산의 오차 범위를 계산합니다.', 'Plot Gibbs densities on one or two lattice sites and bound normalization and covariance.'),
  prerequisites: t('자유에너지와 국소 질서변수, 사차 적분을 복습하세요.', 'Review free energy, local order parameters, and quartic integrals.'),
  prerequisiteIds: ['finite-field-measure-and-thermal-free-energy', 'microscopic-blocks-and-local-order-parameters', 'quartic-integral-asymptotic-truncation'],
  eyebrow: 'STATISTICAL FIELD THEORY',
  formula: 'A = Σᵢ(a xᵢ⁴ + b xᵢ² − hᵢxᵢ) + c(x₁ − x₂)²/2',
  accent: 'block-order',
  questions: [
    {
      id: 'scalar-gibbs-flip',
      question: t('외부장이 0이고 두 점이 결합할 때 대칭이 보장하는 것은 무엇인가요?', 'What does symmetry guarantee for coupled sites with zero sources?'),
      options: [t('각 점의 평균이 0이다.', 'Each site has zero mean.'), t('각 점의 분산이 0이다.', 'Each site has zero variance.'), t('두 점의 공분산이 0이다.', 'The cross-covariance is zero.')],
      answer: 0,
      explanation: t('두 좌표를 동시에 뒤집으면 가중치는 같고 일차 모멘트의 부호는 바뀝니다. 두 좌표의 곱은 부호가 유지됩니다.', 'Flipping both coordinates preserves the weight and reverses each first moment. The product of the coordinates keeps its sign.'),
    },
    {
      id: 'scalar-gibbs-tail',
      question: t('적분 반경 R을 고정하고 분할 수만 늘리면 어느 오차 경계에 영향을 주나요?', 'Which error bound changes when you increase panel count at fixed radius R?'),
      options: [t('상자 내부 적분 오차', 'In-box quadrature error'), t('상자 외부 꼬리 상계', 'Outside-box tail bound'), t('둘 다 같은 비율로 감소', 'Both shrink by the same factor')],
      answer: 0,
      explanation: t('분할 수는 상자 내부 셀의 폭을 바꿉니다. 꼬리 상계는 반경과 사차항으로 결정됩니다.', 'Panel count changes the cell width inside the box. The tail bound depends on the radius and the quartic coefficient.'),
    },
  ],
};

export const scalarGibbsGuide: LearningGuide = {
  intuition: t('두 격자점의 값을 평면에 놓으면 결합이 대각선 주변에 확률을 모으는 모습이 보입니다.', 'Plot the values at two sites on a plane to see how coupling concentrates probability near the diagonal.'),
  steps: [
    {
      title: t('두 좌표의 결합', 'Coupling two coordinates'),
      explanation: t('두 값이 달라질수록 에너지가 올라갑니다. 대각선 위의 점에서는 결합 에너지가 0입니다.', 'The energy rises as the site values differ. Coupling contributes zero energy along the diagonal.'),
      formula: 'Aedge = c(x₁ − x₂)²/2',
      diagram: t('밝을수록 볼츠만 가중치가 큽니다.', 'Brighter cells carry larger Boltzmann weights.'),
    },
    {
      title: t('적분 오차와 잘린 꼬리', 'Quadrature error and truncated tails'),
      explanation: t('상자 안의 적분을 구간으로 계산하고 상자 밖에서 놓친 가중치는 별도의 상계로 더합니다.', 'Enclose the in-box integral and add a separate upper bound for the omitted weight.'),
      formula: 'Zs ∈ [Lbox, Ubox + Btail]',
      diagram: t('표시 범위는 선택한 적분 상자입니다.', 'The plot spans the selected integration box.'),
    },
  ],
  experiments: [
    {
      title: t('평균이 0인 두 점의 공분산', 'Covariance between zero-mean sites'),
      predict: t('두 점의 평균이 0일 때도 두 값이 함께 움직일 수 있을까요?', 'Can the values move together when both means vanish?'),
      action: t('대칭 결합과 독립 격자점을 차례로 계산하고 공분산 구간을 비교하세요.', 'Run symmetric coupling and independent sites, then compare the covariance intervals.'),
      explain: t('동시 부호 반전은 평균을 0으로 만듭니다. 결합은 같은 부호의 값에 더 큰 가중치를 줄 수 있습니다.', 'Global sign reversal forces the means to zero. Coupling can give greater weight to values with the same sign.'),
      preset: 0,
    },
    {
      title: t('상자와 격자 해상도', 'Box and grid resolution'),
      predict: t('분할 수를 늘리면 상자 밖 꼬리의 상계도 줄어들까요?', 'Does increasing panel count shrink the outside-box tail bound?'),
      action: t('R을 4에 두고 분할 수를 32에서 64로 바꿉니다. 다음으로 R을 바꿔 두 오차를 다시 확인하세요.', 'Hold R at 4 and increase the panel count from 32 to 64. Then change R and inspect both error terms again.'),
      explain: t('분할 수와 반경은 서로 다른 오차에 영향을 줍니다. 반경을 넓히면서 셀의 폭도 확인하세요.', 'Panel count and radius influence different errors. When increasing the radius, check the resulting cell width as well.'),
      preset: 0,
    },
  ],
};
