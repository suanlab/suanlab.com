import type { LearningGuide, LearningUnit } from './interactive-book';

export const isingCriticalityId = 'criticality-universality-and-finite-size-scaling' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const isingCriticalityUnit: LearningUnit = {
  id: isingCriticalityId,
  title: t('임계성·보편성과 유한 크기 척도', 'Criticality, Universality and Finite-Size Scaling'),
  subtitle: t('이징 원통에서 Binder 비율을 계산하고 유한 크기 교점을 무한 격자의 임계점과 비교합니다.', 'Compute Binder ratios on Ising cylinders and compare finite-size crossings with the infinite-lattice critical point.'),
  prerequisites: t('이징 상관 길이와 깁스 분포의 모멘트를 복습하세요.', 'Review Ising correlation lengths and Gibbs moments.'),
  prerequisiteIds: ['periodic-ising-ring-correlation-length', 'connected-generators-and-legendre-effective-actions'],
  eyebrow: 'CRITICAL PHENOMENA',
  formula: 'Uₗ = 1 − ⟨M⁴⟩/(3⟨M²⟩²)',
  accent: 'block-order',
  questions: [
    {
      id: 'ising-binder-drift',
      question: t('작은 두 원통의 Binder 곡선 교점은 무엇을 나타내나요?', 'What does the Binder crossing of two small cylinders represent?'),
      options: [t('두 유한 크기의 비율이 같은 결합', 'The coupling where the two finite-size ratios agree'), t('무한 격자의 정확한 임계점', 'The exact infinite-lattice critical point'), t('유한 격자에서 자유에너지가 불연속인 점', 'A finite-lattice free-energy discontinuity')],
      answer: 0,
      explanation: t('유한 크기 보정 때문에 교점은 크기 쌍에 따라 이동할 수 있습니다.', 'Finite-size corrections can shift the crossing as the size pair changes.'),
    },
    {
      id: 'ising-binder-independent',
      question: t('결합이 0인 독립 스핀 N개의 Binder 비율은 얼마인가요?', 'What is the Binder ratio of N independent spins at zero coupling?'),
      options: [t('2/(3N)', '2/(3N)'), t('항상 0', 'Always zero'), t('2/3', '2/3')],
      answer: 0,
      explanation: t('두 번째 모멘트는 N이고 네 번째 모멘트는 3N²−2N입니다.', 'The second moment is N and the fourth is 3N²−2N.'),
    },
  ],
};

export const isingCriticalityGuide: LearningGuide = {
  intuition: t('원통에서는 한 행의 양 끝이 이어집니다. 같은 경계에서 변의 길이를 바꿔 자화 분포가 어떻게 달라지는지 비교하세요.', 'Each row wraps around the cylinder. Compare how the magnetization distribution changes as the side length grows under the same boundary conditions.'),
  steps: [
    {
      title: t('원통의 경계', 'Cylinder boundaries'),
      explanation: t('행 방향은 주기적으로 연결되고 행을 쌓는 방향의 양 끝은 열려 있습니다.', 'Each row wraps periodically. The two ends along the stacking direction are open.'),
      formula: 'N = L²',
      diagram: t('한 행 안의 첫 스핀과 마지막 스핀 사이에도 결합이 있습니다.', 'The first and last spins within a row share a bond.'),
    },
  ],
  experiments: [
    {
      title: t('교점 이동', 'Crossing drift'),
      predict: t('크기 쌍을 바꾸면 교점이 얼마나 이동할까요?', 'How far does the crossing shift when you change size pairs?'),
      action: t('결합비 1에서 L=3,4와 L=6,8의 교점을 임계점과 비교하세요.', 'At coupling ratio 1, compare the L=3,4 and L=6,8 crossings with the critical point.'),
      explain: t('이 예제에서는 큰 크기 쌍의 교점이 기준 임계점에 더 가깝습니다.', 'In this example the larger size pair crosses closer to the reference critical point.'),
      preset: 0,
    },
    {
      title: t('결합의 방향 의존성', 'Anisotropic coupling'),
      predict: t('같은 가로 결합에서 세로 결합을 키우면 임계선의 가로 좌표는 어떻게 바뀔까요?', 'How does the horizontal critical coordinate change when you strengthen the vertical coupling at fixed horizontal coupling?'),
      action: t('결합비를 0.5에서 1.5로 바꿔 기준선과 교점의 이동을 확인하세요.', 'Change the ratio from 0.5 to 1.5 and compare the reference line and crossings.'),
      explain: t('세로 결합을 상대적으로 키우면 임계식이 더 작은 가로 결합에서 성립합니다.', 'A larger vertical-to-horizontal ratio satisfies the critical equation at a smaller horizontal coupling.'),
      preset: 1,
    },
  ],
};
