import type { LearningGuide, LearningUnit } from './interactive-book';

export const isingDecimationId = 'renormalization-and-real-space-coarse-graining' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const isingDecimationUnit: LearningUnit = {
  id: isingDecimationId,
  title: t('재규격화와 실공간 조대화', 'Renormalization and Real-Space Coarse Graining'),
  subtitle: t('스핀을 합해 고리의 유효 결합을 구하고 네 스핀 항을 버리는 근사를 비교합니다.', 'Sum out spins to recover an effective ring coupling and measure the effect of dropping a four-spin interaction.'),
  prerequisites: t('이징 상관 길이, 분배함수와 주변분포를 복습하세요.', 'Review Ising correlation lengths, partition functions, and marginal distributions.'),
  prerequisiteIds: ['criticality-universality-and-finite-size-scaling', 'periodic-ising-ring-correlation-length'],
  eyebrow: 'REAL-SPACE RENORMALIZATION',
  formula: 'tanh K′ = (tanh K)²',
  accent: 'block-order',
  questions: [
    {
      id: 'ising-decimation-normalization',
      question: t('제거한 스핀에서 생긴 공통 인자 A는 어떤 값에 필요한가요?', 'Which quantity requires the common factor A from an eliminated spin?'),
      options: [t('원래 계의 분배함수', 'The original partition function'), t('정규화한 두 배치의 확률 비율', 'The ratio of two normalized probabilities'), t('새 격자의 간격', 'The new lattice spacing')],
      answer: 0,
      explanation: t('공통 인자는 모든 배치의 가중치에 곱해집니다. 확률을 정규화하면 상쇄되고 분배함수에는 남습니다.', 'The common factor multiplies every weight. It cancels in normalized probabilities and remains in the partition function.'),
    },
    {
      id: 'ising-decimation-generated-term',
      question: t('이웃이 네 개인 중심 스핀을 합하면 어떤 항이 생길 수 있나요?', 'Which term can appear after summing a central spin with four neighbors?'),
      options: [t('네 이웃 부호의 곱', 'A product of all four neighbor spins'), t('외부장 0에서 홀수 개 부호의 곱', 'A product of an odd number of spins at zero field'), t('시간 미분', 'A time derivative')],
      answer: 0,
      explanation: t('부호 반전 대칭은 짝수 차수의 곱을 허용합니다. 네 스핀 항을 버리면 배치 사이의 확률이 달라집니다.', 'Spin reversal permits even products. Dropping the four-spin term changes probabilities across configurations.'),
    },
  ],
};

export const isingDecimationGuide: LearningGuide = {
  intuition: t('가운데 스핀의 두 경우를 모두 합하면 남은 두 스핀 사이의 새 가중치가 생깁니다.', 'Sum over both values of the middle spin to obtain a new weight for the two retained spins.'),
  steps: [
    {
      title: t('가운데 스핀을 합하기', 'Sum the middle spin'),
      explanation: t('남긴 두 스핀이 같을 때와 다를 때의 가중치 비율로 유효 결합을 계산합니다.', 'The weight ratio for aligned and opposed retained spins determines the effective coupling.'),
      formula: 'K′ = ½ log cosh(2K)',
      diagram: t('가운데 스핀은 두 부호를 합합니다.', 'Sum both signs of the middle spin.'),
    },
  ],
  experiments: [
    {
      title: t('길이와 분배함수 보존', 'Preserved length and partition function'),
      predict: t('결합과 격자 간격이 바뀔 때 어떤 값이 유지될까요?', 'Which quantities stay fixed as the coupling and lattice spacing change?'),
      action: t('초기 결합을 바꾸고 단계별 aξ와 복원된 log Z를 비교하세요.', 'Change the initial coupling and compare aξ and the recovered log Z across steps.'),
      explain: t('격자 단위 상관 길이는 절반이 되고 간격은 두 배가 됩니다. 누적한 가중치 인자가 분배함수를 복원합니다.', 'The correlation length in lattice units halves and the spacing doubles. Accumulated weight factors recover the partition function.'),
      preset: 0,
    },
    {
      title: t('네 스핀 항을 버리는 근사', 'Truncating the four-spin interaction'),
      predict: t('결합을 키우면 두 분포의 차이가 어떻게 바뀔까요?', 'How does the difference between the distributions change as you strengthen coupling?'),
      action: t('결합을 0과 0.4로 바꾸고 네 스핀 계수와 총변동 거리를 확인하세요.', 'Compare the four-spin coefficient and total variation at coupling 0 and 0.4.'),
      explain: t('결합이 없으면 두 분포가 같습니다. 0.4에서는 네 스핀 항의 제거가 배치 확률을 바꿉니다.', 'The distributions coincide at zero coupling. At 0.4, omitting the four-spin term shifts configuration probabilities.'),
      preset: 1,
    },
  ],
};
