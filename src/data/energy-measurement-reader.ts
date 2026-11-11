import type { LearningGuide, LearningUnit } from './interactive-book';

export const energyMeasurementId = 'energy-bounds-and-system-measurements' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const energyMeasurementUnit: LearningUnit = {
  id: energyMeasurementId,
  title: t('에너지 하한과 시스템 실측', 'Energy Bounds and System Measurements'),
  subtitle: t('쌍별 실험에서 공통 오차를 지우고 반복 오차와 교정 오차를 구분합니다.', 'Use paired experiments to cancel common offsets and distinguish repeatability from calibration.'),
  prerequisites: t('물리적 하한, 평균과 표본 분산을 복습하세요.', 'Review physical bounds, means, and sample variance.'),
  prerequisiteIds: ['landauer-erasure-and-its-assumptions'],
  eyebrow: 'ENERGY MEASUREMENT',
  formula: 'dᵢ = E_A,ᵢ − E_B,ᵢ',
  accent: 'block-order',
  questions: [
    {
      id: 'energy-measurement-paired-offset',
      question: t('두 실행에 같은 가산 오차가 생기면 쌍별 차이는 어떻게 변하나요?', 'What happens to a paired difference if the same additive offset affects both runs?'),
      options: [t('같습니다.', 'It stays the same.'), t('두 배가 됩니다.', 'It doubles.'), t('항상 0입니다.', 'It always becomes zero.')],
      answer: 0,
      explanation: t('같은 수를 두 관측에 더해도 서로 빼면 그 항이 없어집니다.', 'Subtracting the two observations cancels a shared additive term.'),
    },
    {
      id: 'energy-measurement-scale',
      question: t('같은 계측기로 반복 횟수만 늘리면 공통 배율 불확실성은 어떻게 되나요?', 'What happens to common gain uncertainty if you only repeat more runs on the same meter?'),
      options: [t('반복 횟수만 늘려도 그 교정 오차는 남습니다.', 'The calibration uncertainty persists.'), t('횟수에 반비례해 사라집니다.', 'It vanishes in inverse proportion to the count.'), t('항상 0으로 측정됩니다.', 'It always becomes zero.')],
      answer: 0,
      explanation: t('반복은 무작위 변동을 더 잘 추정하며 계측기의 배율을 다시 교정하지는 않습니다.', 'Repetition better estimates random variation, while the meter retains the same calibration.'),
    },
  ],
};

export const energyMeasurementGuide: LearningGuide = {
  intuition: t('같은 조건에서 두 장치를 나란히 재면 블록마다 함께 변한 효과를 차이에서 지울 수 있습니다.', 'Measuring both devices in matched blocks can cancel effects shared within a block.'),
  steps: [{
    title: t('블록별 차이', 'Differences within blocks'),
    explanation: t('각 블록에서 A와 B를 먼저 빼고 그 차이의 평균과 표준오차를 구합니다.', 'Subtract A and B within each block, then compute the mean and standard error of the differences.'),
    formula: 'SE = s_d / √n',
    diagram: t('블록별 에너지와 차이를 함께 보면 공통 이동의 영향을 구분할 수 있습니다.', 'Plotting each energy and its paired difference reveals the effect of common offsets.'),
  }],
  experiments: [
    {
      title: t('공통 오프셋 더하기', 'Add a common offset'),
      predict: t('각 쌍에 같은 오프셋을 더하면 평균 차이는 어떻게 될까요?', 'How does a shared offset in each pair affect the mean difference?'),
      action: t('가산 오프셋 배율을 늘리고 그래프와 평균 차이를 비교하세요.', 'Increase the additive offset multiplier and compare the plot with the mean difference.'),
      explain: t('두 에너지 곡선은 움직이지만 블록별 차이는 같습니다.', 'Both energy traces shift while the paired differences stay the same.'),
      preset: 0,
    },
    {
      title: t('계측기 교정 불확실성 바꾸기', 'Change meter calibration uncertainty'),
      predict: t('교정 불확실성을 두 배로 하면 반복 표준오차도 두 배가 될까요?', 'Does doubling calibration uncertainty also double the repeatability standard error?'),
      action: t('교정 불확실성을 1%에서 2%로 올리세요.', 'Raise calibration uncertainty from 1% to 2%.'),
      explain: t('관측값이 같아 반복 표준오차는 같고 교정 항만 두 배가 됩니다.', 'The observations stay fixed, so the repeatability standard error stays fixed while the calibration term doubles.'),
      preset: 1,
    },
  ],
};
