import type { LearningGuide, LearningUnit } from './interactive-book';

export const fourMomentumId = 'relativistic-kinematics-four-momentum' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const fourMomentumUnit: LearningUnit = {
  id: fourMomentumId,
  title: t('상대론적 운동학과 사원운동량', 'Relativistic Kinematics and Four-Momentum'),
  subtitle: t('고정 표적과 마주 오는 빔의 충돌 문턱을 비교합니다.', 'Compare kinematic thresholds for a fixed target and opposing beams.'),
  prerequisites: t('로런츠 변환과 민코프스키 곱을 복습하세요.', 'Review Lorentz boosts and the Minkowski product.'),
  prerequisiteIds: ['minkowski-metric-and-proper-time'],
  eyebrow: 'RELATIVISTIC KINEMATICS',
  formula: 'W² = Eₜₒₜ² − |pₜₒₜ|²',
  accent: 'block-order',
  questions: [
    {
      id: 'four-momentum-center-frame',
      question: t('운동량 중심계에서 0이 되는 양은?', 'Which quantity vanishes in the center-of-momentum frame?'),
      options: [t('총 공간 운동량', 'Total spatial momentum'), t('총에너지', 'Total energy'), t('계의 불변 질량', 'The invariant mass of the system')],
      answer: 0,
      explanation: t('공간 운동량이 상쇄되며 총에너지는 불변 질량 W와 같습니다.', 'The spatial momenta cancel and the total energy equals the invariant mass W.'),
    },
    {
      id: 'four-momentum-threshold',
      question: t('후보 최종 상태의 질량 합과 비교할 양은?', 'What should be compared with the sum of the final particle masses?'),
      options: [t('운동량 중심계 에너지 W', 'Center-of-momentum energy W'), t('실험실의 총 운동에너지', 'Total lab kinetic energy'), t('입사 입자의 운동량 크기', 'Magnitude of projectile momentum')],
      answer: 0,
      explanation: t('문턱은 최종 입자들이 운동량 중심계에서 모두 정지하는 경우로 정합니다.', 'Threshold is the configuration where all final particles are at rest in the center-of-momentum frame.'),
    },
  ],
};

export const fourMomentumGuide: LearningGuide = {
  intuition: t('총에너지가 같아도 계 전체의 운동량이 다르면 내부에서 사용할 수 있는 에너지가 달라집니다.', 'At equal total energy, different total momenta leave different amounts of energy available internally.'),
  steps: [{
    title: t('총합의 민코프스키 노름', 'Minkowski norm of the total'),
    explanation: t('각 입자의 사원운동량을 더한 뒤 그 총합의 제곱 노름을 구합니다.', 'Add the particle four-momenta before taking the squared norm of the total.'),
    formula: 'W = √(Eₜₒₜ² − |pₜₒₜ|²)',
    diagram: t('총운동량이 큰 계에서는 같은 총에너지에서 W가 줄어듭니다.', 'At equal total energy, larger net momentum leaves a smaller W.'),
  }],
  experiments: [
    {
      title: t('총 운동에너지가 같은 두 배치', 'Two arrangements at equal kinetic energy'),
      predict: t('어느 배치에서 운동량 중심계 에너지가 더 클까요?', 'Which arrangement has greater center-of-momentum energy?'),
      action: t('입자 질량 1, 총 운동에너지 2.5에서 두 값을 비교하세요.', 'At particle mass 1 and total kinetic energy 2.5, compare both energies.'),
      explain: t('마주 오는 빔은 총운동량이 0이므로 실험실의 총에너지가 W입니다.', 'Opposing beams have zero net momentum, so their total lab energy equals W.'),
      preset: 0,
    },
    {
      title: t('문턱을 넘는지 확인', 'Check whether a threshold is met'),
      predict: t('최종 질량의 합이 3이면 총 운동에너지 1.5에서 어느 배치가 문턱을 넘을까요?', 'For a final mass sum of 3, which arrangement reaches threshold at kinetic energy 1.5?'),
      action: t('입자 질량을 1로 두고 총 운동에너지를 1.5로 옮기세요.', 'Keep particle mass at 1 and set total kinetic energy to 1.5.'),
      explain: t('마주 오는 빔은 필요한 에너지를 갖지만 고정 표적에서는 W가 3보다 작습니다.', 'Opposing beams have sufficient energy, while the fixed-target W stays below 3.'),
      preset: 1,
    },
  ],
};
