import type { LearningGuide, LearningUnit } from './interactive-book';

export const phononDispersionId = 'lattice-vibrations-and-phonon-dispersion' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const phononDispersionUnit: LearningUnit = {
  id: phononDispersionId,
  title: t('격자 진동과 포논 분산', 'Lattice Vibrations and Phonon Dispersion'),
  subtitle: t('두 원자가 반복되는 사슬에서 음향·광학 모드의 주파수와 변위를 비교합니다.', 'Compare acoustic and optical frequencies and displacements in a diatomic chain.'),
  prerequisites: t('조화 진동, 푸리에 모드와 대칭 고윳값 문제를 복습하세요.', 'Review harmonic motion, Fourier modes, and symmetric eigenproblems.'),
  prerequisiteIds: ['field-fourier-modes-lattice-continuum'],
  eyebrow: 'LATTICES AND PHONONS',
  formula: 'D(q)v = ω²v',
  accent: 'block-order',
  questions: [
    {
      id: 'phonon-acoustic-zero',
      question: t('q=0에서 음향 모드의 주파수가 0인 이유는?', 'Why is the acoustic frequency zero at q=0?'),
      options: [t('같은 변위가 결합을 늘리지 않기 때문입니다.', 'Equal displacements stretch no spring.'), t('두 원자가 항상 반대로 움직이기 때문입니다.', 'The two atoms always move in opposite directions.'), t('질량이 0이기 때문입니다.', 'The masses are zero.')],
      answer: 0,
      explanation: t('사슬 전체의 병진은 용수철의 길이를 바꾸지 않습니다.', 'Translating the entire chain preserves every spring length.'),
    },
    {
      id: 'phonon-equal-mass-edge',
      question: t('두 질량이 같으면 두 원자 셀의 영역 끝에서 어떻게 되나요?', 'What happens at the two-atom zone edge when the masses are equal?'),
      options: [t('두 주파수가 만납니다.', 'The two frequencies meet.'), t('간격이 가장 커집니다.', 'The gap becomes largest.'), t('음향 모드만 남습니다.', 'Only the acoustic mode survives.')],
      answer: 0,
      explanation: t('같은 용수철과 같은 질량에서는 실제 주기가 셀 길이의 절반입니다.', 'With identical masses and springs the true period is half the chosen cell length.'),
    },
  ],
};

export const phononDispersionGuide: LearningGuide = {
  intuition: t('같은 방향의 이동과 결합을 늘리는 상대 운동은 서로 다른 정상 모드를 이룹니다.', 'Collective translation and relative motion that stretches bonds give different normal modes.'),
  steps: [{
    title: t('질량 가중 고윳값', 'Mass-weighted eigenvalues'),
    explanation: t('고윳값은 제곱 주파수이며 고유벡터에서 원자의 변위를 구합니다.', 'Eigenvalues are squared frequencies, and eigenvectors give the atomic displacements.'),
    formula: 'D(q)v = ω²v',
    diagram: t('같은 사슬에서 파수에 따라 두 주파수가 달라집니다.', 'The two frequencies vary with wave number along the same chain.'),
  }],
  experiments: [
    {
      title: t('영역 끝의 간격', 'Gap at the zone edge'),
      predict: t('두 질량이 같으면 영역 끝의 두 주파수는 어떻게 될까요?', 'What happens to the zone-edge frequencies when both masses are equal?'),
      action: t('q/π=1에서 mᵦ를 4에서 1로 바꾸세요.', 'At q/π=1, change mᵦ from 4 to 1.'),
      explain: t('두 고윳값이 같아지면 그 점에서 가지별 변위 방향은 하나로 정해지지 않습니다.', 'When the eigenvalues coincide, separate branch directions are not uniquely determined there.'),
      preset: 0,
    },
    {
      title: t('같은 방향과 반대 방향의 변위', 'In-phase and opposite displacements'),
      predict: t('q=0에서 어느 모드가 두 원자를 반대 방향으로 움직일까요?', 'At q=0, which mode moves the two atoms in opposite directions?'),
      action: t('q/π를 0으로 두고 음향과 광학 가지를 번갈아 선택하세요.', 'Set q/π to zero and switch between acoustic and optical branches.'),
      explain: t('광학 모드는 질량 중심을 고정하면서 결합을 늘렸다 줄입니다.', 'The optical mode changes bond lengths while keeping the center of mass fixed.'),
      preset: 1,
    },
  ],
};
