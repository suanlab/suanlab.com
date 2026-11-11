import type { LearningGuide, LearningUnit } from './interactive-book';

export const rindlerObservablesId = 'accelerated-coordinates-and-invariant-readings' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const rindlerObservablesUnit: LearningUnit = {
  id: rindlerObservablesId,
  title: t('가속 좌표와 불변 측정량', 'Accelerated Coordinates and Invariant Readings'),
  subtitle: t('평탄한 시공간에서 좌표 속도와 시계·가속도계의 읽기를 비교합니다.', 'Compare coordinate speeds with clock and accelerometer readings in flat spacetime.'),
  prerequisites: t('민코프스키 계량과 고유 시간을 복습하세요.', 'Review the Minkowski metric and proper time.'),
  prerequisiteIds: ['minkowski-metric-and-proper-time', 'spacetime-observation-and-redshift'],
  eyebrow: 'COORDINATES AND OBSERVABLES',
  formula: 'T = ρ sinh η, X = ρ cosh η',
  accent: 'block-order',
  questions: [
    {
      id: 'rindler-clock-reading',
      question: t('ρ가 일정한 가속 시계가 Δη 동안 기록하는 고유 시간은?', 'What proper time does an accelerated clock at fixed ρ accumulate over Δη?'),
      options: [t('ρΔη', 'ρΔη'), t('항상 Δη', 'Always Δη'), t('Δη/ρ', 'Δη/ρ')],
      answer: 0,
      explanation: t('계량에서 dρ=0으로 놓으면 dτ=ρdη입니다.', 'Setting dρ=0 in the metric gives dτ=ρdη.'),
    },
    {
      id: 'rindler-local-light-speed',
      question: t('좌표 빛 속력 |dρ/dη|가 ρ에 따라 달라질 때 국소 빛 속력은?', 'If the coordinate light speed |dρ/dη| changes with ρ, what is the local light speed?'),
      options: [t('c로 일정하다.', 'It remains c.'), t('ρc가 된다.', 'It equals ρc.'), t('항상 0이다.', 'It is always zero.')],
      answer: 0,
      explanation: t('같은 관측자의 고유 시간으로 나누면 ρ가 상쇄됩니다.', 'Dividing by the observer’s proper time cancels the factor of ρ.'),
    },
  ],
};

export const rindlerObservablesGuide: LearningGuide = {
  intuition: t('두 좌표가 같은 사건을 가리켜도 시계 읽기는 경로의 고유 시간으로 계산합니다.', 'Two charts can label the same event. A clock reading comes from proper time along its path.'),
  steps: [{
    title: t('같은 계량의 두 표현', 'Two expressions for the same metric'),
    explanation: t('가속 좌표에서는 시간 계수가 위치에 따라 변합니다. 시공간의 곡률은 0입니다.', 'The time coefficient depends on position in accelerated coordinates. Spacetime curvature remains zero.'),
    formula: '−dT² + dX² = −ρ²dη² + dρ²',
    diagram: t('가속하는 관측자의 세계선은 관성 좌표에서 쌍곡선입니다.', 'The accelerated observer follows a hyperbola in inertial coordinates.'),
  }],
  experiments: [
    {
      title: t('좌표값과 시계 읽기', 'Coordinate values and clock readings'),
      predict: t('같은 Δη에서 ρ를 세 배 키우면 시계 읽기는 어떻게 바뀔까요?', 'How does the clock reading change when ρ triples at fixed Δη?'),
      action: t('η=1에서 ρ를 1과 3으로 바꾸어 읽으세요.', 'At η=1, compare ρ=1 with ρ=3.'),
      explain: t('가속 시계의 고유 시간은 세 배이고 가속도계 읽기는 1/3입니다.', 'The accelerated clock time triples and the accelerometer reading drops to one third.'),
      preset: 0,
    },
    {
      title: t('좌표 빛 속도와 국소 측정', 'Coordinate light speed and local measurement'),
      predict: t('좌표 빛 속도가 변하면 국소 속력도 달라질까요?', 'Does the local light speed change with the coordinate light speed?'),
      action: t('ρ를 움직이고 두 속력 표시를 비교하세요.', 'Change ρ and compare the two speed readouts.'),
      explain: t('국소 측정은 고유 시간을 써서 속력이 1입니다.', 'The local measurement uses proper time and gives speed one in these units.'),
      preset: 1,
    },
  ],
};
