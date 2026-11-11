import type { LearningGuide, LearningUnit } from './interactive-book';

export const schwarzschildRedshiftId = 'spacetime-observation-and-redshift' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const schwarzschildRedshiftUnit: LearningUnit = {
  id: schwarzschildRedshiftId,
  title: t('시공간의 관측과 적색편이', 'Spacetime Observation and Redshift'),
  subtitle: t('빛의 주파수에서 중력과 방출기 운동의 영향을 나누어 계산합니다.', 'Separate gravity and emitter motion in a measured light frequency.'),
  prerequisites: t('고유 시간, 로런츠 변환과 계량을 복습하세요.', 'Review proper time, Lorentz boosts and the metric.'),
  prerequisiteIds: ['einstein-equations-trace-and-sources', 'special-relativity-lorentz-boosts'],
  eyebrow: 'SPACETIME OBSERVABLES',
  formula: 'ω = −g(u,k)',
  accent: 'block-order',
  questions: [
    {
      id: 'redshift-static-outer',
      question: t('방출기와 수신기가 모두 정지하고 수신기가 더 바깥에 있으면 주파수비는?', 'For static emission and reception, with the receiver farther out, what is the frequency ratio?'),
      options: [t('수신/방출 비가 1보다 작다.', 'Received/emitted is below one.'), t('수신/방출 비가 1보다 크다.', 'Received/emitted is above one.'), t('항상 1이다.', 'It is always one.')],
      answer: 0,
      explanation: t('f(r)가 바깥으로 갈수록 커서 √(fₑ/fₒ)가 1보다 작습니다.', 'The lapse factor f(r) grows outward, making √(fₑ/fₒ) less than one.'),
    },
    {
      id: 'redshift-scalar',
      question: t('같은 관측자와 같은 광선을 다른 좌표로 쓰면 측정 주파수는 어떻게 되나요?', 'What happens to the measured frequency when the same observer and ray are written in new coordinates?'),
      options: [t('그대로입니다.', 'It stays the same.'), t('좌표를 바꿀 때마다 달라집니다.', 'It changes with every coordinate transformation.'), t('0이 됩니다.', 'It vanishes.')],
      answer: 0,
      explanation: t('측정값은 두 벡터의 계량 수축입니다.', 'The reading is a metric contraction of two vectors.'),
    },
  ],
};

export const schwarzschildRedshiftGuide: LearningGuide = {
  intuition: t('수신기는 자기 시계로 빛의 마루를 세어 주파수를 측정합니다.', 'The receiver measures frequency by counting crests against its own clock.'),
  steps: [{
    title: t('누가 어떤 주파수를 재는가', 'Who measures each frequency'),
    explanation: t('방출기 정지계의 주파수와 수신기 정지계의 주파수를 비교합니다.', 'Compare the emitter’s rest-frame frequency with the receiver’s rest-frame frequency.'),
    formula: 'ωₒ/ωₑ = √(fₑ/fₒ) · √((1+β)/(1−β))',
    diagram: t('첫 인자는 정지 시계 비교이고 둘째 인자는 방출기 운동입니다.', 'The first factor compares static clocks. The second comes from emitter motion.'),
  }],
  experiments: [
    {
      title: t('정지 시계의 적색편이', 'Redshift between static clocks'),
      predict: t('수신기를 더 멀리 두면 정지 주파수비가 어떻게 바뀔까요?', 'How does the static frequency ratio change as the receiver moves farther out?'),
      action: t('β=0으로 두고 수신 반지름을 늘리세요.', 'Set β=0 and increase the receiver radius.'),
      explain: t('분모 fₒ가 커져 전체 비율이 작아집니다.', 'The denominator fₒ increases and the ratio decreases.'),
      preset: 0,
    },
    {
      title: t('중력과 도플러 효과 상쇄', 'Cancel gravity and Doppler shifts'),
      predict: t('4M와 12M 사이에서 두 효과를 상쇄할 속도는?', 'At 4M and 12M, which speed cancels both shifts?'),
      action: t('β를 0.25로 옮겨 전체 비와 z를 읽으세요.', 'Move β to 0.25 and read the overall ratio and z.'),
      explain: t('전체 비가 1이고 z가 0입니다.', 'The overall ratio is one and z is zero.'),
      preset: 1,
    },
  ],
};
