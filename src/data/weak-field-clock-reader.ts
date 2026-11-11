import type { LearningGuide, LearningUnit } from './interactive-book';

export const weakFieldClockId = 'weak-field-orbital-clock-approximation' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const weakFieldClockUnit: LearningUnit = {
  id: weakFieldClockId,
  title: t('약한 장 근사와 궤도 시계', 'Weak-Field Approximation and Orbital Clocks'),
  subtitle: t('정확한 원형 궤도 시계로 일차 근사의 적용 범위를 검증합니다.', 'Use an exact circular-orbit clock to test the range of a first-order approximation.'),
  prerequisites: t('고유 시간, 측지선과 중력 적색편이를 복습하세요.', 'Review proper time, geodesics, and gravitational redshift.'),
  prerequisiteIds: ['spacetime-observation-and-redshift', 'geodesics-straight-lines-and-great-circles'],
  eyebrow: 'WEAK-FIELD APPROXIMATION',
  formula: 'dτ/dt = √(1−3M/r)',
  accent: 'block-order',
  questions: [
    {
      id: 'weak-clock-sign',
      question: t('원형 궤도에서 일차 시계 근사의 오차 부호는?', 'What is the sign of the first-order clock-rate error on the circular orbit?'),
      options: [t('근사가 정확한 값보다 크다.', 'The approximation exceeds the exact value.'), t('근사가 항상 작다.', 'The approximation is always smaller.'), t('항상 같다.', 'The rates are always equal.')],
      answer: 0,
      explanation: t('두 값의 차이를 유리화하면 분자와 분모가 모두 양수입니다.', 'Rationalizing the difference gives a positive numerator and denominator.'),
    },
    {
      id: 'weak-clock-error-order',
      question: t('u=M/r이 작을 때 일차 근사의 오차는 어느 차수인가요?', 'What is the order of the first-order error for small u=M/r?'),
      options: [t('u²', 'u²'), t('u', 'u'), t('1/u', '1/u')],
      answer: 0,
      explanation: t('차이의 선도항은 9u²/8입니다.', 'The leading difference is 9u²/8.'),
    },
  ],
};

export const weakFieldClockGuide: LearningGuide = {
  intuition: t('근사는 원래 모형과의 오차가 목적에 충분히 작을 때 사용할 수 있습니다.', 'An approximation is usable when its error is small enough for the intended comparison.'),
  steps: [{
    title: t('기준과 근사', 'Exact benchmark and approximation'),
    explanation: t('같은 반지름에서 두 진행률의 차이를 계산합니다.', 'Compute the difference between the rates at the same radius.'),
    formula: '√(1−3u) ≈ 1−3u/2',
    diagram: t('허용오차 선 아래에서 일차 근사가 기준을 충족합니다.', 'The first-order approximation meets the target below the tolerance line.'),
  }],
  experiments: [
    {
      title: t('반지름과 절단 오차', 'Radius and truncation error'),
      predict: t('반지름을 네 배 늘리면 약한 장의 오차는 어느 정도 줄까요?', 'In the weak-field regime, how much does the error shrink when the radius quadruples?'),
      action: t('12M와 48M에서 오차를 읽으세요.', 'Read the errors at 12M and 48M.'),
      explain: t('오차는 대략 반지름의 제곱에 반비례합니다.', 'The error is approximately inversely proportional to the squared radius.'),
      preset: 0,
    },
    {
      title: t('허용오차로 범위 결정', 'Set the range using a tolerance'),
      predict: t('12M에서 0.01과 0.001의 두 기준을 모두 만족할까요?', 'At 12M, are both tolerances 0.01 and 0.001 satisfied?'),
      action: t('반지름을 12M로 놓고 허용오차를 바꾸세요.', 'At radius 12M, change the tolerance.'),
      explain: t('같은 근사라도 요구 정확도에 따라 사용 여부가 달라집니다.', 'The same approximation can satisfy one accuracy requirement and miss another.'),
      preset: 1,
    },
  ],
};
