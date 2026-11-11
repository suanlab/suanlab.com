import type { LearningGuide, LearningUnit } from './interactive-book';

export const coolingInverseId = 'forward-and-inverse-maps-for-cooling' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const coolingInverseUnit: LearningUnit = {
  id: coolingInverseId,
  title: t('냉각 실험의 순방향과 역방향', 'Forward and Inverse Maps for Cooling'),
  subtitle: t('온도 예측과 매개변수 복원을 한 실험에서 비교합니다.', 'Compare temperature prediction and parameter recovery in one experiment.'),
  prerequisites: t('지수함수, 로그와 일차 미분방정식을 복습하세요.', 'Review exponentials, logarithms, and first-order differential equations.'),
  prerequisiteIds: ['physical-computing-validation-by-ablation'],
  eyebrow: 'FORWARD AND INVERSE MAPS',
  formula: 'y(t) = Ae^(−kt)',
  accent: 'block-order',
  questions: [
    {
      id: 'cooling-inverse-single',
      question: t('초기 온도 차와 냉각률을 모두 모르면 한 번의 관측으로 무엇을 알 수 있나요?', 'With both parameters unknown, what does one reading determine?'),
      options: [t('관측을 만족하는 매개변수 곡선', 'A curve of compatible parameter pairs'), t('두 매개변수의 유일한 값', 'A unique pair of parameters'), t('초기 온도 차가 항상 0이라는 사실', 'That initial excess is always zero')],
      answer: 0,
      explanation: t('냉각률마다 같은 관측을 주는 초기 초과 온도를 고를 수 있습니다.', 'Each cooling rate has a corresponding initial excess giving the same reading.'),
    },
    {
      id: 'cooling-inverse-interval',
      question: t('두 관측 시각이 매우 가까울 때 로그비로 추정한 냉각률에 어떤 문제가 생기나요?', 'What happens to the log-ratio rate estimate when observation times are very close?'),
      options: [t('작은 관측 오차도 추정량을 크게 바꿀 수 있습니다.', 'Small measurement errors can greatly change the estimate.'), t('두 관측이 항상 정확해집니다.', 'Both readings always become exact.'), t('주변 온도를 자동으로 알아냅니다.', 'It automatically determines ambient temperature.')],
      answer: 0,
      explanation: t('관측의 로그 차를 작은 시간 간격으로 나누기 때문입니다.', 'The log difference is divided by a small time interval.'),
    },
  ],
};

export const coolingInverseGuide: LearningGuide = {
  intuition: t('하나의 측정점을 지나는 여러 냉각 곡선은 새 관측에서 서로 다른 값을 예측합니다.', 'Many cooling curves pass through one measurement and predict different values at another time.'),
  steps: [{
    title: t('온도비로 냉각률 추정', 'Recover the rate from a temperature ratio'),
    explanation: t('두 초과 온도를 나누면 공통 초기값이 사라지고 시간차와 냉각률만 남습니다.', 'Taking the ratio cancels the common initial amplitude, leaving the time difference and cooling rate.'),
    formula: 'k = ln(y₁/y₂)/(t₂ − t₁)',
    diagram: t('첫 관측에서 만나는 여러 곡선이 두 번째 관측으로 좁혀집니다.', 'A second observation narrows the family of curves passing through the first.'),
  }],
  experiments: [
    {
      title: t('관측 하나와 둘', 'One reading versus two'),
      predict: t('한 관측에 맞는 두 곡선은 6분에도 같은 온도를 예측할까요?', 'Do two curves fitting one reading predict the same temperature at six minutes?'),
      action: t('한 관측의 후보 곡선을 보고 두 번째 관측을 포함하세요.', 'Inspect the one-reading candidates, then include the second reading.'),
      explain: t('두 번째 측정값이 냉각률을 결정합니다.', 'The second reading determines the cooling rate in this model.'),
      preset: 0,
    },
    {
      title: t('가까운 관측과 오차', 'Close observation times and noise'),
      predict: t('두 관측이 가까워지면 같은 측정 오차는 추정에 어떤 영향을 줄까요?', 'How does the same reading error affect the estimate when observations are closer together?'),
      action: t('두 번째 관측에 +0.5 K를 더한 뒤 시각을 6분에서 2.25분으로 줄이세요.', 'Add +0.5 K to the second reading and move it from six minutes to 2.25 minutes.'),
      explain: t('작은 시간 간격이 냉각률의 오차를 확대합니다.', 'A smaller time gap amplifies the rate error.'),
      preset: 1,
    },
  ],
};
