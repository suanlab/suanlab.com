import type { LearningGuide, LearningUnit } from './interactive-book';

export const sensorIdentifiabilityId = 'parameter-identifiability-and-sensor-design' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const sensorIdentifiabilityUnit: LearningUnit = {
  id: sensorIdentifiabilityId,
  title: t('매개변수 식별과 센서 설계', 'Parameter Identifiability and Sensor Design'),
  subtitle: t('서로 다른 관측 방향이 해의 유일성과 불확실성에 주는 영향을 살펴봅니다.', 'Explore how sensor directions affect uniqueness and parameter uncertainty.'),
  prerequisites: t('연립일차방정식과 관측 잡음의 분산을 복습하세요.', 'Review linear systems and observation noise variance.'),
  prerequisiteIds: ['ill-posedness-and-singular-modes'],
  eyebrow: 'PARAMETER IDENTIFIABILITY',
  formula: 'b = (y₁ − y₂)/(1 − r)',
  accent: 'block-order',
  questions: [
    {
      id: 'identifiability-redundancy',
      question: t('두 센서가 정확히 같은 조합을 재면 두 성분을 각각 구할 수 있나요?', 'Can two identical sensor combinations recover both individual amplitudes?'),
      options: [t('합을 구할 수 있고 개별 성분에는 추가 정보가 필요합니다.', 'The sum is observable and individual amplitudes need additional information.'), t('두 성분을 항상 정확히 구합니다.', 'Both amplitudes are always recovered exactly.'), t('합도 관측할 수 없습니다.', 'Even the sum is unobservable.')],
      answer: 0,
      explanation: t('같은 식을 반복하면 관측 행렬의 계수는 그대로입니다.', 'Repeating the same equation leaves the observation matrix rank unchanged.'),
    },
    {
      id: 'identifiability-practical',
      question: t('r이 1에 가까워질 때 b의 표준편차는 어떻게 되나요?', 'How does the standard deviation of b change as r approaches one?'),
      options: [t('커집니다.', 'It increases.'), t('0으로 줄어듭니다.', 'It approaches zero.'), t('항상 같습니다.', 'It is constant.')],
      answer: 0,
      explanation: t('관측 차이의 잡음을 작은 1−r로 나눕니다.', 'Noise in the difference is divided by a small 1−r.'),
    },
  ],
};

export const sensorIdentifiabilityGuide: LearningGuide = {
  intuition: t('두 측정식이 서로 다른 방향의 제약을 주면 교점에서 두 성분을 찾을 수 있습니다.', 'Two measurement equations constrain different directions and recover the amplitudes at their intersection.'),
  steps: [{
    title: t('두 관측 직선의 교점', 'Intersection of two observation lines'),
    explanation: t('r이 1에 가까우면 두 직선이 거의 평행이므로 작은 관측 오차에도 교점이 크게 이동합니다.', 'As r approaches one, nearly parallel lines can move their intersection substantially under a small reading error.'),
    formula: 'SD(b̂) = √2σ / |1 − r|',
    diagram: t('센서 계수가 둘째 관측 직선의 기울기를 결정합니다.', 'The sensor coefficient sets the slope of the second observation line.'),
  }],
  experiments: [
    {
      title: t('센서를 거의 같게 만들기', 'Make the sensors nearly redundant'),
      predict: t('r을 −1에서 0.9로 바꾸면 같은 잡음에서 b의 표준편차는 어떻게 변할까요?', 'At the same noise level, what happens to the standard deviation of b from r=−1 to r=0.9?'),
      action: t('σ를 0.1로 두고 r을 0.9까지 올리세요.', 'Keep σ at 0.1 and raise r to 0.9.'),
      explain: t('표준편차는 약 0.0707에서 약 1.414로 증가합니다.', 'The standard deviation rises from about 0.0707 to about 1.414.'),
      preset: 0,
    },
    {
      title: t('같은 방향을 두 번 관측하기', 'Measure the same direction twice'),
      predict: t('r=1에서 정확한 데이터와 서로 어긋난 데이터가 각각 몇 개의 해를 가질까요?', 'At r=1, how many solutions are there for equal and unequal readings?'),
      action: t('r을 1로 두고 둘째 관측 오차를 0과 0.1로 바꾸세요.', 'Set r to one and change the second reading error between zero and 0.1.'),
      explain: t('같은 관측은 하나의 직선을 남기고, 다른 관측은 평행한 두 직선을 만듭니다.', 'Equal readings leave one line, and unequal readings yield two parallel lines.'),
      preset: 1,
    },
  ],
};
