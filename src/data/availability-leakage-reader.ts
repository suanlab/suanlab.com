import type { LearningGuide, LearningUnit } from './interactive-book';

export const availabilityLeakageId = 'availability-leakage-and-reproducible-evaluation' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const availabilityLeakageUnit: LearningUnit = {
  id: availabilityLeakageId,
  title: t('가용 시각과 평가 누출', 'Availability, Leakage and Evaluation'),
  subtitle: t('사용할 수 있었던 정보와 누출된 미래 정보의 예측 오차를 비교합니다.', 'Compare prediction errors from available and leaked future information.'),
  prerequisites: t('예측 정보 집합, 상관과 조건부 예측을 복습하세요.', 'Review prediction information sets, correlation, and conditional prediction.'),
  prerequisiteIds: ['seeds-uncertainty-and-paired-sign-tests'],
  eyebrow: 'LEAKAGE AND REPRODUCIBILITY',
  formula: 'MSE_available = 0.64(1 − ρ²) + 0.36',
  accent: 'block-order',
  questions: [
    {
      id: 'leakage-date-availability',
      question: t('값의 관측 날짜가 판단 시각보다 이르면 항상 사용할 수 있나요?', 'Is an earlier observation date enough to establish feature availability?'),
      options: [t('공개 시각과 수정 이력도 확인해야 합니다.', 'Release time and revision history also matter.'), t('항상 사용할 수 있습니다.', 'It is always available.'), t('측정값은 어떤 예측에도 사용할 수 없습니다.', 'Measurements cannot be used in prediction.')],
      answer: 0,
      explanation: t('날짜가 지난 측정값도 보고 지연 때문에 나중에 공개될 수 있습니다.', 'A measurement with an earlier date can be released later because of reporting delay.'),
    },
    {
      id: 'leakage-reproducibility',
      question: t('같은 실행으로 같은 결과가 나오면 누출이 없다고 할 수 있나요?', 'Does reproducing a result establish that it is free of leakage?'),
      options: [t('정보 집합의 타당성을 별도로 검사합니다.', 'The validity of the information set needs a separate check.'), t('항상 누출이 없습니다.', 'It always rules out leakage.'), t('시드를 바꾸면 정보 검사가 필요 없습니다.', 'Changing the seed makes information checks unnecessary.')],
      answer: 0,
      explanation: t('같은 누출을 포함한 실행도 같은 결과를 재현할 수 있습니다.', 'A pipeline with the same leakage can reproduce its optimistic result.'),
    },
  ],
};

export const availabilityLeakageGuide: LearningGuide = {
  intuition: t('점수가 좋아진 이유가 미래 정보를 미리 본 것인지 확인하려면 판단 시각의 정보 경계를 정해야 합니다.', 'A decision-time information boundary reveals whether a gain depends on seeing the future.'),
  steps: [{
    title: t('정보에 따른 세 예측', 'Three predictions under different information'),
    explanation: t('현재 특성, 미래 특성, 목표 자체를 쓴 예측의 오차를 각각 구합니다.', 'Compute errors using the present feature, the future feature, and the target itself.'),
    formula: 'Y = 0.8F + 0.6N',
    diagram: t('진짜 예측 가능성을 바꾸며 누출의 이득을 비교합니다.', 'Vary genuine predictability and compare the apparent gain from leakage.'),
  }],
  experiments: [
    {
      title: t('예측 정보가 없는 현재', 'An uninformative present'),
      predict: t('ρ=0일 때 현재 정보와 누출된 미래 정보의 MSE는?', 'At ρ=0, what are the errors with present and leaked future information?'),
      action: t('ρ를 0으로 두세요.', 'Set ρ to zero.'),
      explain: t('현재 정보의 MSE는 1이고 미래 특성을 쓰면 0.36입니다.', 'Present information gives MSE one, and the future feature gives 0.36.'),
      preset: 0,
    },
    {
      title: t('현재 정보에 신호가 있을 때', 'A genuinely predictive present'),
      predict: t('ρ가 1이면 현재 관측으로 미래 신호를 알 수 있을까요?', 'At ρ=1, does the current observation determine the future signal?'),
      action: t('ρ를 1로 올리세요.', 'Raise ρ to one.'),
      explain: t('현재 특성이 미래 신호와 같아 두 MSE가 모두 0.36입니다.', 'The present feature equals the future signal and both MSEs are 0.36.'),
      preset: 1,
    },
  ],
};
