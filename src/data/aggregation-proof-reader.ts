import type { LearningGuide, LearningUnit } from './interactive-book';

export const aggregationProofId = 'assumptions-and-aggregation-counterexamples' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const aggregationProofUnit: LearningUnit = {
  id: aggregationProofId,
  title: t('집계 정리의 가정과 반례', 'Assumptions and Aggregation Counterexamples'),
  subtitle: t('관측 가중치에 따라 결론의 부호가 바뀌는 예와 일반 항등식을 증명합니다.', 'Prove a general identity and examine a sign reversal caused by observation weights.'),
  prerequisites: t('평균, 가중평균과 공분산을 복습하세요.', 'Review means, weighted means, and covariance.'),
  prerequisiteIds: ['related-work-and-contribution-evidence'],
  eyebrow: 'ASSUMPTIONS AND PROOFS',
  formula: 'd̄_weighted = d̄ + Cov(n,d)/n̄',
  accent: 'block-order',
  questions: [
    {
      id: 'aggregation-equal-counts',
      question: t('두 평균이 같음을 보장하는 충분조건은?', 'Which condition guarantees equality of the two means?'),
      options: [t('모든 사건의 관측 횟수가 같음', 'Equal counts for every episode'), t('모든 횟수가 양수임', 'All counts are positive'), t('사건별 평균 효과가 양수임', 'Positive mean episode effect')],
      answer: 0,
      explanation: t('횟수가 같으면 횟수와 효과의 공분산이 0입니다.', 'Equal counts give zero covariance between counts and effects.'),
    },
    {
      id: 'aggregation-counterexample',
      question: t('양의 사건 평균과 음의 시점 평균을 가진 예는 무엇을 보여 주나요?', 'What does a positive episode mean with a negative point mean establish?'),
      options: [t('양의 사건 평균이 양의 시점 평균을 보장한다는 주장에 대한 반례', 'A counterexample to the claimed sign guarantee'), t('어떤 가중평균도 계산할 수 없음', 'No weighted mean can be computed'), t('두 평균은 항상 반대 부호임', 'The signs are always opposite')],
      answer: 0,
      explanation: t('보편 주장은 조건을 만족하는 반례 하나로 부정할 수 있습니다.', 'One compatible counterexample refutes a universal claim.'),
    },
  ],
};

export const aggregationProofGuide: LearningGuide = {
  intuition: t('길고 어려운 사건에 큰 가중치를 주면 전형적인 사건의 결과와 달라질 수 있습니다.', 'Giving long, difficult episodes more weight can change the result relative to a typical episode.'),
  steps: [{
    title: t('두 평균의 차이를 설명하는 항', 'The term explaining the gap'),
    explanation: t('횟수와 효과의 공분산을 평균 횟수로 나눈 만큼 두 평균이 다릅니다.', 'The gap equals covariance between count and effect divided by mean count.'),
    formula: 'd̄_weighted − d̄ = Cov(n,d)/n̄',
    diagram: t('고정된 사건별 효과에 관측 횟수만 바꿉니다.', 'Change the observation counts while holding episode effects fixed.'),
  }],
  experiments: [
    {
      title: t('부호가 바뀌는 집계', 'Aggregation that reverses the sign'),
      predict: t('효과가 나쁜 사건의 횟수를 늘리면 어떤 평균이 변할까요?', 'Which mean changes when the worse episode gets more readings?'),
      action: t('두 번째 사건의 횟수를 1에서 9로 늘리세요.', 'Raise the second episode count from one to nine.'),
      explain: t('사건 평균은 0.15 K로 같고 시점 평균은 −0.05 K가 됩니다.', 'The episode mean stays at 0.15 K and the point mean becomes −0.05 K.'),
      preset: 0,
    },
    {
      title: t('가정이 충족될 때', 'When the sufficient condition holds'),
      predict: t('횟수가 같으면 두 평균은 같아질까요?', 'Do the two means coincide at equal counts?'),
      action: t('두 번째 사건의 횟수를 1로 두세요.', 'Set the second count to one.'),
      explain: t('공분산이 0이고 두 평균이 0.15 K입니다.', 'The covariance is zero and both means are 0.15 K.'),
      preset: 1,
    },
  ],
};
