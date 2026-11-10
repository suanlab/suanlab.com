import type { LearningGuide, LearningUnit } from './interactive-book';

export const causalRelayId = 'light-cones-and-causal-order' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const causalRelayUnit: LearningUnit = {
  id: causalRelayId,
  title: t('빛의 원뿔과 인과 순서', 'Light Cones and Causal Order'),
  subtitle: t('두 원뿔의 교집합으로 신호를 중계할 수 있는 위치를 구합니다.', 'Intersect two light cones to find feasible relay positions.'),
  prerequisites: t('로런츠 변환과 민코프스키 간격을 복습하세요.', 'Review Lorentz boosts and the Minkowski interval.'),
  prerequisiteIds: ['minkowski-metric-and-proper-time'],
  eyebrow: 'CAUSAL STRUCTURE',
  formula: 'Δct ≥ |Δx|',
  accent: 'block-order',
  questions: [
    {
      id: 'causal-both-legs',
      question: t('송신과 도착 사이에 중계할 수 있는 사건의 조건은?', 'Which condition must a relay event satisfy?'),
      options: [t('송신의 미래와 도착의 과거에 모두 속한다.', 'It lies in both the sender’s future and the deadline’s past.'), t('송신의 미래에만 속하면 된다.', 'It only needs to lie in the sender’s future.'), t('도착과 동시이면 된다.', 'It only needs to be simultaneous with the deadline.')],
      answer: 0,
      explanation: t('두 구간 모두 빛 이하의 속력으로 이어져야 합니다.', 'Both legs must be traversable at or below light speed.'),
    },
    {
      id: 'causal-order-reversal',
      question: t('관성계를 바꿨을 때 시간 순서가 뒤집힐 수 있는 변위는?', 'Which separation can reverse its coordinate order under a boost?'),
      options: [t('공간꼴', 'Spacelike'), t('미래 시간꼴', 'Future timelike'), t('미래 빛꼴', 'Future null')],
      answer: 0,
      explanation: t('공간꼴 변위에서는 서로 다른 관성계가 다른 시간 순서를 줄 수 있습니다.', 'Different inertial frames can assign different time orders to a spacelike pair.'),
    },
  ],
};

export const causalRelayGuide: LearningGuide = {
  intuition: t('중계기는 송신기에서 받은 신호를 기한 안에 도착점으로 보내야 합니다.', 'A relay has to receive the signal and still have time to reach the deadline event.'),
  steps: [{
    title: t('두 조건의 교집합', 'Intersection of both constraints'),
    explanation: t('송신의 미래 원뿔과 기한의 과거 원뿔을 같은 시각에서 자릅니다.', 'Slice the sender’s future cone and the deadline’s past cone at the same time.'),
    formula: '[-t,t] ∩ [b−(6−t), b+(6−t)]',
    diagram: t('두 범위가 겹치는 곳에 중계기를 둡니다.', 'Place the relay where both ranges overlap.'),
  }],
  experiments: [
    {
      title: t('중계 위치 범위', 'Feasible relay positions'),
      predict: t('도착 위치를 멀리 옮기면 중계 위치 범위가 어떻게 바뀔까요?', 'What happens to the feasible interval as the deadline moves farther away?'),
      action: t('중계 시각을 3으로 두고 b=2, 6, 7을 비교하세요.', 'At relay time 3, compare b=2, 6, and 7.'),
      explain: t('겹치는 구간은 길이가 양수인 구간에서 한 점, 그리고 빈집합으로 바뀝니다.', 'The overlap changes from a nonzero interval to a point and then to the empty set.'),
      preset: 0,
    },
    {
      title: t('좌표 순서 뒤집기', 'Reverse coordinate order'),
      predict: t('부스트로 공간꼴 도착 사건이 송신 사건보다 앞에 놓일까요?', 'Can a boost place a spacelike deadline before the emission?'),
      action: t('b=7에서 β를 0에서 0.9로 바꾸고 ct′를 비교하세요.', 'At b=7, change β from 0 to 0.9 and compare ct′.'),
      explain: t('순서는 바뀌고 두 사건 사이에 가능한 중계 경로는 계속 없습니다.', 'The coordinate order reverses while the relay interval stays empty.'),
      preset: 1,
    },
  ],
};
