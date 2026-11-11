import type { LearningGuide, LearningUnit } from './interactive-book';

export const dynamicsResidualsId = 'observation-and-dynamics-residuals' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const dynamicsResidualsUnit: LearningUnit = {
  id: dynamicsResidualsId,
  title: t('관측 잔차와 방정식 잔차', 'Observation and Dynamics Residuals'),
  subtitle: t('같은 관측을 지나는 여러 후보의 미분방정식 적합도를 비교합니다.', 'Compare how well candidates passing through the same readings satisfy a differential equation.'),
  prerequisites: t('지수 감쇠, 미분과 제곱평균제곱근을 복습하세요.', 'Review exponential decay, derivatives, and root mean squares.'),
  prerequisiteIds: ['forward-and-inverse-maps-for-cooling'],
  eyebrow: 'PHYSICAL RESIDUALS',
  formula: 'r_eq = x′ + x',
  accent: 'block-order',
  questions: [
    {
      id: 'dynamics-residual-observation',
      question: t('관측 시점의 오차가 모두 0이면 방정식 잔차도 항상 0인가요?', 'Does exact agreement at the sampled times force the equation residual to be zero?'),
      options: [t('관측 사이의 움직임과 미분은 따로 확인해야 합니다.', 'The motion and derivative between readings need separate checks.'), t('항상 0입니다.', 'It is always zero.'), t('두 잔차는 정의가 같습니다.', 'The residuals have the same definition.')],
      answer: 0,
      explanation: t('유한한 점에서 함수가 같아도 그 점의 미분은 다를 수 있습니다.', 'Functions agreeing at finitely many points can have different derivatives there.'),
    },
    {
      id: 'dynamics-residual-scale',
      question: t('두 잔차를 하나의 손실에 넣기 전에 무엇을 확인해야 하나요?', 'What should be checked before combining both residuals into one loss?'),
      options: [t('각 잔차의 단위와 정규화 척도', 'The units and normalization of each residual'), t('시간 단위만', 'Only the time unit'), t('관측 개수의 홀짝', 'Whether the observation count is odd or even')],
      answer: 0,
      explanation: t('관측 잔차와 방정식 잔차는 시간 차원이 달라 합치려면 척도가 필요합니다.', 'The two residuals have different time dimensions and need explicit scales.'),
    },
  ],
};

export const dynamicsResidualsGuide: LearningGuide = {
  intuition: t('측정점을 잘 지나는 곡선도 그 사이에서 물리 법칙과 다르게 움직일 수 있습니다.', 'A curve can fit every measurement and still move differently from the physical law.'),
  steps: [{
    title: t('관측점과 방정식 비교', 'Compare measurements and dynamics'),
    explanation: t('관측 잔차는 위치를 비교하고 방정식 잔차는 기울기를 포함합니다.', 'Observation residuals compare values, while the equation residual includes the slope.'),
    formula: 'x_b(t) = exp(−t) + b sin(πt)',
    diagram: t('같은 점을 지나지만 다른 기울기를 가진 곡선을 비교합니다.', 'Compare curves sharing data points but having different slopes.'),
  }],
  experiments: [
    {
      title: t('관측을 유지한 채 후보 바꾸기', 'Change the candidate while preserving the observations'),
      predict: t('b를 바꾸면 관측 잔차와 방정식 잔차 중 어느 값이 달라질까요?', 'Which residual changes with b at an observed time?'),
      action: t('t=1에서 b를 0.2와 0으로 바꾸세요.', 'At t=1, compare b=0.2 with b=0.'),
      explain: t('관측 잔차는 0이고 방정식 잔차는 −πb입니다.', 'The observation residual is zero and the equation residual is −πb.'),
      preset: 0,
    },
    {
      title: t('관측 사이의 새 측정', 'A new reading between training times'),
      predict: t('t=0.5에 새 측정값이 있으면 후보를 구분할 수 있을까요?', 'Would a reading at t=0.5 distinguish the candidates?'),
      action: t('b=0.2에서 t를 0.5로 옮겨 기준값과의 차를 확인하세요.', 'At b=0.2, move t to 0.5 and read the gap from the reference.'),
      explain: t('후보가 기준보다 0.2 큽니다.', 'The candidate is 0.2 above the reference.'),
      preset: 1,
    },
  ],
};
