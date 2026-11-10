import type { LearningGuide, LearningUnit } from './interactive-book';

export const minkowskiPathId = 'minkowski-metric-and-proper-time' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const minkowskiPathUnit: LearningUnit = {
  id: minkowskiPathId,
  title: t('민코프스키 계량과 고유 시간', 'The Minkowski Metric and Proper Time'),
  subtitle: t('같은 출발·재회 사건을 잇는 경로의 시계 읽기를 비교합니다.', 'Compare clock readings on paths that share departure and reunion events.'),
  prerequisites: t('로런츠 변환과 제곱 간격을 복습하세요.', 'Review Lorentz boosts and the squared interval.'),
  prerequisiteIds: ['special-relativity-lorentz-boosts'],
  eyebrow: 'SPACETIME GEOMETRY',
  formula: 'c²dτ² = c²dt² − dx² − dy² − dz²',
  accent: 'block-order',
  questions: [
    {
      id: 'minkowski-nonzero-null',
      question: t('0이 아닌 변위의 제곱 간격이 0이면 어떤 변위인가요?', 'What is a nonzero displacement with zero squared interval?'),
      options: [t('빛꼴', 'Null'), t('시간꼴', 'Timelike'), t('공간꼴', 'Spacelike')],
      answer: 0,
      explanation: t('빛꼴에서는 시간 성분과 공간 성분의 제곱이 상쇄됩니다.', 'The time and space squares cancel for a null displacement.'),
    },
    {
      id: 'minkowski-proper-time-maximum',
      question: t('평탄한 시공간에서 같은 시간꼴 끝점을 잇는 경로 중 고유 시간이 가장 긴 것은?', 'In flat spacetime, which path maximizes proper time between fixed timelike endpoints?'),
      options: [t('관성 직선 경로', 'The straight inertial path'), t('중간에 가장 멀리 갔다 돌아오는 경로', 'The path with the largest detour'), t('모든 경로가 같다.', 'All paths give the same time.')],
      answer: 0,
      explanation: t('끝점의 정지계에서 속력이 0일 때 고유 시간 적분의 인자가 최대입니다.', 'The proper-time integrand is largest at zero speed in the endpoint rest frame.'),
    },
  ],
};

export const minkowskiPathGuide: LearningGuide = {
  intuition: t('같은 사건에서 출발해 다시 만나더라도 시계가 거친 경로에 따라 경과 시간이 달라집니다.', 'Clocks that depart and reunite at the same events can accumulate different times along their paths.'),
  steps: [{
    title: t('간격의 부호', 'The sign of the interval'),
    explanation: t('시간차와 공간차의 제곱을 빼서 경로 구간을 분류합니다.', 'Subtract the squared spatial displacement from the squared time displacement to classify each leg.'),
    formula: 's² = (Δct)² − (Δx)²',
    diagram: t('빛꼴 구간은 제곱 간격이 0입니다.', 'A null leg has zero squared interval.'),
  }],
  experiments: [
    {
      title: t('같은 끝점과 다른 고유 시간', 'Same endpoints and different proper times'),
      predict: t('방향 전환 위치를 더 멀리 두면 고유 시간은 어떻게 바뀔까요?', 'How does proper time change when the turnaround is farther away?'),
      action: t('전환 시각을 3으로 두고 위치를 0과 2에서 비교하세요.', 'At turnaround time 3, compare positions 0 and 2.'),
      explain: t('두 직선 구간에서 속력이 커지면 시계의 경과 시간이 줄어듭니다.', 'Larger speed on both straight legs reduces elapsed proper time.'),
      preset: 0,
    },
    {
      title: t('빛의 원뿔 경계', 'The light-cone boundary'),
      predict: t('전환 위치가 허용 영역 밖으로 나가면 어느 구간이 먼저 공간꼴이 될까요?', 'Which leg becomes spacelike first when the turnaround leaves the feasible region?'),
      action: t('전환 시각 2와 위치 3을 설정해 두 구간을 확인하세요.', 'Set turnaround time 2 and position 3, then inspect both legs.'),
      explain: t('첫 구간의 공간차가 시간차보다 커서 공간꼴입니다. 전체 경로에는 시계의 고유 시간을 부여할 수 없습니다.', 'The first leg is spacelike because its spatial displacement exceeds its time displacement. The full route cannot be assigned clock time.'),
      preset: 1,
    },
  ],
};
