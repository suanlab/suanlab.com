import type { LearningGuide, LearningUnit } from './interactive-book';

export const researchHypothesesId = 'hypotheses-and-testable-research-questions' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const researchHypothesesUnit: LearningUnit = {
  id: researchHypothesesId,
  title: t('검증할 수 있는 연구 가설', 'Testable Research Hypotheses'),
  subtitle: t('관측할 수 있는 실패에서 출발해 대상 집단과 비교 기준을 정합니다.', 'Start from an observable failure and specify a population and a comparison.'),
  prerequisites: t('쌍별 오차와 표준오차를 복습하세요.', 'Review paired errors and standard errors.'),
  prerequisiteIds: ['regime-dependent-model-discrepancy'],
  eyebrow: 'RESEARCH QUESTIONS',
  formula: 'Δ = E[L_B − L_C]',
  accent: 'block-order',
  questions: [
    {
      id: 'research-hypotheses-threshold',
      question: t('실질적 개선 기준을 언제 정하는 것이 좋을까요?', 'When should the practical improvement threshold be chosen?'),
      options: [t('최종 평가 결과를 보기 전에', 'Before seeing final evaluation outcomes'), t('최종 평가 결과를 본 직후', 'Immediately after seeing final outcomes'), t('가장 잘 맞는 하위 집단을 고른 후', 'After choosing the most favorable subgroup')],
      answer: 0,
      explanation: t('먼저 정한 기준을 사용하면 결과에 맞춰 의미를 바꾸는 일을 줄일 수 있습니다.', 'A prespecified threshold reduces the chance of redefining success around the observed data.'),
    },
    {
      id: 'research-hypotheses-episode',
      question: t('재시작 직후 연속 온도 표본에 상관이 있으면 무엇을 비교 단위로 삼을까요?', 'What comparison unit respects correlation between consecutive readings after a restart?'),
      options: [t('재시작 사건이나 더 큰 군집', 'The restart episode or an appropriate larger cluster'), t('각 시간 표본을 독립적으로', 'Treat every time sample as independent'), t('모형별 전체 표본 수만', 'Only the total sample count for each model')],
      answer: 0,
      explanation: t('같은 사건의 오차를 먼저 요약해 상관된 시간 표본을 독립 반복으로 세지 않습니다.', 'Summarizing each episode avoids counting correlated time samples as independent replicates.'),
    },
  ],
};

export const researchHypothesesGuide: LearningGuide = {
  intuition: t('좋아졌다는 주장에는 어떤 조건에서 어떤 오차가 얼마나 줄어야 하는지 들어갑니다.', 'An improvement claim states the conditions, the metric, and a practically meaningful effect.'),
  steps: [{
    title: t('사건별 쌍별 차이', 'Paired differences by episode'),
    explanation: t('같은 사건에서 두 모델의 오차를 비교해 사건 난이도의 영향을 줄입니다.', 'Comparing models on the same episode accounts for shared episode difficulty.'),
    formula: 'dᵢ = L_Bᵢ − L_Cᵢ',
    diagram: t('사건별 차이와 실용 기준을 함께 표시합니다.', 'Plot paired effects alongside the practical threshold.'),
  }],
  experiments: [
    {
      title: t('같은 평균, 다른 변동', 'The same mean with different spread'),
      predict: t('평균 개선량이 같아도 표준오차는 달라질까요?', 'Can equal mean improvement come with different standard errors?'),
      action: t('두 합성 평가를 번갈아 선택하세요.', 'Switch between the two synthetic evaluations.'),
      explain: t('두 평균은 0.105 K로 같고 사건별 차이의 변동이 다릅니다.', 'Both means are 0.105 K, and the episode-level spreads differ.'),
      preset: 0,
    },
    {
      title: t('실용 기준과 점추정', 'Practical threshold and point estimate'),
      predict: t('기준을 높이면 어떤 수치가 바뀔까요?', 'What changes if the practical threshold increases?'),
      action: t('기준을 0.10 K에서 0.12 K로 올리세요.', 'Raise the practical threshold from 0.10 K to 0.12 K.'),
      explain: t('데이터와 추정값은 같고 점추정이 기준을 넘는지에 대한 판단만 바뀝니다.', 'Data and estimates stay fixed, while the descriptive threshold comparison changes.'),
      preset: 1,
    },
  ],
};
