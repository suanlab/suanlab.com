import type { LearningGuide, LearningUnit } from './interactive-book';

export const baselineBudgetsId = 'baselines-and-matched-compute-budgets' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const baselineBudgetsUnit: LearningUnit = {
  id: baselineBudgetsId,
  title: t('기준선과 공정한 예산 비교', 'Baselines and Matched Compute Budgets'),
  subtitle: t('예산에 따라 순위가 바뀌는 학습 곡선으로 비교 조건을 살펴봅니다.', 'Examine comparison conditions with learning curves whose ranking changes with budget.'),
  prerequisites: t('기준선, 자원 경계와 학습 곡선을 복습하세요.', 'Review baselines, resource boundaries, and learning curves.'),
  prerequisiteIds: ['assumptions-and-aggregation-counterexamples'],
  eyebrow: 'FAIR COMPARISONS',
  formula: 'L_A(b) = 0.3 + 0.4/√b',
  accent: 'block-order',
  questions: [
    {
      id: 'baseline-budget-match',
      question: t('연산 예산이 다른 두 점수를 비교하면 무엇을 확인하기 어렵나요?', 'What is hard to infer from scores measured at different compute budgets?'),
      options: [t('같은 예산에서 어느 모형의 오차가 작은지', 'Which model has lower error at the same budget'), t('보고한 숫자의 크기', 'The numerical order of the reported values'), t('지표의 이름', 'The name of the metric')],
      answer: 0,
      explanation: t('모형 효과와 예산 효과를 함께 바꾸면 같은 자원에서의 비교를 알 수 없습니다.', 'Changing both model and budget does not isolate performance at equal resources.'),
    },
    {
      id: 'baseline-selection-split',
      question: t('초매개변수를 고른 뒤 최종 비교는 어떤 데이터에서 하나요?', 'After choosing hyperparameters, where should final comparison be performed?'),
      options: [t('선택에 사용하지 않은 최종 평가 데이터', 'Final evaluation data unused for selection'), t('가장 좋은 결과가 나온 개발 부분집합', 'The most favorable development subset'), t('각 모형이 따로 고른 평가 집합', 'Different evaluation sets chosen by each model')],
      answer: 0,
      explanation: t('같은 최종 평가로 선택 이후 성능을 측정합니다.', 'The same held-out final evaluation measures performance after selection.'),
    },
  ],
};

export const baselineBudgetsGuide: LearningGuide = {
  intuition: t('한 모형은 작은 예산에서 잘 맞고 다른 모형은 큰 예산에서 더 좋아질 수 있습니다.', 'One model can do better at a small budget while another improves more with a large budget.'),
  steps: [{
    title: t('두 곡선의 교차점', 'Where the curves cross'),
    explanation: t('두 손실을 같게 두어 동일 성능의 예산을 구합니다.', 'Set the two losses equal to find the budget of equal performance.'),
    formula: '0.3 + 0.4/√b = 0.2 + 0.8/√b',
    diagram: t('같은 예산에 해당하는 두 점의 높이를 비교합니다.', 'Compare both heights at the same budget.'),
  }],
  experiments: [
    {
      title: t('작은 예산과 큰 예산', 'Small and large budgets'),
      predict: t('예산 4와 64에서 같은 모형이 더 좋을까요?', 'Does the same model win at budgets four and 64?'),
      action: t('예산을 4에서 64로 바꾸세요.', 'Move the budget from four to 64.'),
      explain: t('작은 예산에서는 A, 큰 예산에서는 B의 손실이 작습니다.', 'A has lower loss at the small budget and B at the large budget.'),
      preset: 0,
    },
    {
      title: t('성능이 같은 예산', 'A budget of equal performance'),
      predict: t('두 합성 곡선이 만나는 예산은 얼마일까요?', 'At which budget do the synthetic curves meet?'),
      action: t('예산을 16으로 맞추세요.', 'Set the budget to 16.'),
      explain: t('두 손실이 모두 0.4입니다.', 'Both losses equal 0.4.'),
      preset: 1,
    },
  ],
};
