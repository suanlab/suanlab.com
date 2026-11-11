import type { LearningGuide, LearningUnit } from './interactive-book';

export const researchRecordsId = 'negative-results-and-research-records' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const researchRecordsUnit: LearningUnit = {
  id: researchRecordsId,
  title: t('부정적 결과와 연구 기록', 'Negative Results and Research Records'),
  subtitle: t('완료한 실행과 선택한 결과를 연결해 연구 판단의 근거를 남깁니다.', 'Link completed runs and selected results to preserve the evidence behind research decisions.'),
  prerequisites: t('모형 선택, 평가 분할과 실행 기록을 복습하세요.', 'Review model selection, evaluation splits, and experiment records.'),
  prerequisiteIds: ['theorems-evidence-and-working-hypotheses'],
  eyebrow: 'RESEARCH RECORDS',
  formula: 'reported summary ↔ retained runs',
  accent: 'block-order',
  questions: [
    {
      id: 'records-crashed-run',
      question: t('실행이 중단되어 최종 점수가 없으면 어떻게 기록하나요?', 'How should a run without a valid final score be recorded?'),
      options: [t('실패 단계와 오류, 남은 결과물을 기록', 'Record the failed stage, error, and partial artifacts'), t('점수 0으로 대체', 'Replace the score with zero'), t('아무 기록 없이 제외', 'Omit it without a record')],
      answer: 0,
      explanation: t('실행 오류와 완료한 과학적 결과를 구분해야 합니다.', 'Execution failures and completed scientific outcomes have different meanings.'),
    },
    {
      id: 'records-selection',
      question: t('최고 실행의 점수를 보고할 때 필요한 정보는?', 'What accompanies a best-run score?'),
      options: [t('선택 규칙과 전체 실행 기록', 'The selection policy and retained run records'), t('가장 좋은 점수만', 'Only the best score'), t('실행 이름의 길이', 'The length of the run name')],
      answer: 0,
      explanation: t('선택 과정이 있어야 보고한 수치의 의미를 재구성할 수 있습니다.', 'The selection process is needed to reconstruct what the reported quantity means.'),
    },
  ],
};

export const researchRecordsGuide: LearningGuide = {
  intuition: t('선택한 결과를 전체 실행 기록에 연결하면 좋은 결과와 실패의 맥락을 함께 볼 수 있습니다.', 'Linking selected results to all runs preserves the context of successes and failures.'),
  steps: [{
    title: t('요약값의 출처', 'Where a summary comes from'),
    explanation: t('전체 평균과 선택 평균을 각각 이름 붙이고 선택에 사용한 집합을 남깁니다.', 'Label the overall and selected means and retain the set used for selection.'),
    formula: 'mean_selected = Σ selected losses / selected count',
    diagram: t('정렬한 전체 실행과 선택한 실행을 함께 봅니다.', 'View sorted completed runs alongside the selected subset.'),
  }],
  experiments: [
    {
      title: t('최고 한 개만 보고하기', 'Report only the best run'),
      predict: t('선택한 평균과 전체 평균은 얼마인가요?', 'What are the selected and overall means?'),
      action: t('선택 수를 1로 두세요.', 'Set the selected count to one.'),
      explain: t('최고값은 0.42이고 전체 평균은 0.534입니다.', 'The best value is 0.42 and the overall mean is 0.534.'),
      preset: 0,
    },
    {
      title: t('전체 실행 포함하기', 'Include every completed run'),
      predict: t('선택 수를 모두 포함하면 두 평균은 같아질까요?', 'Do the means agree when every run is selected?'),
      action: t('선택 수를 10으로 올리세요.', 'Raise the selected count to ten.'),
      explain: t('같은 목록을 요약하므로 두 평균이 같습니다.', 'Both means summarize the same list.'),
      preset: 1,
    },
  ],
};
