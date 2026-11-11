import type { LearningGuide, LearningUnit } from './interactive-book';

export const contributionEvidenceId = 'related-work-and-contribution-evidence' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const contributionEvidenceUnit: LearningUnit = {
  id: contributionEvidenceId,
  title: t('선행 연구와 기여의 근거', 'Related Work and Contribution Evidence'),
  subtitle: t('같은 조건의 비교에서 새 변경의 효과를 찾습니다.', 'Isolate the effect of a new change through matched comparisons.'),
  prerequisites: t('검증할 가설과 비교 조건을 복습하세요.', 'Review testable hypotheses and matched comparisons.'),
  prerequisiteIds: ['hypotheses-and-testable-research-questions'],
  eyebrow: 'CONTRIBUTION EVIDENCE',
  formula: 'I = Δ_long − Δ_short',
  accent: 'block-order',
  questions: [
    {
      id: 'contribution-matched-effect',
      question: t('긴 관측 구간에서 열수지 항의 개선량을 구하는 비교는?', 'Which comparison estimates the physical term’s gain in long context?'),
      options: [t('긴 관측 구간에서 열수지 항의 유무 비교', 'Long context with and without the term'), t('짧은 기준선과 긴 열수지 모형 비교', 'Short baseline versus long physical model'), t('열수지 항 없는 두 관측 구간 비교', 'Both contexts without the term')],
      answer: 0,
      explanation: t('관측 구간을 같게 유지하고 열수지 항만 바꿉니다.', 'Keep context length fixed and change the physical term.'),
    },
    {
      id: 'contribution-novelty',
      question: t('같은 조건에서 오차가 줄었다면 무엇을 추가로 확인해야 새 기여를 주장할 수 있나요?', 'What else is needed to claim novelty after observing a matched error reduction?'),
      options: [t('가장 가까운 선행 방법과 새로 검증한 범위', 'The closest prior methods and what was newly tested'), t('가장 약한 비교 방법의 오차만', 'Only the weakest comparison error'), t('새 방법에 붙인 이름', 'The name of the new method')],
      answer: 0,
      explanation: t('성능 비교와 선행 방법에 대한 기여 주장은 각기 근거가 필요합니다.', 'Performance and novelty claims require their own evidence.'),
    },
  ],
};

export const contributionEvidenceGuide: LearningGuide = {
  intuition: t('조건 두 가지가 함께 바뀌면 총 변화에 각각의 효과가 섞입니다.', 'When two conditions change together, the total difference mixes their effects.'),
  steps: [{
    title: t('같은 조건끼리 비교', 'Compare within the same condition'),
    explanation: t('관측 구간마다 열수지 항을 넣기 전과 후의 차이를 구합니다.', 'Compute the before-and-after difference for the physical term within each context.'),
    formula: 'Δ_context = L_base − L_physics',
    diagram: t('두 관측 구간에 걸쳐 개선량을 나란히 봅니다.', 'Compare improvement across the two context lengths.'),
  }],
  experiments: [
    {
      title: t('두 기여 구분하기', 'Separate the two changes'),
      predict: t('전체 개선과 긴 구간에서의 열수지 항 개선은 각각 얼마일까요?', 'What are the total improvement and the physical-term gain in long context?'),
      action: t('기본 예에서 두 수치를 비교하세요.', 'Compare the two readouts in the default example.'),
      explain: t('전체 개선은 0.20 K이고 긴 구간에서 열수지 항의 개선은 0.10 K입니다.', 'The combined gain is 0.20 K and the matched long-context gain is 0.10 K.'),
      preset: 0,
    },
    {
      title: t('개선량을 같게 만들기', 'Equalize the matched gains'),
      predict: t('두 구간의 개선량이 같으면 개선량의 차는 얼마일까요?', 'What is the difference of effects when both gains agree?'),
      action: t('긴 구간의 열수지 모형 오차를 0.44 K로 올리세요.', 'Raise the long-context physical model error to 0.44 K.'),
      explain: t('두 개선량이 0.06 K가 되어 개선량의 차가 0입니다.', 'Both gains are 0.06 K, making the difference of effects zero.'),
      preset: 1,
    },
  ],
};
