import type { LearningUnit, LearningGuide } from './interactive-book';
import { entropyKinds, entropyTitles, entropyText as t } from '@/lib/learning-entropy-production';
import { entropyConceptIntuition, entropyConceptSteps, entropyGuides } from '@/lib/learning-entropy-production-concepts';
const kind = entropyKinds[0];
export const entropyTeaching: { unit: LearningUnit; guide: LearningGuide } = {
  unit: {
    id: kind, title: entropyTitles[kind],
    subtitle: t('정상 통행, 한 경로의 경계항, 평균 생성률과 유한 시간 지수 평균을 나누어 배웁니다.', 'Distinguish stationary traffic, path boundary terms, mean production and finite-time exponential moments.'),
    prerequisiteIds: ['continuous-cycle-path-reversal', 'stationary-flux-three-state-cycle', 'two-state-continuous-time-generator-euler'],
    prerequisites: t('연속 시간 생성자·정상 통행·경로 역전을 먼저 확인하세요. 로그 확률 비와 평균의 기본 성질을 사용합니다.', 'Review continuous-time generators, stationary traffic and path reversal. We use basic properties of log probability ratios and expectations.'),
    eyebrow: 'ENTROPY PRODUCTION', formula: 'Sₜ=ln(dP/dPΘ); σ=J𝒜; Mₜ(z)=π exp(tLz)1', accent: 'entropy-production',
    questions: [1, 2, 3].map((index, i) => {
      const c = entropyConceptSteps[kind][index];
      return { id: `entropy-production-quiz-${i}`, question: c.check.question, options: c.check.options, answer: c.check.answer, explanation: c.check.explanation };
    }),
  },
  guide: {
    intuition: entropyConceptIntuition[kind],
    steps: entropyConceptSteps[kind].map(({ title, explanation, formula, diagram }) => ({ title, explanation, formula, diagram })),
    experiments: entropyGuides[kind].map(g => ({ title: g.title, predict: g.predict, action: g.action, explain: g.explain, preset: g.presetIndices[0] })),
  },
};
