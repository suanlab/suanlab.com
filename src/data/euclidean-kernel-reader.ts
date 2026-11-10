import type { LearningUnit, LearningGuide } from './interactive-book';
import { euclideanKernelTeachingId, type EuclideanKernelTeachingId } from '@/lib/learning-euclidean-kernel-teaching';
import { euclideanKernelSteps, euclideanKernelExperiments, euclideanKernelQuestions, euclideanKernelLabels } from './euclidean-kernel-teaching';
import { euclideanKernelExamples } from '@/lib/learning-euclidean-kernel-input';
const t = (ko: string, en: string) => ({ ko, en });
export const euclideanKernelReader: { unit: Omit<LearningUnit, 'id'> & { id: EuclideanKernelTeachingId }; guide: LearningGuide } = {
  unit: {
    id: euclideanKernelTeachingId,
    title: t('Euclidean 양자 커널과 시간 분할', 'Euclidean Quantum Kernels and Time Slicing'),
    subtitle: t('양의 경로 가중치에서 커널·고정끝 법칙·열적 trace를 구별하고, 유한 Gaussian 적분과 연속 Mehler 기준해를 비교합니다.', 'Distinguish positive path weights, endpoint kernels, pinned laws and thermal traces; compare finite Gaussian integrals with the continuous Mehler reference.'),
    prerequisiteIds: ['brownian-bridge-time-slicing-measure', 'finite-gaussian-field-sources-connected-correlations', 'finite-spectrum-partition-cumulant-generators'],
    prerequisites: t('고정끝 Gaussian 정규화, 정밀도 행렬과 분배함수를 복습하세요.', 'Review pinned Gaussian normalization, precision matrices and partition functions.'),
    eyebrow: 'EUCLIDEAN KERNELS · GAUSSIAN INTEGRALS · THERMAL TRACE', formula: 'K_N=√(P/π)exp[−g(a²+b²)+hab]; Q=s(L+vI)', accent: 'euclidean-kernel',
    questions: euclideanKernelQuestions.map((q, i) => ({ id: `euclidean-kernel-${i + 1}`, question: q.prompt, options: [...q.options], answer: q.correct, explanation: q.explanation })),
  },
  guide: {
    intuition: t('위치 비용을 더한 Gaussian 경로 적분은 확률 정규화를 자동으로 보존하지 않습니다. 한 변수 손계산부터 커널, 조건부 법칙과 trace를 각각 만들어 보세요.', 'Adding a position cost to a Gaussian path integral does not automatically preserve probability normalization. Start with a one-variable calculation and construct the kernel, conditional law and trace separately.'),
    steps: euclideanKernelSteps.map(s => ({ title: s.title, explanation: s.body, formula: s.formula, diagram: s.diagram })),
    experiments: euclideanKernelExperiments.map(g => ({ title: g.title, predict: g.predict, action: t(`${g.examples.map(id => euclideanKernelLabels[id].ko).join(' → ')} 예제를 선택하고 평균·공분산·커널 계수·열적 trace의 정확한 값과 정련의 근사 로그 오차를 비교하세요.`, `Select ${g.examples.map(id => euclideanKernelLabels[id].en).join(' → ')}; compare exact means, covariance, kernel coefficients and thermal traces, alongside approximate refinement log errors.`), explain: g.explanation, preset: euclideanKernelExamples.findIndex(c => c.id === g.examples[0]) })),
  },
};
