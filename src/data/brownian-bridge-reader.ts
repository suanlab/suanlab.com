import type { LearningUnit, LearningGuide } from './interactive-book';
import { brownianBridgeTeachingId, type BrownianBridgeTeachingId } from '@/lib/learning-brownian-bridge-teaching';
import { brownianBridgeSteps, brownianBridgeExperiments, brownianBridgeQuestions, brownianBridgeLabels } from './brownian-bridge-teaching';
import { brownianBridgeExamples } from '@/lib/learning-brownian-bridge-input';
const t = (ko: string, en: string) => ({ ko, en });
export const brownianBridgeReader: { unit: Omit<LearningUnit, 'id'> & { id: BrownianBridgeTeachingId }; guide: LearningGuide } = {
  unit: {
    id: brownianBridgeTeachingId,
    title: t('Brownian bridge와 시간 분할의 측도', 'Brownian Bridges: Time Slicing and Conditional Measures'),
    subtitle: t('고정끝 Gaussian 적분의 정규화·주변분포·면적 오차와 거칠기를 정확한 계산과 시간 시각화로 연결합니다.', 'Connect normalized fixed-endpoint Gaussian integrals, consistent marginals, area error and roughness through exact calculations and temporal visualizations.'),
    prerequisiteIds: ['finite-gaussian-field-sources-connected-correlations', 'spatial-spectra-decay-and-correlation-length', 'multivariate-gaussians'],
    prerequisites: t('가우스 조건화와 공분산, 격자 간격과 이산 Green 행렬을 복습하세요.', 'Review Gaussian conditioning and covariance, grid spacing and discrete Green matrices.'),
    eyebrow: 'PATH INTEGRALS · TIME SLICING · CONDITIONING', formula: 'Cᵢⱼ=2Dδ[min(i,j)−ij/N]; p_N=∏gδ/Kτ', accent: 'brownian-bridge',
    questions: brownianBridgeQuestions.map((q, i) => ({ id: `brownian-bridge-${i + 1}`, question: q.prompt, options: [...q.options], answer: q.correct, explanation: q.explanation })),
  },
  guide: {
    intuition: t('두 끝을 고정해도 중간 경로는 불확실합니다. 유한 적분부터 시작해 같은 시각의 법칙과 격자에 의존하는 관측량을 분리해 보세요.', 'Fixing both endpoints leaves uncertainty in between. Begin with finite integrals and separate laws at fixed times from observables that depend on the grid.'),
    steps: brownianBridgeSteps.map(s => ({ title: s.title, explanation: s.body, formula: s.formula, diagram: s.diagram })),
    experiments: brownianBridgeExperiments.map(g => ({ title: g.title, predict: g.predict, action: t(`${g.examples.map(id => brownianBridgeLabels[id].ko).join(' → ')} 예제를 선택하고 평균·위치/증분 공분산·면적과 정련의 정확한 값을 비교하세요.`, `Select ${g.examples.map(id => brownianBridgeLabels[id].en).join(' → ')}; compare exact means, position/increment covariance, area and refinement.`), explain: g.explanation, preset: brownianBridgeExamples.findIndex(c => c.id === g.examples[0]) })),
  },
};
