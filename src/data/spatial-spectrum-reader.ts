import type { LearningUnit, LearningGuide } from './interactive-book';
import { spatialSpectrumTeachingId, type SpatialSpectrumTeachingId } from '@/lib/learning-spatial-spectrum-teaching';
import { spatialSpectrumSteps, spatialSpectrumExperiments, spatialSpectrumQuestions, spatialSpectrumLabels } from './spatial-spectrum-teaching';
import { spatialSpectrumExamples } from '@/lib/learning-spatial-spectrum-input';
const t = (ko: string, en: string) => ({ ko, en });
export const spatialSpectrumReader: { unit: Omit<LearningUnit, 'id'> & { id: SpatialSpectrumTeachingId }; guide: LearningGuide } = {
  unit: {
    id: spatialSpectrumTeachingId,
    title: t('공간 스펙트럼·감쇠·상관 길이', 'Spatial Spectra, Decay and Correlation Length'),
    subtitle: t('위치 공분산과 Fourier 파워를 연결하고, 경계·제약·간격을 바꾸며 유한 상관과 무한 감쇠 길이를 구별합니다.', 'Connect position covariance with Fourier power; vary boundaries, constraints and spacing to distinguish finite correlations from infinite decay length.'),
    prerequisiteIds: ['finite-gaussian-field-sources-connected-correlations', 'field-fourier-modes-lattice-continuum', 'interacting-nongaussian-connected-correlations'],
    prerequisites: t('가우스 공분산·제약, 경계별 장 모드와 비가우스 고차 연결량을 복습하세요.', 'Review Gaussian covariance and constraints, boundary field modes and non-Gaussian higher connected quantities.'),
    eyebrow: 'SPATIAL CORRELATIONS · FOURIER POWER · DECAY', formula: 'C = T G; C̃ = FCF*; ξ = −a/log r', accent: 'spatial-spectrum',
    questions: spatialSpectrumQuestions.map((q, i) => ({ id: `spatial-spectrum-${i + 1}`, question: q.prompt, options: [...q.options], answer: q.correct, explanation: q.explanation }))
  },
  guide: {
    intuition: t('함께 흔들리는 위치와 강하게 나타나는 공간 무늬는 같은 공분산의 두 표현입니다. 유한 고리·끝 경계·전역 제약을 먼저 구별하고 무한계 감쇠 길이와 비교하세요.', 'Cofluctuating positions and prominent spatial patterns are two expressions of the same covariance. Distinguish finite rings, endpoints and global constraints before comparing with infinite-system decay length.'),
    steps: spatialSpectrumSteps.map(s => ({ title: s.title, explanation: s.body, formula: s.formula, diagram: s.diagram })),
    experiments: spatialSpectrumExperiments.map(g => ({ title: g.title, predict: g.predict, action: t(`${g.examples.map(id => spatialSpectrumLabels[id].ko).join(' → ')} 예제를 선택하고 공분산·모드 파워·경계와 감쇠 상태를 비교하세요.`, `Select ${g.examples.map(id => spatialSpectrumLabels[id].en).join(' → ')}; compare covariance, modal power, boundaries and decay status.`), explain: g.explanation, preset: spatialSpectrumExamples.findIndex(c => c.id === g.examples[0]) }))
  }
};
