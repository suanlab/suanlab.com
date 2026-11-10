import type { LearningUnit, LearningGuide } from './interactive-book';
import { boundaryLayerTeachingId, type BoundaryLayerTeachingId } from '@/lib/learning-boundary-layer-teaching';
import { boundaryLayerSteps, boundaryLayerExperiments, boundaryLayerQuestions, boundaryLayerLabels } from './boundary-layer-teaching';
import { boundaryLayerExamples } from '@/lib/learning-boundary-layer-input';
const t = (ko: string, en: string) => ({ ko, en });
type DraftUnit = Omit<LearningUnit, 'id'> & { id: BoundaryLayerTeachingId };
const diagrams=[
 t('이차에서 일차로 줄어든 식은 두 경계를 일반적으로 함께 만족시키지 못합니다.','A reduction from second to first order generally loses one boundary condition.'),
 t('같은 폭에서 흐름 부호를 반전하면 경계층이 반대쪽으로 이동합니다.','At the same width, reversing drift moves the layer to the other boundary.'),
 t('바깥해와 안쪽해를 더하고 공통값을 한 번 뺍니다.','Add outer and inner solutions and subtract the common value once.'),
 t('바깥해의 최대 오차와 합성해의 증명된 전 구간 오차를 구분합니다.','Distinguish outer maximum error from the composite whole-domain certificate.'),
 t('기울기와 곡률의 큰 항이 작은 확산 계수와 균형을 이룹니다.','Large slopes and curvature balance the small diffusion coefficient.'),
 t('내부 방정식 잔차0은 경계 결함을 제거하지 않습니다.','Zero interior equation residual does not remove boundary defects.'),
 t('진폭0의 직선과 흐름0의 정확한 이차식을 구분합니다.','Distinguish a zero-amplitude line from the exact zero-drift quadratic.'),
 t('더 촘촘한 격자도 표시 목표에 못 미칠 수 있으며 수렴 증명은 별도입니다.','A finer mesh can still miss the display target; convergence proof remains separate.'),
];
export const boundaryLayerReader: { unit: DraftUnit; guide: LearningGuide } = {
  unit: { id: boundaryLayerTeachingId,
    title: t('특이 섭동과 경계층의 정합', 'Singular Perturbation and Boundary-Layer Matching'),
    subtitle: t('안쪽·바깥·합성해와 정확해를 비교하고 점별·균일 오차, 미분·경계 잔차와 격자 해상도를 구분합니다.','Compare inner, outer, composite and exact solutions; distinguish pointwise and uniform errors, derivatives, boundary residuals and mesh resolution.'),
    prerequisiteIds: ['regular-perturbation-residual-certificates', 'differential-equations', 'pointwise-uniform-convergence'],
    prerequisites: t('정칙 섭동, 미분방정식과 점별·균일 수렴을 복습하세요.','Review regular perturbation, differential equations and pointwise/uniform convergence.'),
    eyebrow: 'SINGULAR PERTURBATION · MATCHING AND BOUNDARIES', formula: 'εu″+au′=f; δ=ε/|a|; ||u−u_comp||∞=|D|e^(−1/δ)', accent: 'boundary-layer',
    questions: boundaryLayerQuestions.map((q, i) => ({ id: `boundary-layer-${i + 1}`, question: q.prompt, options: [...q.options], answer: q.correct, explanation: q.explanation })) },
  guide: { intuition: t('좁은 공간에서 큰 미분이 작은 계수와 균형을 이루는 이유를 살펴보고, 정합과 전 구간 보증으로 근사를 평가합니다.','Explore how large derivatives in a narrow region balance a small coefficient, then evaluate matching with whole-domain certificates.'),
    steps: boundaryLayerSteps.map((step, i) => ({ title: step.title, explanation: step.body, formula: step.formula, diagram: diagrams[i] })),
    experiments: boundaryLayerExperiments.map(guide => ({ title: guide.title, predict: guide.predict,
      action: t(`${guide.examples.map(id => boundaryLayerLabels[id].ko).join(' → ')} 예제를 선택하고 δ·정확해·합성해·경계 결함과 그림을 비교하세요.`, `Select ${guide.examples.map(id => boundaryLayerLabels[id].en).join(' → ')}; compare δ, exact/composite solutions, boundary defects and plots.`),
      explain: guide.explanation, preset: boundaryLayerExamples.findIndex(example => example.id === guide.examples[0]) })) },
};
