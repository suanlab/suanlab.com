import type { LearningUnit, LearningGuide } from './interactive-book';
import { regularPerturbationTeachingId, type RegularPerturbationTeachingId } from '@/lib/learning-regular-perturbation-teaching';
import { regularPerturbationSteps, regularPerturbationExperiments, regularPerturbationQuestions, regularPerturbationLabels } from './regular-perturbation-teaching';
import { regularPerturbationExamples } from '@/lib/learning-regular-perturbation-input';
const t = (ko: string, en: string) => ({ ko, en });
type DraftUnit = Omit<LearningUnit, 'id'> & { id: RegularPerturbationTeachingId };
const diagrams = [
  t('선형 에너지에 양의 사차 항을 추가해 평형을 비교합니다.', 'Add a positive quartic term to the linear energy and compare equilibria.'),
  t('기울기 하한과 정확한 근 구간을 구분합니다.', 'Distinguish the lower slope bound from the exact root bracket.'),
  t('작은 λ라도 큰 원천과 연한 강성에서는 g가 커집니다.', 'Small λ can still give large g with a large source or soft stiffness.'),
  t('기본 예제의 계수와 실제 보정항을 나란히 계산합니다.', 'Compare coefficients with evaluated correction terms in the default example.'),
  t('P₂=0.93의 잔차와 해 오차 상계를 구별합니다.', 'Distinguish the residual at P₂=0.93 from a root-error bound.'),
  t('같은 잔차로부터 변위와 에너지의 서로 다른 보증을 구합니다.', 'Obtain distinct displacement and energy certificates from the same residual.'),
  t('양의 결합에서 유일한 실근과 Taylor 급수의 유한 반경을 구분합니다.', 'Separate uniqueness at positive coupling from the finite Taylor radius.'),
  t('강성에 따른 조건 민감도와 연구 검증 항목을 확인합니다.', 'Inspect stiffness-dependent sensitivity and checks for research.'),
];
export const regularPerturbationReader: { unit: DraftUnit; guide: LearningGuide } = {
  unit: { id: regularPerturbationTeachingId,
    title: t('정칙 섭동과 잔차 기반 오차 보증', 'Regular Perturbation and Residual-Based Error Certificates'),
    subtitle: t('비선형 평형의 차수별 근사, 무차원 결합과 수렴 반경, 정확한 잔차와 오차 보증을 비교합니다.', 'Compare order-by-order nonlinear equilibria, dimensionless coupling, convergence radius, exact residuals and error certificates.'),
    prerequisiteIds: ['taylor-remainders', 'conditioning-backward-error'],
    prerequisites: t('Taylor 나머지와 조건 민감도·후방 오차를 복습하세요.', 'Review Taylor remainders, conditioning and backward error.'),
    eyebrow: 'PERTURBATION THEORY · APPROXIMATION AND CERTIFICATES', formula: 'kx+λx³=h; g=λh²/k³; |p−x*|≤|F(p)|/k', accent: 'regular-perturbation',
    questions: regularPerturbationQuestions.map((q, i) => ({ id: `regular-perturbation-${i + 1}`, question: q.prompt, options: [...q.options], answer: q.correct, explanation: q.explanation })) },
  guide: { intuition: t('풀 수 있는 평형에서 차례로 보정하되, 작은 계수와 작은 오차, 유일한 해와 수렴하는 급수를 구별합니다.', 'Correct a solvable equilibrium step by step while distinguishing small coefficients from small errors, and unique roots from convergent series.'),
    steps: regularPerturbationSteps.map((step, i) => ({ title: step.title, explanation: step.body, formula: step.formula, diagram: diagrams[i] })),
    experiments: regularPerturbationExperiments.map(guide => ({ title: guide.title, predict: guide.predict,
      action: t(`${guide.examples.map(id => regularPerturbationLabels[id].ko).join(' → ')} 예제를 선택하고 g·부분합·잔차·오차 보증과 그림을 비교하세요.`, `Select ${guide.examples.map(id => regularPerturbationLabels[id].en).join(' → ')}; compare g, partial sums, residuals, certificates and plots.`),
      explain: guide.explanation, preset: regularPerturbationExamples.findIndex(example => example.id === guide.examples[0]) })) },
};
