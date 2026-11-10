import type { LearningGuide, LearningUnit, LearningUnitId } from './interactive-book';
import { quantumVariationalKinds as kinds, quantumVariationalTitles as titles, quantumVariationalText as t, type QuantumVariationalKind } from '@/lib/learning-quantum-variational';
import { quantumVariationalSteps as steps } from '@/lib/learning-quantum-variational-concepts';

/** Draft metadata; reader registration follows the integration checks. */
export type QuantumVariationalTeachingUnit = Omit<LearningUnit, 'id' | 'prerequisiteIds'> & {
  id: QuantumVariationalKind; prerequisiteIds: (LearningUnitId | QuantumVariationalKind)[];
};
const subtitles = [
  t('Rayleigh 에너지와 고유값 잔차를 계산하고 영분산·바닥 공간·gap 조건을 구별합니다.', 'Calculate Rayleigh energies and eigenvalue residuals; distinguish zero variance, ground spaces and gap assumptions.'),
  t('겹치는 기저의 metric을 유지하며 공간의 포함·좌표 변경·내부와 바깥 잔차를 비교합니다.', 'Retain the metric of overlapping bases; compare nested spaces, coordinate changes and inside versus outside residuals.'),
];
const questions: LearningUnit['questions'][] = [
  [
    { id: 'rayleigh-normalization', question: t('H=diag(0,2), z=(2,2i)일 때 제곱 노름·평균 에너지·분산은?', 'For H=diag(0,2), z=(2,2i), what are the squared norm, mean energy and variance?'), options: [t('8, 1, 1', '8, 1, 1'), t('1, 4, 0', '1, 4, 0'), t('8, 2, 0', '8, 2, 0')], answer: 0, explanation: t('N=4+4=8입니다. 두 준위의 확률은 각각 1/2이므로 E=1, E[H²]=2, Var=2−1=1입니다. 원시 상태를 단위 상태로 착각하지 마세요.', 'N=4+4=8. Both levels have probability 1/2, so E=1, E[H²]=2 and Var=2−1=1. The raw vector is not a unit state.') },
    { id: 'rayleigh-gap', question: t('H=diag(−3,1), z=(2,1)에서 바닥 공간 누출과 초과 에너지/gap은?', 'For H=diag(−3,1), z=(2,1), what are ground-space leakage and excess energy divided by the gap?'), options: [t('둘 다 4/5', 'Both 4/5'), t('둘 다 1/5', 'Both 1/5'), t('누출은 분산과 항상 같음', 'Leakage always equals variance')], answer: 1, explanation: t('바닥 확률은 4/5, 누출은 1/5입니다. E=−11/5, E−E₀=4/5, gap=4여서 비율도 1/5입니다. 여기서는 두 준위뿐이라 부등식이 등호가 됩니다.', 'Ground probability is 4/5 and leakage is 1/5. E=−11/5, excess energy=4/5 and gap=4, giving 1/5. With only two levels this bound is an equality.') },
    { id: 'rayleigh-zero-path', question: t('z+tη=0인 경로 위치에서 Rayleigh 에너지는?', 'What is the Rayleigh energy where z+tη=0?'), options: [t('0', 'Zero'), t('이전 값으로 유지', 'Equal to the previous value'), t('정의되지 않음', 'Undefined')], answer: 2, explanation: t('분모 노름이 0이어서 몫이 정의되지 않습니다. 그 점을 에너지 0으로 그리거나 주변 결과로 메우면 안 됩니다.', 'The norm denominator vanishes, so the quotient is undefined. Do not draw zero energy or fill the point with a neighbouring result.') },
  ],
  [
    { id: 'ritz-single-column', question: t('H=diag(0,1,3), B=(1,0,1)ᵀ일 때 S, A, Ritz 에너지는?', 'For H=diag(0,1,3), B=(1,0,1)ᵀ, what are S, A and the Ritz energy?'), options: [t('2, 3, 3/2', '2, 3, 3/2'), t('1, 3, 3', '1, 3, 3'), t('2, 3, 0', '2, 3, 0')], answer: 0, explanation: t('S=B†B=2, A=B†HB=3이므로 Ac=ESc에서 E=3/2입니다. B를 정규직교 기저처럼 다루면 잘못된 E=3이 나옵니다.', 'S=B†B=2 and A=B†HB=3, so Ac=ESc gives E=3/2. Treating B as orthonormal incorrectly gives E=3.') },
    { id: 'ritz-outside', question: t('공간 안 분산 0, 바깥 분산 8/3이면 전체 고유상태인가요?', 'If inside variance is zero and outside variance is 8/3, is the state an ambient eigenstate?'), options: [t('예, Galerkin 조건이면 충분함', 'Yes; the Galerkin condition suffices'), t('아니오, 전체 분산은 8/3', 'No; the full variance is 8/3')], answer: 1, explanation: t('직교 분해에서 전체 분산=내부+바깥=8/3입니다. 시험 공간 안의 모든 변분에 정류해도 공간 밖으로 고유값 잔차가 남을 수 있습니다.', 'Orthogonality gives full variance=inside+outside=8/3. Stationarity under all trial-space variations can leave a nonzero residual outside.') },
    { id: 'ritz-nonnested', question: t('H=diag(0,1,2)에서 V=span(e₀)를 W=span(e₁,e₂)로 바꾸면 최소 에너지는?', 'For H=diag(0,1,2), replacing V=span(e₀) with W=span(e₁,e₂) changes the minimum energy how?'), options: [t('차원이 커져 반드시 감소', 'It must decrease because dimension increases'), t('0에서 1로 증가', 'It increases from 0 to 1'), t('항상 동일', 'It is always unchanged')], answer: 1, explanation: t('V는 W에 포함되지 않습니다. 실제 최소값은 각각 0과 1입니다. 단조 감소 정리는 포함 관계가 있는 공간 열에 적용되며 차원 수 비교만으로는 부족합니다.', 'V is not contained in W. Their minima are 0 and 1. Monotonicity requires nested spaces; dimension counts alone do not suffice.') },
  ],
];
export const quantumVariationalTeaching: { unit: QuantumVariationalTeachingUnit; guide: LearningGuide }[] = kinds.map((kind, i) => ({
  unit: {
    id: kind, title: titles[kind], subtitle: subtitles[i],
    prerequisiteIds: i === 0 ? ['complex-state-rays-basis-amplitudes', 'hermitian-spectrum-born-expectation', 'functional-directional-variations'] : [kinds[0], 'hermitian-spectrum-born-expectation'],
    prerequisites: t('복소 내적·Hermitian 스펙트럼·정규화와 방향 변분을 복습하세요.', 'Review complex inner products, Hermitian spectra, normalization and directional variations.'),
    eyebrow: 'QUANTUM VARIATIONAL PRINCIPLE', formula: steps[kind][0].formula, accent: 'quantum-variational', questions: questions[i],
  },
  guide: {
    intuition: subtitles[i],
    steps: steps[kind].map(s => ({ title: s.title, explanation: s.explanation, formula: s.formula, diagram: s.action })),
    experiments: steps[kind].map(s => ({ title: s.title, predict: s.predict, action: s.action, explain: s.solution, preset: s.presets[0] })),
  },
}));
