import type { LearningUnit, LearningGuide } from './interactive-book';
import { quarticAsymptoticTeachingId, type QuarticAsymptoticTeachingId } from '@/lib/learning-quartic-asymptotic-teaching';
import { quarticAsymptoticSteps, quarticAsymptoticExperiments, quarticAsymptoticQuestions, quarticAsymptoticLabels } from './quartic-asymptotic-teaching';
import { quarticAsymptoticExamples } from '@/lib/learning-quartic-asymptotic-input';
const t = (ko: string, en: string) => ({ ko, en });
type DraftUnit = Omit<LearningUnit, 'id'> & { id: QuarticAsymptoticTeachingId };
const diagrams = [
  t('같은 최소점과 곡률에서도 꼬리의 억제가 적분을 바꿉니다.', 'Tail suppression changes the integral despite identical minima and curvature.'),
  t('가우스 모멘트를 평균내어 첫 세 계수를 계산합니다.', 'Average Gaussian moments to calculate the first three coefficients.'),
  t('고정한 결합에서 항 비율이 결국 1을 넘습니다.', 'At fixed coupling the term ratio eventually exceeds one.'),
  t('짝수와 홀수 부분합이 적분의 양쪽을 감쌉니다.', 'Even and odd partial sums enclose the integral from opposite sides.'),
  t('g=1/10에서 첫 누락 항 상계가 최소인 유지 차수는 2입니다.', 'At g=1/10 the minimum omitted bound occurs at retained order 2.'),
  t('g=1/1000의 전역 유지 차수249는 예산32를 넘습니다.', 'At g=1/1000 the global retained order 249 exceeds budget 32.'),
  t('적분·점근 나머지·교집합의 정보 출처를 구분합니다.', 'Distinguish quadrature, asymptotic remainder and intersection sources.'),
  t('구간이 부분합을 포함해도 정확한 오차0을 증명하지는 않습니다.', 'An interval containing the partial sum does not prove exact zero error.'),
];
export const quarticAsymptoticReader: { unit: DraftUnit; guide: LearningGuide } = {
  unit: { id: quarticAsymptoticTeachingId,
    title: t('사차 적분과 발산 점근급수의 절단', 'Quartic Integrals and Truncation of Divergent Asymptotic Series'),
    subtitle: t('유한한 적분과 발산하는 급수, 부호 있는 나머지와 유용한 절단 차수를 비교합니다.', 'Compare a finite integral with its divergent series, signed remainders and useful truncation orders.'),
    prerequisiteIds: ['regular-perturbation-residual-certificates', 'expectation-variance', 'series-convergence-tests'],
    prerequisites: t('정칙 섭동, 기댓값과 급수 수렴 판정을 복습하세요.', 'Review regular perturbation, expectation and series convergence tests.'),
    eyebrow: 'ASYMPTOTIC SERIES · INTEGRALS AND REMAINDERS', formula: 'Z(g)=E[e^(−gX⁴/4)]; |Z−S_N|≤|c_(N+1)g^(N+1)|', accent: 'quartic-asymptotic',
    questions: quarticAsymptoticQuestions.map((q, i) => ({ id: `quartic-asymptotic-${i + 1}`, question: q.prompt, options: [...q.options], answer: q.correct, explanation: q.explanation })) },
  guide: { intuition: t('유한한 적분을 발산급수로 근사할 수 있는 이유와, 언제 항 추가를 멈출지 나머지 상계로 판단합니다.', 'Use remainder bounds to understand why a divergent series can approximate a finite integral and when to stop adding terms.'),
    steps: quarticAsymptoticSteps.map((step, i) => ({ title: step.title, explanation: step.body, formula: step.formula, diagram: diagrams[i] })),
    experiments: quarticAsymptoticExperiments.map(guide => ({ title: guide.title, predict: guide.predict,
      action: t(`${guide.examples.map(id => quarticAsymptoticLabels[id].ko).join(' → ')} 예제를 선택하고 g·부분합·첫 누락 항·적분 구간과 그림을 비교하세요.`, `Select ${guide.examples.map(id => quarticAsymptoticLabels[id].en).join(' → ')}; compare g, partial sums, first omitted terms, quadrature intervals and plots.`),
      explain: guide.explanation, preset: quarticAsymptoticExamples.findIndex(example => example.id === guide.examples[0]) })) },
};
