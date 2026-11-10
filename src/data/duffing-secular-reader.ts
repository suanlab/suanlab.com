import type { LearningUnit, LearningGuide } from './interactive-book';
import { duffingSecularTeachingId, type DuffingSecularTeachingId } from '@/lib/learning-duffing-secular-teaching';
import { duffingSecularSteps, duffingSecularExperiments, duffingSecularQuestions, duffingSecularLabels } from './duffing-secular-teaching';
import { duffingSecularExamples } from '@/lib/learning-duffing-secular-input';
const t = (ko: string, en: string) => ({ ko, en });
type DraftUnit = Omit<LearningUnit, 'id'> & { id: DuffingSecularTeachingId };
const diagrams=[
 t('같은 변위에서 더 큰 복원력을 주는 hardening potential을 비교합니다.','Compare hardening potentials with greater restoring forces at the same displacement.'),
 t('λ=0.01에서 A를1에서4로 바꾸면 g는0.01에서0.16으로 변합니다.','At λ=0.01 changing A from1 to4 changes g from0.01 to0.16.'),
 t('단순 전개의 보정에 포함된 t·sin(t)는 시간에 비례해 커집니다.','The t·sin(t) correction in the naive expansion grows with time.'),
 t('시간 재매개화는 주파수와 파형을 함께 조정하며 세속 성장을 제거합니다.','Time reparameterization adjusts frequency and shape together to remove secular growth.'),
 t('AGM의 산술 평균과 기하 평균을 반복하면 같은 한계로 모입니다.','Repeated arithmetic and geometric means converge to the same AGM limit.'),
 t('g=0.01의 반올림된 기준 주파수로 시간에 비례하는 위상 누적을 비교합니다.','Compare linear phase accumulation using a rounded reference frequency at g=0.01.'),
 t('같은 에너지 곡선의 다른 위상은 다른 위치를 뜻할 수 있습니다.','Different phases on the same energy curve can give different positions.'),
 t('조화 진동을2π마다 표본화하면 모두1로 보여 실제 진동을 숨깁니다.','Sampling harmonic motion every2π gives ones and hides the actual oscillation.'),
];
export const duffingSecularReader: { unit: DraftUnit; guide: LearningGuide } = {
  unit: { id: duffingSecularTeachingId,
    title: t('Duffing 진동의 세속항과 장시간 위상', 'Duffing Secular Growth and Long-Time Phase'),
    subtitle:t('단순 섭동과 Lindstedt 1·2차를 정확 기준해와 비교하며 주파수 오차·세속 성장·잔차·에너지·표본 해상도를 구분합니다.','Compare naive and first/second Lindstedt approximations with an exact reference; distinguish frequency drift, secular growth, residuals, energy and sample resolution.'),
    prerequisiteIds:['regular-perturbation-residual-certificates','differential-equations','oscillator-nondimensional-similarity'],
    prerequisites:t('정칙 섭동, 미분방정식과 진동자의 무차원화를 복습하세요.','Review regular perturbation, differential equations and oscillator nondimensionalization.'),
    eyebrow:'SECULAR GROWTH · FREQUENCY AND LONG TIME',formula:'x″+x+λx³=0; g=λA²; Δφ(H)=(ω−ν)H',accent:'duffing-secular',
    questions: duffingSecularQuestions.map((q, i) => ({ id: `duffing-secular-${i + 1}`, question: q.prompt, options: [...q.options], answer: q.correct, explanation: q.explanation })) },
  guide: {intuition:t('주파수가 조금만 달라도 오래 관측하면 같은 시각의 위치가 크게 달라집니다. 시계 보정, 감지 않은 위상, 국소 파형으로 근사의 유효 시간을 살펴보세요.','A small frequency mismatch can produce very different positions after long observation. Explore validity in time using clock corrections, unwrapped phase and local waveforms.'),
    steps: duffingSecularSteps.map((step, i) => ({ title: step.title, explanation: step.body, formula: step.formula, diagram: diagrams[i] })),
    experiments: duffingSecularExperiments.map(guide => ({ title: guide.title, predict: guide.predict,
      action: t(`${guide.examples.map(id => duffingSecularLabels[id].ko).join(' → ')} 예제를 선택하고 g·위상 누적·국소 파형·잔차·에너지 결함을 비교하세요.`, `Select ${guide.examples.map(id => duffingSecularLabels[id].en).join(' → ')}; compare g, accumulated phase, local waveforms, residuals and energy defects.`),
      explain: guide.explanation, preset: duffingSecularExamples.findIndex(example => example.id === guide.examples[0]) })) },
};
