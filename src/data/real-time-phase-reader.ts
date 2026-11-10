import type {LearningUnit,LearningGuide} from './interactive-book';
import {realTimePhaseTeachingId,type RealTimePhaseTeachingId} from '@/lib/learning-real-time-phase-teaching';
import {realTimePhaseSteps,realTimePhaseGuides,realTimePhaseQuestions,realTimePhaseGuidePreparations} from './real-time-phase-teaching';
const t=(ko:string,en:string)=>({ko,en});
export const realTimePhaseReader:{unit:Omit<LearningUnit,'id'>&{id:RealTimePhaseTeachingId};guide:LearningGuide}={
 unit:{id:realTimePhaseTeachingId,title:t('실시간 경로적분·위상·고전극한','Real-Time Path Integrals, Phases and Classical Limits'),
  subtitle:t('복소 위상의 상쇄, 정상 경로와 음의 모드, 정확 초점, 정규화된 파동묶음의 두 극한과 fold/Airy 합류를 비교합니다.','Compare complex-phase cancellation, stationary paths and negative modes, exact caustics, two normalized-packet limits and fold/Airy mergers.'),
  prerequisiteIds:['euclidean-quantum-kernels-and-time-slicing','saddle-fluctuation-and-nongaussian-approximation','wavefunction-l2-projection-completeness'],
  prerequisites:t('유한 Gaussian 경로 적분, 정상 경로·Hessian·점근 전개, 복소 파동함수의 정규화와 확률밀도를 복습하세요.','Review finite Gaussian path integrals, stationary paths, Hessians and asymptotic expansions, and the normalization and probability density of complex wavefunctions.'),
  eyebrow:'COMPLEX PHASES · CAUSTICS · PACKETS · CLASSICAL LIMITS',formula:'K ∝ exp(iS*/ℏ−iπν/2); Σout=MΣinMᵀ; J=2πℏ¹ᐟ³Ai(−u/ℏ²ᐟ³)',accent:'real-time-phase',
  questions:realTimePhaseQuestions.map((q,i)=>({id:`real-time-phase-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation})),
 },
 guide:{intuition:t('위상 화살표는 길이가1이어도 방향이 달라 상쇄합니다. 정상성은 최소성을 뜻하지 않고, 초점의 delta 커널은 정규화된 상태에 작용해야 읽을 수 있습니다. 위치와 운동량의 폭을 함께 비교하고, 정상점 합류에서는 Airy 진폭으로 Gaussian 근사의 한계를 확인하세요.','Unit-length phase arrows can cancel when their directions differ. Stationarity does not imply a minimum, and a caustic delta kernel is understood through its action on a normalized state. Compare position and momentum widths together; at merging stationary points, use the Airy amplitude to inspect the limits of a Gaussian approximation.'),
  steps:realTimePhaseSteps.map(s=>({title:s.title,explanation:s.body,formula:s.formula,diagram:s.diagram})),
  // The custom lab owns these six guide presets, including its additional inputs.
  experiments:realTimePhaseGuides.map((g,i)=>({title:g.title,predict:g.predict,action:t(`${g.preparations.map(id=>realTimePhaseGuidePreparations.find(p=>p.id===id)!.label.ko).join(' → ')} 순서로 적용하고 현재 결과·원래 구간·해설을 비교하세요.`,`Apply ${g.preparations.map(id=>realTimePhaseGuidePreparations.find(p=>p.id===id)!.label.en).join(' → ')} and compare current results, original intervals and explanations.`),explain:g.explanation,preset:i})),
 },
};
