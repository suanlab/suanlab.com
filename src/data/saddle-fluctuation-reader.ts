import type {LearningUnit,LearningGuide} from './interactive-book';
import {saddleFluctuationTeachingId,type SaddleFluctuationTeachingId} from '@/lib/learning-saddle-fluctuation-teaching';
import {saddleFluctuationSteps,saddleFluctuationExperiments,saddleFluctuationQuestions,saddleFluctuationLabels} from './saddle-fluctuation-teaching';
import {saddleFluctuationExamples} from '@/lib/learning-saddle-fluctuation-input';
const t=(ko:string,en:string)=>({ko,en});
export const saddleFluctuationReader:{unit:Omit<LearningUnit,'id'>&{id:SaddleFluctuationTeachingId};guide:LearningGuide}={
 unit:{
  id:saddleFluctuationTeachingId,title:t('안장점·요동·비가우스 근사','Saddles, Fluctuations and Non-Gaussian Approximations'),
  subtitle:t('잔차로 인증한 정상 경로와 Gaussian 폭·삼차/사차 보정·조건부 적분비 경계를 비교하고, 여러 우물과 퇴화의 다른 척도를 탐구합니다.','Compare residual-certified stationary paths, Gaussian widths, cubic/quartic corrections and conditional ratio bounds; explore multiple wells and the different scale at degeneracy.'),
  prerequisiteIds:['euclidean-quantum-kernels-and-time-slicing','interacting-nongaussian-connected-correlations','quartic-integral-asymptotic-truncation'],
  prerequisites:t('유한 Gaussian 경로 적분, Hessian과 공분산, 사차 Laplace 전개와 나머지 경계의 조건을 복습하세요.','Review finite Gaussian path integrals, Hessians and covariance, quartic Laplace expansions and the conditions on remainder bounds.'),
  eyebrow:'STATIONARY PATHS · GAUSSIAN FLUCTUATIONS · NON-GAUSSIAN CORRECTIONS',formula:'H=∇²S(x*); C_G=ℏH⁻¹; K/K_G ~ 1+ℏc₁+…',accent:'saddle-fluctuation',
  questions:saddleFluctuationQuestions.map((q,i)=>({id:`saddle-fluctuation-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation})),
 },
 guide:{
  intuition:t('작용의 가장 낮은 경로뿐 아니라 그 주변의 폭과 비대칭이 적분에 기여합니다. 작은 손계산으로 모드·평균·국소 공분산을 구별하고, 첫 근사와 실제 적분비 경계의 조건을 확인하세요.','An integral depends on the width and asymmetry around the minimum-action path as well as its depth. Use a small hand calculation to distinguish a mode, mean and local covariance; then inspect the conditions separating a first approximation from actual ratio bounds.'),
  steps:saddleFluctuationSteps.map(s=>({title:s.title,explanation:s.body,formula:s.formula,diagram:s.diagram})),
  experiments:saddleFluctuationExperiments.map(g=>({title:g.title,predict:g.predict,action:t(`${g.examples.map(id=>saddleFluctuationLabels[id].ko).join(' → ')} 예제를 적용하세요. 모형에 맞게 정상 경로·국소 폭·인증된 보정/경계 또는 우물의 최소점·곡률·퇴화 척도를 비교하세요.`,`Apply ${g.examples.map(id=>saddleFluctuationLabels[id].en).join(' → ')}. According to the model, compare stationary paths, local widths and certified corrections/bounds, or well minima, curvatures and degenerate scales.`),explain:g.explanation,preset:saddleFluctuationExamples.findIndex(e=>e.id===g.examples[0])})),
 },
};
