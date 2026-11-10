import type {LearningUnit,LearningGuide} from './interactive-book';
import {thermalFieldTeachingId,type ThermalFieldTeachingId} from '@/lib/learning-thermal-field-teaching';
import {thermalFieldSteps,thermalFieldExperiments,thermalFieldQuestions,thermalFieldLabels} from './thermal-field-teaching';
const t=(ko:string,en:string)=>({ko,en});
export const thermalFieldReader:{unit:Omit<LearningUnit,'id'>&{id:ThermalFieldTeachingId};guide:LearningGuide}={
 unit:{id:thermalFieldTeachingId,title:t('유한 장의 측도와 열 자유에너지','Finite Field Measure and Thermal Free Energy'),subtitle:t('격자 측도·절대 정규화·온도 경로를 명시하고 유효 에너지, 내부 에너지와 두 엔트로피를 구별합니다.','Specify lattice measure,absolute normalization and thermal paths;distinguish effective energy,internal energy and two entropies.'),
 prerequisiteIds:['finite-gaussian-field-sources-connected-correlations','spatial-spectra-decay-and-correlation-length','canonical-binary-energy-fluctuations'],
 prerequisites:t('유한 Gaussian 정규화, 공분산과 원천, 격자와 경계, 정준 열 미분을 복습하세요.','Review finite Gaussian normalization,covariance and sources,lattices and boundaries,and canonical thermal derivatives.'),
 eyebrow:'FIELD MEASURE · NORMALIZATION · THERMAL PATHS',formula:'U=⟨E−TE_T⟩; Cᵥ=Var(E−TE_T)/T²−T⟨E_TT⟩',accent:'thermal-field',
 questions:thermalFieldQuestions.map((q,i)=>({id:`thermal-field-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation}))},
 guide:{intuition:t('넓은 골짜리는 더 많은 부피를 담습니다. 에너지의 최솟값뿐 아니라 측도와 determinant를 유지하고, 온도에 따라 변하는 계수의 미분까지 포함하세요.','A broad valley contains more volume. Retain the measure and determinant as well as the minimum,and include derivatives of coefficients along the temperature path.'),
 steps:thermalFieldSteps.map(s=>({title:s.title,explanation:s.body,formula:s.formula,diagram:s.caption})),
 experiments:thermalFieldExperiments.map((g,i)=>({title:g.title,predict:g.predict,explain:g.explanation,preset:i,action:t(`${g.examples.map(id=>thermalFieldLabels[id].ko).join(' → ')} 순서로 비교하고 적용한 값·온도 미분·정확한 관측량을 읽으세요.`,`Compare ${g.examples.map(id=>thermalFieldLabels[id].en).join(' → ')};inspect applied values,thermal derivatives and exact observables.`)}))},
};
