import type {LearningUnit,LearningGuide} from './interactive-book';
import {fieldModesTeachingId,type FieldModesTeachingId} from '@/lib/learning-field-modes-teaching';
import {fieldModesSteps,fieldModesExperiments,fieldModesQuestions} from './field-modes-teaching';
import {fieldModesExamples} from '@/lib/learning-field-modes-examples';
const t=(ko:string,en:string)=>({ko,en});
type DraftUnit=Omit<LearningUnit,'id'>&{id:FieldModesTeachingId};
const diagrams=[
 t('길이4·활성4점의 세 배치입니다. 외부 고정0은 자유도가 아닙니다.','Three layouts with length4 and four active sites; exterior fixed zeros are not degrees of freedom.'),
 t('주기4점·h=1의 질량 정규화 실수 기저입니다. 각 행의 배율을 명시했습니다.','Mass-normalized real basis for four periodic sites at h=1; each row states its multiplier.'),
 t('고정 벽 최저 사인과 셀 중심 무유속 상수 모드의 모양 개념도입니다.','Shape schematic of the lowest fixed-wall sine and the cell-centered zero-flux constant mode.'),
 t('기본4점 상태의 좌표·속도·에너지 손계산입니다.','Worked coordinates,velocities and energies for the default four-site state.'),
 t('정지 기준장의 동차 선형 방정식 해 형태를 그린 개념도이며 계산한 궤적이 아닙니다.','Schematic solution forms for the homogeneous linear equation at a stationary reference; not computed trajectories.'),
 t('주기4점 이중 우물 예제의 정확한 유한 에너지 전개입니다.','Exact finite-energy decomposition for the periodic four-site double-well example.'),
 t('질량항 없는 주기 Nyquist의 표본과 두 속도 척도입니다. 파동 묶음을 적분한 그림이 아닙니다.','Samples and two speed scales of massless periodic Nyquist; not an integrated wave packet.'),
 t('고정 물리적 m과 매 격자의 Nyquist 재선택은 다른 실험입니다.','A fixed physical m and re-selected Nyquist on each grid are different experiments.'),
];
export const fieldModesReader:{unit:DraftUnit;guide:LearningGuide}={
 unit:{id:fieldModesTeachingId,title:t('Fourier 모드·격자 분산·연속 근사','Fourier Modes, Lattice Dispersion and Continuum Approximation'),subtitle:t('경계와 질량 내적에서 모드 분해까지, 고정 파수 세분화와 Nyquist 반례를 비교합니다.','Connect boundaries and mass inner products to modal decomposition; compare fixed-wave refinement with Nyquist counterexamples.'),prerequisiteIds:['scalar-field-action-boundary-variation','field-noether-current-boundary-balance'],prerequisites:t('스칼라장의 질량 행렬·경계 변분과 국소 에너지·대칭의 조건을 복습하세요.','Review scalar-field mass matrices,boundary variation,local energy and symmetry conditions.'),eyebrow:'CLASSICAL FIELDS · MODES AND CONTINUUM LIMITS',formula:'λh=μ²+4c²sin²(kh/2)/h²; uαᵀMuβ=δαβ',accent:'field-modes',questions:fieldModesQuestions.map((q,i)=>({id:`field-modes-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation}))},
 guide:{intuition:t('같은 상태를 올바른 경계·질량 기저로 읽고, 순간 곡률과 물리 동역학·격자 주파수와 고정 파수 극한을 구별합니다.','Read one state in the correct boundary and mass basis; separate frozen curvature from dynamics and grid frequencies from a fixed-wave limit.'),steps:fieldModesSteps.map((step,i)=>({title:step.title,explanation:step.body,formula:step.formula,diagram:diagrams[i]})),experiments:fieldModesExperiments.map(g=>{const labels=g.examples.map(id=>fieldModesExamples.find(p=>p.id===id)!.label);return{title:g.title,predict:g.predict,action:t(`${labels.map(x=>x.ko).join(' → ')} 예제를 순서대로 고르고 기저·계수·정확한 에너지 전개·세분화 표를 비교하세요.`,`Select ${labels.map(x=>x.en).join(' → ')} in order; compare basis shapes,coefficients,exact energy expansions and refinement tables.`),explain:g.explanation,preset:fieldModesExamples.findIndex(p=>p.id===g.examples[0])};})},
};
