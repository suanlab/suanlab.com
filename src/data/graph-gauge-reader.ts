import type {LearningUnit,LearningGuide} from './interactive-book';
import {graphGaugeTeachingId,type GraphGaugeTeachingId} from '@/lib/learning-graph-gauge-teaching';
import {graphGaugeSteps,graphGaugeExperiments,graphGaugeQuestions} from './graph-gauge-teaching';
import {graphGaugeExamples} from '@/lib/learning-graph-gauge-examples';
const t=(ko:string,en:string)=>({ko,en});
type DraftUnit=Omit<LearningUnit,'id'>&{id:GraphGaugeTeachingId};
const diagrams=[
 t('세 노드의 국소 성분과 j→i 비교 링크를 보여 주는 추상 그래프입니다. 실제 공간 배치가 아닙니다.','An abstract graph of three local components and j-to-i comparison links,not a physical spatial layout.'),
 t('차분의 복소 성분은 회전하지만 크기 제곱은 유지되는 기준 예제입니다.','A reference example in which complex difference components rotate while squared magnitude stays fixed.'),
 t('기본 삼각형의 원래 에너지0, 함께 변환0, 물질만 변환8을 손계산합니다.','Worked default-triangle energies: original0,coordinated0,matter-only8.'),
 t('닫힌 U(1) 곱의 중간 기준 소거와 열린 곱의 양끝 물질을 구분합니다.','Distinguish intermediate-frame cancellation in a closed U(1) product from endpoint matter in an open product.'),
 t('같은 실수부의+i와−i, 왕복 경로가 놓치는 다른 순환을 비교합니다.','Compare+i and−i with equal real parts,and a backtrack that misses another cycle.'),
 t('기본 삼각형의 정확한 H와H′, 자기 루프의 대각 항과 Wirtinger 계수를 표시합니다.','Exact default-triangle H andH′,a self-loop diagonal term and the Wirtinger factor.'),
 t('분리된 두 성분에 서로 다른 공통 위상을 주어도 링크는 유지됩니다.','Different common phases on two disconnected components still preserve their links.'),
 t('Cayley 매개변수1/2의 정확한3/5+4i/5 위상과 현재 모형의 범위입니다.','The exact Cayley phase3/5+4i/5 at parameter1/2 and the limits of the present model.'),
];
export const graphGaugeReader:{unit:DraftUnit;guide:LearningGuide}={
 unit:{id:graphGaugeTeachingId,title:t('국소 기준·U(1) 연결·Wilson 루프','Local Frames, U(1) Connections and Wilson Loops'),subtitle:t('함께 변환하는 물질과 링크, 불변 관측량과 불완전한 측정을 정확한 유한 그래프로 비교합니다.','Compare coordinated matter/link transformations,invariant observations and incomplete measurements on an exact finite graph.'),prerequisiteIds:['complex-polar-maps','wirtinger-gradients','field-noether-current-boundary-balance'],prerequisites:t('복소 위상, Wirtinger 미분과 장의 대칭 조건을 복습하세요.','Review complex phases,Wirtinger derivatives and the conditions for field symmetries.'),eyebrow:'GAUGE THEORY · LOCAL FRAMES AND OBSERVABLES',formula:'ψ′i=giψi; U′ij=giUijgj*; W′=W',accent:'graph-gauge',questions:graphGaugeQuestions.map((q,i)=>({id:`graph-gauge-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation}))},
 guide:{intuition:t('표현 기준의 변경과 물질만의 변화를 구별하고, 정확한 불변량과 충분한 관측을 따로 검토합니다.','Separate a change of representation from a matter-only change; distinguish exact invariance from sufficient observation.'),steps:graphGaugeSteps.map((step,i)=>({title:step.title,explanation:step.body,formula:step.formula,diagram:diagrams[i]})),experiments:graphGaugeExperiments.map(g=>{const items=g.examples.map(id=>graphGaugeExamples.find(p=>p.id===id)!);return{title:g.title,predict:g.predict,action:t(`${items.map(x=>x.titleKo).join(' → ')} 예제를 순서대로 고르고 세 조작·복소 루프·정확한 표와 행렬을 비교하세요.`,`Select ${items.map(x=>x.titleEn).join(' → ')} in order; compare the three operations,complex loops,exact tables and matrices.`),explain:g.explanation,preset:graphGaugeExamples.findIndex(p=>p.id===g.examples[0])};})},
};
