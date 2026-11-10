import type {LearningUnit,LearningGuide} from './interactive-book';
import {graphGaussTeachingId,type GraphGaussTeachingId} from '@/lib/learning-graph-gauss-teaching';
import {graphGaussSteps,graphGaussExperiments,graphGaussQuestions} from './graph-gauss-teaching';
import {graphGaussExamples} from '@/lib/learning-graph-gauss-examples';
const t=(ko:string,en:string)=>({ko,en});type DraftUnit=Omit<LearningUnit,'id'>&{id:GraphGaussTeachingId};
const diagrams=[
 t('링크의 원형 좌표와 실수 전기 운동량을 별개의 정준 변수로 둡니다.','Separate the circular link coordinate from its real electric momentum.'),
 t('삼각형 접속행렬과 E=(1,0,0)의 발산(1,−1,0)을 계산합니다.','Compute triangle incidence and divergence(1,−1,0) for E=(1,0,0).'),
 t('전하 조건의 해 존재와 현재 입력의 허용 여부를 세 사례로 구분합니다.','Three examples distinguish level existence from current-input admissibility.'),
 t('앵커 사슬의 내부 전하1을 경계 발산1,−2가 맞춥니다.','Boundary divergences1,−2 balance interior charge1 on an anchored chain.'),
 t('삼각형의6차원에서 독립 제약2개와 유효 게이지 방향2개를 제거합니다.','Remove two independent constraints and two effective gauge directions from the triangle’s six dimensions.'),
 t('P=(1,1,1)과 비숲 운동량 E2가 원형 좌표 Q의 공액 쌍을 이룹니다.','P=(1,1,1) and non-forest momentum E2 supply a canonical pair with circular Q.'),
 t('중성 삼각형의 잘못된 E와 같은 비숲 값을 유지한 올림값을 구분합니다.','Distinguish an invalid neutral-triangle E from its fixed-non-forest-value lift.'),
 t('숲을 바꾸면 특수해·운동량 좌표가 달라져도 같은 상태의 H=3/2는 유지됩니다.','Changing forests changes the particular field and momentum chart,but preserves H=3/2 for the same state.'),
 t('기본 삼각형의 전기 변화율−1,−1,−1과 Gauss변화율0을 검산합니다.','Check electric rates−1,−1,−1 and zero Gauss rates for the default triangle.'),
 t('에너지 불변성과 Gauss허용 여부는 서로 다른 검사임을 구분합니다.','Separate energy invariance from Gauss admissibility as distinct tests.'),
];
export const graphGaussReader:{unit:DraftUnit;guide:LearningGuide}={
 unit:{id:graphGaussTeachingId,title:t('정준 그래프·Gauss 제약·물리 위상공간','Canonical Graphs, Gauss Constraints and Physical Phase Space'),subtitle:t('고정 내부 전하와 경계 선속,정준 쌍과 정확한 순간 Hamiltonian흐름을 구분합니다.','Distinguish fixed interior charges,boundary flux,canonical pairs and exact instantaneous Hamiltonian flow.'),prerequisiteIds:['gauge-orbits-tree-fixing-boundary-anchors','poisson-observable-canonical-flow'],prerequisites:t('앵커 호환 숲·게이지 궤도와 Poisson괄호·정준 흐름을 복습하세요.','Review anchor-compatible forests,gauge orbits,Poisson brackets and canonical flow.'),eyebrow:'GAUGE THEORY · CANONICAL CONSTRAINTS AND REDUCTION',formula:'BI E=ρI; E=E⁽⁰⁾+Pᵀπ; dim physical=2(M−rank BI)',accent:'graph-gauss',questions:graphGaussQuestions.map((q,i)=>({id:`graph-gauss-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation}))},
 guide:{intuition:t('표현의 중복을 없애는 일과 전기장을 전하 제약으로 제한하는 일을 분리하고,경계 가정에 맞는 물리 정준 쌍을 확인합니다.','Separate removing representation redundancy from restricting electric fields to a charge level; inspect canonical pairs under the declared boundary assumptions.'),steps:graphGaussSteps.map((s,i)=>({title:s.title,explanation:s.body,formula:s.formula,diagram:diagrams[i]})),experiments:graphGaussExperiments.map(g=>{const items=g.examples.map(id=>graphGaussExamples.find(p=>p.id===id)!);return{title:g.title,predict:g.predict,action:t(`${items.map(x=>x.label.ko).join(' → ')} 예제를 순서대로 고르고 두 판정·경계 수지·정준 좌표·정확한 에너지와 변화율을 비교하세요.`,`Select ${items.map(x=>x.label.en).join(' → ')} in order; compare both decisions,boundary balance,canonical coordinates,exact energy and rates.`),explain:g.explanation,preset:graphGaussExamples.findIndex(p=>p.id===g.examples[0])};})},
};
