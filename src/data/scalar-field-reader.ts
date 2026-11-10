import type {LearningUnit,LearningGuide} from './interactive-book';
import {scalarFieldTeachingId,type ScalarFieldTeachingId} from '@/lib/learning-scalar-field-teaching';
import {scalarFieldSteps,scalarFieldExperiments,scalarFieldQuestions} from './scalar-field-teaching';
import {scalarFieldExamples} from '@/lib/learning-scalar-field-examples';
const t=(ko:string,en:string)=>({ko,en});
type DraftUnit=Omit<LearningUnit,'id'>&{id:ScalarFieldTeachingId};
const diagrams=[
 t('기준 세 점의 변위와 초기 속도입니다.','Reference three-site displacements and initial velocities.'),
 t('고정된 공간 위치에서 시간 끝점을 공유하는 두 허용 이력의 개념도입니다. 해를 계산한 그림이 아닙니다.','Schematic admissible histories at one spatial position share time endpoints. These are not computed solutions.'),
 t('균일 변위1의 세 경계 비교입니다. 고정 경계의 바깥0은 자유도가 아닙니다.','Three boundaries for uniform displacement one; exterior fixed zeros are not degrees of freedom.'),
 t('같은 좌표 배열에서 간격과 가속도 척도를 비교합니다.','Spacing and acceleration scales for the same coordinate array.'),
 t('첫 예제의 에너지와 상쇄하는 일률입니다.','Energies and cancelling powers for the first example.'),
 t('h=1,c=2,a=−1,b=0의 고정 경계3점 기준 헤시안입니다.','Reference three-site fixed-boundary Hessian at h=1,c=2,a=−1,b=0.'),
 t('V(q)=q⁴/4, q=1,δ=1의 국소 기울기 기여만 나타낸 기준 예제입니다.','Reference local gradient contribution only: V(q)=q⁴/4 at q=1,δ=1.'),
];
export const scalarFieldReader:{unit:DraftUnit;guide:LearningGuide}={
 unit:{id:scalarFieldTeachingId,title:t('스칼라장의 작용과 경계 변분','Scalar-Field Action and Boundary Variation'),subtitle:t('작용·경계항에서 힘과 에너지까지, 같은 장을 세 가지 경계에서 비교합니다.','Connect action and boundary terms to forces and energy; compare one field under three boundary conditions.'),prerequisiteIds:['coupled-chain-continuum-wave-limit','quartic-potential-work-allowed-states'],prerequisites:t('연결 진동자의 파동 극한과 퍼텐셜·일의 관계를 복습하세요.','Review coupled-chain wave limits and the relationship between potential and work.'),eyebrow:'CLASSICAL FIELDS · ACTION AND BOUNDARIES',formula:'ℒ=φt²/2−c²φx²/2−V(φ); hq̈=−∇U',accent:'scalar-field',questions:scalarFieldQuestions.map((q,i)=>({id:`scalar-field-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation}))},
 guide:{intuition:t('움직이는 모양을 보기 전에 상태·경계·질량 행렬을 정하고 정확한 에너지 변화를 확인합니다.','Specify state, boundaries and mass matrix before interpreting motion; inspect exact energy changes.'),steps:scalarFieldSteps.map((step,i)=>({title:step.title,explanation:step.body,formula:step.formula,diagram:diagrams[i]})),experiments:scalarFieldExperiments.map(g=>{const labels=g.examples.map(id=>scalarFieldExamples.find(p=>p.id===id)!.label);return{title:g.title,predict:g.predict,action:t(`${labels.map(x=>x.ko).join(' → ')} 예제를 순서대로 고르고 정확한 표와 에너지 항을 비교하세요.`,`Select ${labels.map(x=>x.en).join(' → ')} in order; compare exact tables and energy terms.`),explain:g.explanation,preset:scalarFieldExamples.findIndex(p=>p.id===g.examples[0])};})},
};
