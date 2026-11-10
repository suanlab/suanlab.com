import type {LearningUnit,LearningGuide} from './interactive-book';
import {fieldBalanceTeachingId,type FieldBalanceTeachingId} from '@/lib/learning-field-balance-teaching';
import {fieldBalanceSteps,fieldBalanceExperiments,fieldBalanceQuestions} from './field-balance-teaching';
import {fieldBalanceExamples} from '@/lib/learning-field-balance-examples';
const t=(ko:string,en:string)=>({ko,en});
type DraftUnit=Omit<LearningUnit,'id'>&{id:FieldBalanceTeachingId};
const diagrams=[
 t('주기3점·α=1/2 기준의 에너지와 변화율입니다.','Reference energies and rates for three periodic sites at α=1/2.'),
 t('같은 물리 상태에서 α만 바꾼 세 에너지 장부입니다.','Three energy bookkeeping conventions for the same physical state, changing only α.'),
 t('기본 예제의 변 방향과 부호입니다. 음수 흐름은 지정 방향의 반대입니다.','Reference edge orientations and signs; a negative current opposes its assigned orientation.'),
 t('0,1번 구간의 내부 흐름은 상쇄되고 절단된 두 변만 남습니다.','The internal exchange cancels in region0,1; only its two cut edges remain.'),
 t('고정 경계의 순간 Q̇=0과 Q̈=−1을 작용 대칭의 조건과 비교합니다.','Compare instantaneous Q̇=0 and Q̈=−1 at a fixed boundary with action-symmetry conditions.'),
 t('q=(1,2,0),v=0,a=0,b=c=h=1의 주기 사차 반례입니다.','Periodic quartic counterexample: q=(1,2,0),v=0,a=0,b=c=h=1.'),
 t('전체 합을 유지해도 국소적으로 틀린 수지를 만들 수 있다는 검사 예시입니다.','A checking example: incorrect local balances can preserve a global sum.'),
];
export const fieldBalanceReader:{unit:DraftUnit;guide:LearningGuide}={
 unit:{id:fieldBalanceTeachingId,title:t('국소 보존 흐름과 Noether 변환','Local Conservation Currents and Noether Transformations'),subtitle:t('전체 보존에서 점별 전달로: 에너지 배분·경계·소스와 대칭의 조건을 구별합니다.','From global conservation to local exchange: distinguish energy allocation,boundaries,sources and symmetry conditions.'),prerequisiteIds:['scalar-field-action-boundary-variation'],prerequisites:t('스칼라장의 작용·경계 변분에서 질량 행렬과 에너지 기울기를 복습하세요.','Review mass matrices and energy gradients in scalar-field action and boundary variation.'),eyebrow:'CLASSICAL FIELDS · LOCAL BALANCES',formula:'dEi/dt=Jin−Jout; dQi/dt=Cin−Cout+si',accent:'field-balance',questions:fieldBalanceQuestions.map((q,i)=>({id:`field-balance-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation}))},
 guide:{intuition:t('이웃 사이에 무엇이 전달되는지 정확한 장부로 확인하고, 순간 상쇄와 모든 상태의 대칭을 구별합니다.','Track exchanges between neighbors with exact bookkeeping; distinguish instantaneous cancellation from symmetry across all states.'),steps:fieldBalanceSteps.map((step,i)=>({title:step.title,explanation:step.body,formula:step.formula,diagram:diagrams[i]})),experiments:fieldBalanceExperiments.map(g=>{const labels=g.examples.map(id=>fieldBalanceExamples.find(p=>p.id===id)!.label);return{title:g.title,predict:g.predict,action:t(`${labels.map(x=>x.ko).join(' → ')} 예제를 순서대로 고르고 점별 값·변 방향·부분 구간의 경계 수지를 비교하세요.`,`Select ${labels.map(x=>x.en).join(' → ')} in order; compare site values,edge orientation and region boundary balances.`),explain:g.explanation,preset:fieldBalanceExamples.findIndex(p=>p.id===g.examples[0])};})},
};
