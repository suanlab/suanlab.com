import type {LearningUnit,LearningGuide} from './interactive-book';
import {gaugeOrbitTeachingId,type GaugeOrbitTeachingId} from '@/lib/learning-gauge-orbits-teaching';
import {gaugeOrbitSteps,gaugeOrbitExperiments,gaugeOrbitQuestions} from './gauge-orbits-teaching';
import {gaugeOrbitExamples} from '@/lib/learning-gauge-orbits-examples';
const t=(ko:string,en:string)=>({ko,en});
type DraftUnit=Omit<LearningUnit,'id'>&{id:GaugeOrbitTeachingId};
const diagrams=[
 t('같은 쌍의 전체/앵커 군 판정을 구분합니다. 앵커0,1을 고정하면 연결 동치도 깨집니다.','Compare full/anchored-group decisions for one pair. Fixing anchors0,1 obstructs even connection equivalence.'),
 t('기본 삼각형의 오른쪽 고정 위상은1,−i,−1이며 Q와z는 모두1이 됩니다.','The default right fixing phases1,−i,−1 make every Q and z equal1.'),
 t('신장 트리의 두 간선을 고정하면 남은 간선 하나가 복소 순환 좌표입니다.','After fixing two spanning-tree edges,one remaining edge supplies the complex cycle coordinate.'),
 t('같은 링크와 밀도도 상대 물질 위상을 분류하지 못합니다. Gram01은1과i로 다릅니다.','Equal links and densities do not classify relative matter phase: Gram01 differs as1 versus i.'),
 t('외부 기준이 정한 앵커와 계산 루트는 서로 다른 조건입니다. 영 물질값은 앵커를 뜻하지 않습니다.','An externally fixed anchor differs from a computational root. Zero matter does not establish an anchor.'),
 t('앵커1에서 h1=−i이지만 전체 h를i배하면 앵커를 유지하며 트리를 고정할 수 있습니다.','Although h1=−i at anchor1,multiplying all h by i fixes the tree while preserving that anchor.'),
 t('앵커0,1을 함께 고정하면 간선0을 숲에서 제외하고 두 열린 곱을 남깁니다.','Fixing anchors0,1 removes edge0 from the forest and leaves two open products.'),
 t('영 물질과 비영 물질의 안정자 차원을 구분합니다. 이 수는 물리적 자유도 수가 아닙니다.','Compare stabilizer dimensions for zero/nonzero matter. These are not physical degrees of freedom.'),
 t('연결 변환의 존재가 물질 변환의 존재를 뜻하지 않는 앵커 예제입니다.','An anchored example where a connection witness does not imply a joint matter witness.'),
];
export const gaugeOrbitReader:{unit:DraftUnit;guide:LearningGuide}={
 unit:{id:gaugeOrbitTeachingId,title:t('게이지 궤도·트리 고정·경계 앵커','Gauge Orbits, Tree Fixing and Boundary Anchors'),subtitle:t('두 상태의 동치, 잔여 위상과 외부 기준을 정확한 정상형·경로·변환으로 비교합니다.','Compare equivalence of two states,residual phases and external references through exact normal forms,paths and transformations.'),prerequisiteIds:['graph-u1-local-frames-wilson-loops','complex-polar-maps'],prerequisites:t('국소 U(1) 연결·Wilson 루프와 복소 위상을 복습하세요.','Review local U(1) connections,Wilson loops and complex phases.'),eyebrow:'GAUGE THEORY · ORBITS, TREES AND ANCHORS',formula:'Qij=hiUijhj*; zi=hiψi; GA={g: ga=1}',accent:'gauge-orbits',questions:gaugeOrbitQuestions.map((q,i)=>({id:`gauge-orbits-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation}))},
 guide:{intuition:t('같은 불변량을 관측하는 것과 실제 허용 변환이 존재하는 것을 연결하고, 외부 기준이 바꾸는 관측을 확인합니다.','Connect invariant observations to existence of an allowed transformation,and inspect how external references change observability.'),steps:gaugeOrbitSteps.map((s,i)=>({title:s.title,explanation:s.body,formula:s.formula,diagram:diagrams[i]})),experiments:gaugeOrbitExperiments.map(g=>{const items=g.examples.map(id=>gaugeOrbitExamples.find(p=>p.id===id)!);return{title:g.title,predict:g.predict,action:t(`${items.map(x=>x.label.ko).join(' → ')} 예제를 순서대로 선택하고 네 동치 판정·숲·경로·변환 잔차를 비교하세요.`,`Select ${items.map(x=>x.label.en).join(' → ')} in order; compare all four equivalence decisions,the forest,paths and witness residuals.`),explain:g.explanation,preset:gaugeOrbitExamples.findIndex(p=>p.id===g.examples[0])};})},
};
