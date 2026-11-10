import type {LearningUnit,LearningGuide,LearningUnitId} from './interactive-book';
import {tensorKinds as kinds,tensorTitles as titles,tensorText as t,type TensorKind} from '@/lib/learning-tensor';
import {tensorSteps as steps} from '@/lib/learning-tensor-concepts';
/** Draft teaching metadata:the IDs are added to the registry only at integration. */
export type TensorTeachingUnit=Omit<LearningUnit,'id'|'prerequisiteIds'>&{id:TensorKind;prerequisiteIds:(LearningUnitId|TensorKind)[]};
const subtitles=[t('진폭 표에서 부분추적과 Schmidt 가중치를 유도하고 국소 조건부·무신호성을 실험합니다.','Derive reduced states and Schmidt weights from amplitude tables;explore local conditioning and no-signalling.'),t('같은 국소 혼합도의 서로 다른 공동 상태·PPT·negativity·선택/최적 CHSH를 비교합니다.','Compare joint states with equal local mixedness,PPT,negativity and selected versus optimized CHSH.')];
export const tensorTeaching:{unit:TensorTeachingUnit;guide:LearningGuide}[]=kinds.map((kind,i)=>({
 unit:{id:kind,title:titles[kind],subtitle:subtitles[i],prerequisiteIds:i===0?['complex-state-rays-basis-amplitudes','finite-ensemble-density-gram-tomography','degenerate-projectors-sequential-observation']:[kinds[0],'qubit-density-spectrum-entropy-dephasing','qubit-povm-instrument-backaction'],prerequisites:t('복소 내적·텐서 색인·양의 행렬·밀도 상태와 사영 측정을 복습하세요.','Review complex inner products,tensor indexing,positive matrices,density states and projective measurements.'),eyebrow:'TENSOR PRODUCT & ENTANGLEMENT',formula:steps[kind][0].formula,accent:'tensor',questions:steps[kind].map((s,j)=>({id:`${kind}-quiz-${j}`,question:s.question,options:s.options,answer:s.answer,explanation:s.solution}))},
 guide:{intuition:subtitles[i],steps:steps[kind].map(s=>({title:s.title,explanation:s.explanation,formula:s.formula,diagram:s.action})),experiments:steps[kind].map(s=>({title:s.title,predict:s.predict,action:s.action,explain:s.solution,preset:s.presets[0]}))}
}));
