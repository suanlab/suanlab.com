import type {LearningUnit,LearningGuide} from './interactive-book';
import {observableKinds as kinds,observableTitles as titles,observableText as t} from '@/lib/learning-observable';
import {observableSteps as steps} from '@/lib/learning-observable-concepts';
const subtitles=[t('고유공간·Born 확률·평균과 분산을 정확 계산하고 수동 기저 변환과 관측 선택을 구별합니다.','Calculate eigenspaces, Born probabilities, mean and variance while distinguishing passive coordinates from measurement choice.'),t('축퇴 공간의 coherence와 측정 도구·순서가 조건부 상태와 후속 확률에 미치는 영향을 비교합니다.','Compare how degenerate-space coherence, instruments and measurement order affect conditional states and later probabilities.')];
export const observableTeaching:{unit:LearningUnit;guide:LearningGuide}[]=kinds.map((kind,i)=>({
 unit:{id:kind,title:titles[kind],subtitle:subtitles[i],prerequisiteIds:i===0?['complex-state-rays-basis-amplitudes','eigenvalues','inner-products']:[kinds[0],'orthogonal-hilbert-projections'],prerequisites:t('복소 내적·정규화·고유값과 정규직교 사영을 복습하세요.','Review complex inner products, normalization, eigenvalues and orthogonal projections.'),eyebrow:'QUANTUM OBSERVABLES & MEASUREMENT',formula:steps[kind][0].formula,accent:'observable',questions:steps[kind].map((s,j)=>({id:`${kind}-quiz-${j}`,question:s.question,options:s.options,answer:s.answer,explanation:s.solution}))},
 guide:{intuition:subtitles[i],steps:steps[kind].map(s=>({title:s.title,explanation:s.explanation,formula:s.formula,diagram:s.action})),experiments:steps[kind].map(s=>({title:s.title,predict:s.predict,action:s.action,explain:s.solution,preset:s.presets[0]}))}
}));
