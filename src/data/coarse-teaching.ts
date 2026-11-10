import type {LearningUnit,LearningGuide} from './interactive-book';
import {coarseKinds as kinds,coarseTitles as titles,coarseText as t} from '@/lib/learning-coarse';
import {coarseSteps as steps} from '@/lib/learning-coarse-concepts';
const subtitles=[t('같은 거시값에서 다른 미래가 생기는 이유를 정확한 집계·기억식으로 살펴봅니다.','Investigate different futures behind the same aggregate using exact lumping and memory.'),t('숨은 초기값과 기억 적분을 유지하고, 빠른 변수의 Markov 근사를 비교합니다.','Retain hidden initial data and memory integrals, then compare fast-variable Markov approximations.')];
export const coarseTeaching:{unit:LearningUnit;guide:LearningGuide}[]=kinds.map((kind,i)=>({
 unit:{id:kind,title:titles[kind],subtitle:subtitles[i],prerequisiteIds:i===0?['finite-macro-partition-conditional-entropy','finite-state-transitions']:[kinds[0],'differential-equations'],prerequisites:t('확률 법칙·행렬 전파·집계와 선형 미분방정식을 복습하세요.','Review probability laws, matrix propagation, aggregation and linear differential equations.'),eyebrow:'COARSE GRAINING & MEMORY',formula:steps[kind][0].formula,accent:'coarse',questions:steps[kind].map((s,j)=>({id:`${kind}-quiz-${j}`,question:s.question,options:s.options,answer:s.answer,explanation:s.solution}))},
 guide:{intuition:subtitles[i],steps:steps[kind].map(s=>({title:s.title,explanation:s.explanation,formula:s.formula,diagram:s.action})),experiments:steps[kind].map(s=>({title:s.title,predict:s.predict,action:s.action,explain:s.solution,preset:s.presets[0]}))},
}));
