import type{LearningUnit,LearningGuide}from'./interactive-book';
import{nongaussianFieldTeachingId,type NongaussianFieldTeachingId}from'@/lib/learning-nongaussian-field-teaching';
import{nongaussianFieldSteps,nongaussianFieldExperiments,nongaussianFieldQuestions,nongaussianFieldLabels}from'./nongaussian-field-teaching';
import{nongaussianFieldExamples}from'@/lib/learning-nongaussian-field-input';
const t=(ko:string,en:string)=>({ko,en});
const diagrams=[
 t('네 독립 스핀의16배치는 각각 확률1/16입니다.','The16 configurations of four independent spins each have probability1/16.'),
 t('q=2,T=1은 J=log2입니다. 원천 인자1은 물리 원천0입니다.','q=2,T=1 means J=log2. Source factor1 means zero physical source.'),
 t('q=pᵢ=1에서 r=1,2,1/2는 같은 μ,C와 다른 κ₀₁₂₃를 줍니다.','At q=pᵢ=1, r=1,2,1/2 give the same μ,C but different κ₀₁₂₃.'),
 t('독립 평균0스핀을 네 번 삽입한 사차량입니다.','Fourth-order quantities for an unbiased independent spin inserted four times.'),
 t('q=r=1,p₀=2인 반복 스핀의 정확한 기준값입니다.','Exact reference values for the repeated spin with q=r=1,p₀=2.'),
 t('다섯 위치 고리에서 적어도 한 반강자성 간선은 만족할 수 없습니다.','At least one antiferromagnetic edge is unsatisfied on a five-site ring.'),
 t('고정 인자 재척도와 같은 물리 에너지의 가열을 구분합니다.','Distinguish rescaling at fixed factors from heating the same physical energy.'),
 t('q=2,r=pᵢ=T=1인 네 위치 고리의 공분산입니다.','Covariance of the four-site ring with q=2,r=pᵢ=T=1.'),
];
export const nongaussianFieldReader:{unit:Omit<LearningUnit,'id'>&{id:NongaussianFieldTeachingId};guide:LearningGuide}={
 unit:{id:nongaussianFieldTeachingId,title:t('상호작용·비가우스 연결 상관','Interactions and Non-Gaussian Connected Correlations'),subtitle:t('같은 평균·공분산의 다른 사차 구조를 찾아보고, 정확한 배치 확률·누적량·물리 원천 응답을 비교합니다.','Find different fourth-order structure at the same means and covariance; compare exact configuration probabilities, cumulants and physical source response.'),prerequisiteIds:['finite-gaussian-field-sources-connected-correlations','binary-connected-correlations-field-hessian'],prerequisites:t('가우스 연결 상관과 이진계의 원천·공분산을 복습하세요.','Review Gaussian connected correlations and sources/covariance in binary systems.'),eyebrow:'NON-GAUSSIAN FIELDS · HIGHER CONNECTED STRUCTURE',formula:'κ₄=⟨ηᵢηⱼηₖηₗ⟩−Σ(CC); χ=C/T',accent:'nongaussian-field',questions:nongaussianFieldQuestions.map((q,i)=>({id:`nongaussian-field-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation}))},
 guide:{intuition:t('쌍별로 독립처럼 보여도 네 변수의 곱에는 정보가 남을 수 있습니다. 배치 전체의 확률과 반복 첨자를 함께 읽으며 가우스 쌍짓기의 한계를 확인하세요.','Pairwise independence can hide information in a product of four variables. Read entire configuration probabilities and repeated indices to examine the limits of Gaussian pairings.'),steps:nongaussianFieldSteps.map((s,i)=>({title:s.title,explanation:s.body,formula:s.formula,diagram:diagrams[i]})),experiments:nongaussianFieldExperiments.map(g=>({title:g.title,predict:g.predict,action:t(`${g.examples.map(id=>nongaussianFieldLabels[id].ko).join(' → ')} 예제를 선택하고 배치 확률·공분산·사차량과 온도 경로를 비교하세요.`,`Select ${g.examples.map(id=>nongaussianFieldLabels[id].en).join(' → ')}; compare configuration probabilities, covariance, fourth-order quantities and the temperature path.`),explain:g.explanation,preset:nongaussianFieldExamples.findIndex(c=>c.id===g.examples[0])}))},
};
