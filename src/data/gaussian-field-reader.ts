import type {LearningUnit,LearningGuide} from './interactive-book';
import {gaussianFieldTeachingId,type GaussianFieldTeachingId} from '@/lib/learning-gaussian-field-teaching';
import {gaussianFieldSteps,gaussianFieldExperiments,gaussianFieldQuestions,gaussianFieldLabels} from './gaussian-field-teaching';
import {gaussianFieldExamples} from '@/lib/learning-gaussian-field-input';
const t=(ko:string,en:string)=>({ko,en});
const diagrams=[
 t('네 위치의 주기 고리와 한 배치를 그립니다. 시간 표본이 아닙니다.','Four sites form a periodic ring; the bars show one configuration, not time samples.'),
 t('기본 네 위치 고리의0,1 성분: 1/5+7/75=22/75입니다.','Default four-site ring, entry 0,1: 1/5+7/75=22/75.'),
 t('기본 K와 J를 유지하며 T=1,2를 비교합니다.','Compare T=1,2 while keeping the default K and J fixed.'),
 t('네 삽입을 두 쌍으로 묶는 세 가지 방법을 그립니다.','Three ways to pair four insertions; positions can repeat.'),
 t('기본 고리에서 올바른 누적량은0, 잘못된 원래 쌍 빼기는−28/5625입니다.','On the default ring the correct cumulant is0; incorrect raw-pair subtraction gives −28/5625.'),
 t('질량0 고리의 상수 모드는 에너지가0입니다. 평균0 제약에서는 그 방향을 제외합니다.','A massless ring has zero energy along its constant mode. The mean-zero constraint excludes that direction.'),
 t('질량0·강성1·온도1인 바깥 고정 세 위치의 첫 분산을 비교합니다.','Compare first-site variances for three pinned massless sites with stiffness and temperature1.'),
 t('기본 고리의 연결 공분산: 행과 열은 이산 위치 첨자입니다.','Connected covariance of the default ring; rows and columns are discrete site indices.'),
];
export const gaussianFieldReader:{unit:Omit<LearningUnit,'id'>&{id:GaussianFieldTeachingId};guide:LearningGuide}={
 unit:{id:gaussianFieldTeachingId,title:t('유한 가우스 장의 원천과 연결 상관','Finite Gaussian Fields: Sources and Connected Correlations'),
 subtitle:t('원천·경계·제약을 바꾸며 평균과 요동, Wick 모멘트, 응답과 측도를 정확한 분수와 이산 시각화로 비교합니다.','Compare means, fluctuations, Wick moments, response and measure using exact fractions and discrete visualizations as sources, boundaries and constraints change.'),
 prerequisiteIds:['field-fourier-modes-lattice-continuum','multivariate-gaussians'],prerequisites:t('이산 장과 다변량 가우스 분포를 복습하세요.','Review discrete fields and multivariate Gaussian distributions.'),
 eyebrow:'GAUSSIAN FIELDS · SOURCES AND CORRELATIONS',formula:'μ=GJ; C=TG; ⟨φφᵀ⟩=C+μμᵀ',accent:'gaussian-field',
 questions:gaussianFieldQuestions.map((q,i)=>({id:`gaussian-field-${i+1}`,question:q.prompt,options:[...q.options],answer:q.correct,explanation:q.explanation}))},
 guide:{intuition:t('평균이 움직이는 것과 요동이 연결되는 것은 다릅니다. 이웃을 묶는 국소 에너지가 먼 위치의 공분산을 만드는 과정을 살펴보세요.','Moving a mean differs from correlating fluctuations. Explore how local energy coupling creates covariance between distant sites.'),
 steps:gaussianFieldSteps.map((s,i)=>({title:s.title,explanation:s.body,formula:s.formula,diagram:diagrams[i]})),
 experiments:gaussianFieldExperiments.map(g=>({title:g.title,predict:g.predict,action:t(`${g.examples.map(id=>gaussianFieldLabels[id].ko).join(' → ')} 예제를 선택하고 평균·연결/원래 행렬·정확한 잔차를 비교하세요.`,`Select ${g.examples.map(id=>gaussianFieldLabels[id].en).join(' → ')}; compare means, connected/raw matrices and exact residuals.`),explain:g.explanation,preset:gaussianFieldExamples.findIndex(c=>c.id===g.examples[0])}))},
};
