import type { LearningUnit, LearningGuide } from './interactive-book';
import { blockOrderTeachingId, type BlockOrderTeachingId } from '@/lib/learning-block-order-teaching';
import { blockOrderSteps, blockOrderExperiments, blockOrderQuestions, blockOrderLabels } from './block-order-teaching';
const t = (ko: string,en: string) => ({ ko,en });
export const blockOrderReader: { unit: Omit<LearningUnit,'id'> & { id: BlockOrderTeachingId }; guide: LearningGuide } = {
  unit: { id: blockOrderTeachingId,title: t('미시상태에서 국소 질서변수로','From Microscopic States to Local Order Parameters'),
    subtitle: t('공간 블록의 정확한 제한 합과 숨긴 엔트로피를 계산하고 국소 평균곱 근사와 조건부 의존성을 비교합니다.','Compute exact constrained spatial-block sums and hidden entropy; compare local mean-product approximations and conditional dependence.'),
    prerequisiteIds: ['binary-shell-multiplicity-block-counts','finite-macro-partition-conditional-entropy','finite-field-measure-and-thermal-free-energy'],
    prerequisites: t('미시 상태 수, 조건부 확률·엔트로피와 자유에너지의 정규화·측도를 복습하세요.','Review microscopic multiplicities, conditional probability and entropy, and free-energy normalization and measures.'),
    eyebrow: 'MICROSTATES · LOCAL ORDER · COARSE STATISTICS',formula: 'Q_k=Σ_d n(k,d)x^d; P(k)=Q_k Πy_b^k_b / Zrel',accent: 'block-order',
    questions: blockOrderQuestions.map((q,i) => ({ id: `block-order-${i+1}`,question: q.prompt,options: [...q.options],answer: q.correct,explanation: q.explanation })) },
  guide: { intuition: t('동네 평균은 골목의 위치를 숨깁니다. 블록 평균을 남길 때 실제 미시 가중치를 합하고, 사라진 경계 정보가 이웃 예측에 어떤 영향을 주는지 확인하세요.','A regional average hides street positions. Sum the actual microscopic weights when retaining block means, and inspect how hidden boundary detail affects neighboring predictions.'),
    steps: blockOrderSteps.map(s => ({ title: s.title,explanation: s.body,formula: s.formula,diagram: s.caption })),
    experiments: blockOrderExperiments.map((g,i) => ({ title: g.title,predict: g.predict,explain: g.explanation,preset: i,
      action: t(`${g.examples.map(id => blockOrderLabels[id].ko).join(' → ')} 순서로 비교하고 적용한 프로필·조건부 평균·정확한 확률과 이웃 판정을 확인하세요.`,`Compare ${g.examples.map(id => blockOrderLabels[id].en).join(' → ')}; inspect applied profiles, conditional means, exact probabilities and actual adjacency.`) })) },
};
