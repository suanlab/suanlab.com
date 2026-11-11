import type { LearningGuide, LearningUnit } from './interactive-book';

export const landauerErasureId = 'landauer-erasure-and-its-assumptions' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const landauerErasureUnit: LearningUnit = {
  id: landauerErasureId,
  title: t('Landauer 초기화와 가정', 'Landauer Erasure and Its Assumptions'),
  subtitle: t('입력 분포와 초기화 정확도로부터 메모리 엔트로피 감소를 계산합니다.', 'Calculate memory entropy reduction from input probabilities and reset accuracy.'),
  prerequisites: t('이진 엔트로피와 열역학 제2법칙을 복습하세요.', 'Review binary entropy and the second law of thermodynamics.'),
  prerequisiteIds: ['energy-per-task-and-thermal-cost'],
  eyebrow: 'INFORMATION AND ERASURE',
  formula: 'Q ≥ kᵦT[h(p) − h(ε)]',
  accent: 'block-order',
  questions: [
    {
      id: 'landauer-known-input',
      question: t('초기 상태가 항상 0이면 완전한 초기화에서 이진 엔트로피는 얼마나 감소하나요?', 'If the memory always starts in state 0, how much binary entropy does perfect reset remove?'),
      options: [t('0입니다.', 'Zero'), t('ln 2입니다.', 'ln 2'), t('온도 T입니다.', 'The temperature T')],
      answer: 0,
      explanation: t('초기 상태와 최종 상태가 모두 확정되어 있습니다.', 'The initial and final logical states are both certain.'),
    },
    {
      id: 'landauer-reset-conditions',
      question: t('열 하한을 적용하기 전에 어떤 정보가 필요한가요?', 'What information is needed before applying the heat bound?'),
      options: [t('초기 분포, 최종 오차, 열저장고와 제어기의 조건', 'The initial distribution, final error, and bath and controller conditions'), t('연산 수만', 'Only the operation count'), t('장치의 최대 처리율만', 'Only the peak device throughput')],
      answer: 0,
      explanation: t('하한은 지정한 초기화 과정의 엔트로피 수지에서 나옵니다.', 'The bound follows from the entropy balance of the specified reset.'),
    },
  ],
};

export const landauerErasureGuide: LearningGuide = {
  intuition: t('메모리의 불확실성이 줄면 그 엔트로피를 물리적으로 어디에 옮겼는지 추적할 수 있습니다.', 'When memory uncertainty decreases, the entropy balance tracks where it goes physically.'),
  steps: [{
    title: t('두 엔트로피의 차이', 'Difference of two entropies'),
    explanation: t('초기 상태와 최종 상태의 이진 엔트로피를 각각 계산한 뒤 차이에 kᵦT를 곱합니다.', 'Compute initial and final binary entropies, then multiply their difference by kᵦT.'),
    formula: 'Q ≥ kᵦT[h(p) − h(ε)]',
    diagram: t('초기 확률과 최종 오차가 엔트로피 곡선에서 다른 높이에 놓입니다.', 'Initial probability and final error mark different heights on the entropy curve.'),
  }],
  experiments: [
    {
      title: t('균등 입력에서 남는 오차', 'Residual error for a fair input'),
      predict: t('초기 상태가 균등할 때 최종 오차를 허용하면 하한은 어떻게 변할까요?', 'What happens to the bound when residual error is allowed for a fair input?'),
      action: t('p=0.5에서 ε를 0에서 0.1로 높이세요.', 'At p=0.5, raise ε from zero to 0.1.'),
      explain: t('최종 상태의 엔트로피가 커지므로 엔트로피 감소량이 작아집니다.', 'The final state has higher entropy, reducing the entropy decrease.'),
      preset: 0,
    },
    {
      title: t('이미 아는 비트', 'A bit already known'),
      predict: t('완전히 확정된 초기 비트를 0으로 초기화할 때 정보에 해당하는 열 하한은 얼마일까요?', 'What is the informational heat bound for perfectly resetting a known bit to zero?'),
      action: t('p와 ε를 모두 0으로 두세요.', 'Set both p and ε to zero.'),
      explain: t('두 엔트로피가 모두 0이므로 하한은 0입니다.', 'Both entropies are zero, so the bound is zero.'),
      preset: 1,
    },
  ],
};
