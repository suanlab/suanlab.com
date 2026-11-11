import type { LearningGuide, LearningUnit } from './interactive-book';

export const dimensionConvergenceId = 'dimensions-limits-and-convergence-checks' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const dimensionConvergenceUnit: LearningUnit = {
  id: dimensionConvergenceId,
  title: t('단위·극한·보존과 수렴 검사', 'Dimensions, Limits, Invariants and Convergence'),
  subtitle: t('정확한 진동자와 수치 궤적을 같은 시간에서 비교합니다.', 'Compare a numerical trajectory with an exact oscillator at a fixed horizon.'),
  prerequisites: t('미분방정식의 시간 간격, 에너지와 수렴을 복습하세요.', 'Review time steps, energy, and convergence of an ODE solver.'),
  prerequisiteIds: ['availability-leakage-and-reproducible-evaluation'],
  eyebrow: 'SCIENTIFIC VALIDATION',
  formula: 'Eₙ₊₁ = (1 + h²)Eₙ',
  accent: 'block-order',
  questions: [
    {
      id: 'convergence-fixed-horizon',
      question: t('시간 간격 수렴을 비교할 때 무엇을 고정해야 하나요?', 'What should stay fixed when checking time-step convergence?'),
      options: [t('최종 물리 시간과 초기 조건', 'The final physical time and initial conditions'), t('단계 수만', 'Only the number of steps'), t('그림의 데이터 점 개수만', 'Only the number of plotted points')],
      answer: 0,
      explanation: t('다른 최종 시간의 오차를 비교하면 간격의 효과와 시간이 섞입니다.', 'Comparing different final times mixes time-step and horizon effects.'),
    },
    {
      id: 'convergence-euler-energy',
      question: t('이 명시적 오일러 갱신에서 0이 아닌 간격의 에너지는?', 'What happens to energy under this explicit Euler update at a nonzero step?'),
      options: [t('매 단계 증가합니다.', 'It increases every step.'), t('정확히 보존됩니다.', 'It is exactly conserved.'), t('항상 0이 됩니다.', 'It always becomes zero.')],
      answer: 0,
      explanation: t('에너지에 곱해지는 1+h²가 1보다 큽니다.', 'The per-step factor 1+h² exceeds one.'),
    },
  ],
};

export const dimensionConvergenceGuide: LearningGuide = {
  intuition: t('눈에 비슷한 궤적에서도 보존량과 세분화 오차를 계산하면 방법의 성질이 드러납니다.', 'Invariant and refinement errors reveal numerical behavior even when trajectories look similar.'),
  steps: [{
    title: t('교차항이 사라지는 에너지 계산', 'Energy after the cross terms cancel'),
    explanation: t('다음 단계의 두 좌표 제곱을 더해 에너지 비율을 구합니다.', 'Add the squared next-step coordinates to obtain the energy ratio.'),
    formula: 'xₙ₊₁² + pₙ₊₁² = (1+h²)(xₙ² + pₙ²)',
    diagram: t('정확한 원과 바깥으로 나가는 오일러 궤적을 비교합니다.', 'Compare the exact circle with the outward Euler trajectory.'),
  }],
  experiments: [
    {
      title: t('고정 시간에서 세분화', 'Refine at a fixed horizon'),
      predict: t('20단계에서 160단계로 바꾸면 에너지 비율은 어떻게 될까요?', 'How does the energy ratio change from 20 to 160 steps?'),
      action: t('단계 수를 최대로 올리세요.', 'Raise the step count to its maximum.'),
      explain: t('비율이 약 2.19에서 약 1.105로 내려가고 상태 오차도 작아집니다.', 'The ratio falls from about 2.19 to about 1.105, and state error shrinks.'),
      preset: 0,
    },
    {
      title: t('유도한 식으로 검산', 'Check the derived formula'),
      predict: t('직접 궤적을 계산한 에너지와 닫힌 식은 같을까요?', 'Do trajectory energy and the closed-form ratio agree?'),
      action: t('단계 수를 바꾸며 두 비율을 비교하세요.', 'Change the step count and compare both ratios.'),
      explain: t('반올림 범위 안에서 같은 비율을 얻습니다.', 'They agree within rounding precision.'),
      preset: 1,
    },
  ],
};
