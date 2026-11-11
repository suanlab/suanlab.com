import type { LearningGuide, LearningUnit } from './interactive-book';

export const discreteAdjointId = 'discrete-adjoint-for-terminal-control' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const discreteAdjointUnit: LearningUnit = {
  id: discreteAdjointId,
  title: t('이산 수반법과 민감도', 'Discrete Adjoint Sensitivity'),
  subtitle: t('상태를 순방향으로 계산하고 손실의 미분을 역방향으로 전파합니다.', 'Compute states forward and propagate loss derivatives backward.'),
  prerequisites: t('연쇄법칙, 이산 상태 갱신과 방향 미분을 복습하세요.', 'Review the chain rule, discrete updates, and directional derivatives.'),
  prerequisiteIds: ['gaussian-bayesian-inversion'],
  eyebrow: 'SENSITIVITY AND ADJOINTS',
  formula: 'pₙ = apₙ₊₁',
  accent: 'block-order',
  questions: [
    {
      id: 'adjoint-terminal-condition',
      question: t('최종 오차 제곱의 절반을 손실로 쓰면 마지막 수반 변수는?', 'For half the squared terminal error, what is the terminal adjoint?'),
      options: [t('최종 상태에서 목표를 뺀 값', 'Final state minus target'), t('최종 상태의 제곱', 'The square of the final state'), t('입력의 개수', 'The number of controls')],
      answer: 0,
      explanation: t('최종 손실을 xₙ에서 미분한 값으로 시작합니다.', 'Start with the derivative of the terminal loss with respect to xₙ.'),
    },
    {
      id: 'adjoint-control-index',
      question: t('입력 uₙ이 직접 바꾸는 상태는 무엇인가요?', 'Which state does input uₙ directly change?'),
      options: [t('다음 상태 xₙ₊₁', 'The next state xₙ₊₁'), t('현재 상태 xₙ', 'The current state xₙ'), t('항상 초기 상태 x₀', 'Always the initial state x₀')],
      answer: 0,
      explanation: t('갱신식에서 입력은 다음 상태에 더해지므로 입력 미분에는 pₙ₊₁이 들어갑니다.', 'The update adds the input to the next state, so the derivative uses pₙ₊₁.'),
    },
  ],
};

export const discreteAdjointGuide: LearningGuide = {
  intuition: t('마지막 상태의 손실이 바뀌는 방향을 거꾸로 따라가면 앞선 입력의 효과를 얻습니다.', 'Tracing the terminal loss derivative backward reveals how earlier controls affect it.'),
  steps: [{
    title: t('순방향 상태와 역방향 민감도', 'Forward states and reverse sensitivity'),
    explanation: t('최종 오차에서 시작해 갱신식의 미분을 차례로 곱합니다.', 'Start from the terminal error and multiply by each update derivative.'),
    formula: '∂J/∂uₙ = pₙ₊₁ + γuₙ',
    diagram: t('상태와 수반 변수를 같은 단계 번호에서 비교합니다.', 'Compare state and adjoint values at each step index.'),
  }],
  experiments: [
    {
      title: t('손으로 계산한 두 단계', 'The two-step example by hand'),
      predict: t('a=0.5이고 입력이 0이면 첫째와 둘째 입력 중 어느 쪽의 미분이 더 클까요?', 'At a=0.5 with zero controls, which control derivative is larger?'),
      action: t('기본 설정에서 두 입력 미분을 비교하세요.', 'Compare the two control derivatives at the defaults.'),
      explain: t('둘째 미분은 0.25, 첫째 미분은 0.125입니다.', 'The second derivative is 0.25 and the first is 0.125.'),
      preset: 0,
    },
    {
      title: t('최종 오차를 없애기', 'Remove terminal error'),
      predict: t('목표를 최종 상태와 같게 두면 수반 변수는 어떻게 될까요?', 'What happens to the adjoint when the target equals the final state?'),
      action: t('a=0.5와 입력 0을 유지하고 목표를 0.25로 바꾸세요.', 'Keep a=0.5 and both controls zero, and set the target to 0.25.'),
      explain: t('최종 오차가 0이므로 수반 변수가 모든 단계에서 0입니다.', 'Zero terminal error gives zero adjoint at every step.'),
      preset: 1,
    },
  ],
};
