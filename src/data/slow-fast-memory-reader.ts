import type { LearningGuide, LearningUnit } from './interactive-book';

export const slowFastMemoryId = 'scale-separation-memory-and-adiabatic-elimination' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const slowFastMemoryUnit: LearningUnit = {
  id: slowFastMemoryId,
  title: t('척도 분리, 기억과 빠른 변수 제거', 'Scale Separation, Memory and Adiabatic Elimination'),
  subtitle: t('빠른 변수를 제거해 기억항을 구하고 순간 응답 근사의 오차를 확인합니다.', 'Eliminate a fast variable, recover the memory term, and test the instantaneous approximation.'),
  prerequisites: t('선형 미분방정식의 고윳값과 유효 이론을 복습하세요.', 'Review eigenvalues of linear differential equations and effective theories.'),
  prerequisiteIds: ['effective-theory-heavy-mode-matching', 'singular-perturbation-boundary-layer-matching'],
  eyebrow: 'SCALE SEPARATION',
  formula: 'ẋ = −x + y,  ẏ = D(γx − y)',
  accent: 'block-order',
  questions: [
    {
      id: 'slow-fast-hidden-state',
      question: t('같은 x₀에서 y₀를 바꾸면 어떤 값이 즉시 달라지나요?', 'At fixed x₀, which quantity changes immediately when y₀ changes?'),
      options: [t('전체 계의 초기 기울기', 'The full system’s initial slope'), t('x₀', 'The initial x₀'), t('고윳값', 'The eigenvalues')],
      answer: 0,
      explanation: t('초기 기울기는 −x₀+y₀입니다. 고윳값은 방정식의 계수로 정해집니다.', 'The initial slope is −x₀+y₀. Eigenvalues depend on the equation coefficients.'),
    },
    {
      id: 'slow-fast-closure-lag',
      question: t('y₀=γx₀로 시작하면 남는 잔차의 원인은 무엇인가요?', 'What can produce residual error when y₀=γx₀?'),
      options: [t('빠른 변수가 응답하는 동안 x가 변하는 효과', 'Changes in x during the fast response'), t('초기 y₀의 불일치', 'A mismatch in the initial y₀'), t('축소 모형의 임의 표본', 'Random samples from the reduced model')],
      answer: 0,
      explanation: t('초기 불일치 항은 0이 됩니다. 기억항은 빠른 응답의 폭에서 x가 얼마나 바뀌는지에 따라 남습니다.', 'The initial mismatch vanishes. Memory remains when x varies during the fast response.'),
    },
  ],
};

export const slowFastMemoryGuide: LearningGuide = {
  intuition: t('느린 변수가 같은 값에서 시작해도 숨은 빠른 변수가 다르면 다음 순간의 기울기가 달라집니다.', 'The same initial slow value can yield a different next slope when the hidden fast value differs.'),
  steps: [
    {
      title: t('초기 조건이 남기는 흔적', 'The trace of an initial condition'),
      explanation: t('빠른 변수의 초기값은 지수적으로 감소하는 항으로 느린 식에 들어갑니다.', 'The initial fast value enters the slow equation through an exponentially decaying term.'),
      formula: 'y₀e⁻ᴰᵗ',
      diagram: t('같은 x₀에서 y₀에 따라 초기 기울기가 바뀝니다.', 'The initial slope varies with y₀ at fixed x₀.'),
    },
  ],
  experiments: [
    {
      title: t('숨은 초기값 바꾸기', 'Change the hidden initial value'),
      predict: t('x₀가 같으면 초기 궤적도 같을까요?', 'Will trajectories initially agree when they share x₀?'),
      action: t('y₀를 0과 γ+1로 바꾸고 초기 구간을 확대하세요.', 'Switch y₀ between 0 and γ+1, and zoom into the initial interval.'),
      explain: t('전체 계의 기울기는 y₀에 따라 달라집니다. 순간 응답 근사는 두 경우에 같은 궤적을 냅니다.', 'The full initial slope depends on y₀. The instantaneous approximation yields the same trajectory for both cases.'),
      preset: 0,
    },
    {
      title: t('감쇠 시간 차이 늘리기', 'Increase the separation of decay times'),
      predict: t('D를 크게 하면 관측 시점의 오차가 얼마나 줄까요?', 'How much does the error at the observation time shrink when D grows?'),
      action: t('y₀=γ에서 D를 4와 32로 설정하고 같은 관측 시점에서 비교하세요.', 'At y₀=γ, compare D=4 and D=32 at the same observation time.'),
      explain: t('빠른 반응의 시간 폭이 줄면 과거 값을 현재 값으로 대체한 오차도 줄어듭니다.', 'Narrowing the fast response reduces the error from replacing recent past values with the current value.'),
      preset: 1,
    },
  ],
};
