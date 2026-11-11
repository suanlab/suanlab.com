import type { LearningGuide, LearningUnit } from './interactive-book';

export const illPosednessId = 'ill-posedness-and-singular-modes' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const illPosednessUnit: LearningUnit = {
  id: illPosednessId,
  title: t('비정칙 역문제와 특이 모드', 'Ill-Posedness and Singular Modes'),
  subtitle: t('약한 관측 방향에서 오차가 얼마나 커지는지 계산합니다.', 'Calculate error amplification along a weak observation direction.'),
  prerequisites: t('선형 사상, 노름과 특잇값을 복습하세요.', 'Review linear maps, norms, and singular values.'),
  prerequisiteIds: ['forward-and-inverse-maps-for-cooling'],
  eyebrow: 'INVERSE STABILITY',
  formula: 'x̂₂ = 1 + δ/s',
  accent: 'block-order',
  questions: [
    {
      id: 'singular-weak-direction',
      question: t('둘째 특잇값을 0.1에서 0.01로 줄이면 같은 둘째 관측 오차의 증폭은?', 'How does reducing the second singular value from 0.1 to 0.01 change amplification of the same weak-mode error?'),
      options: [t('10배 커집니다.', 'It increases tenfold.'), t('10분의 1로 줄어듭니다.', 'It falls to one tenth.'), t('같습니다.', 'It stays the same.')],
      answer: 0,
      explanation: t('복원 오차는 특잇값의 역수에 비례합니다.', 'Reconstruction error is proportional to the reciprocal singular value.'),
    },
    {
      id: 'singular-zero-direction',
      question: t('둘째 특잇값과 둘째 관측값이 모두 0이면 x₂는 어떻게 되나요?', 'If the second singular value and the second reading are both zero, what can be inferred about x₂?'),
      options: [t('어느 x₂도 같은 관측을 줍니다.', 'Every x₂ gives the same reading.'), t('x₂는 반드시 0입니다.', 'x₂ must be zero.'), t('어느 x₂도 가능한 해가 아닙니다.', 'No x₂ is compatible.')],
      answer: 0,
      explanation: t('순방향 사상은 둘째 방향 전체를 0으로 보냅니다.', 'The forward map sends the entire second direction to zero.'),
    },
  ],
};

export const illPosednessGuide: LearningGuide = {
  intuition: t('약한 방향은 관측에서는 작게 보이지만 역산 과정에서 크게 증폭됩니다.', 'A weak direction contributes little to the data and is amplified by inversion.'),
  steps: [{
    title: t('특잇값으로 나누기', 'Divide by a singular value'),
    explanation: t('순방향에서 s를 곱하면 역방향에서는 s로 나눕니다. 관측 오차에도 같은 역연산이 적용됩니다.', 'The forward map multiplies by s and the inverse divides by s, including any observation error.'),
    formula: 'A = diag(1, s),  κ₂(A) = 1/s',
    diagram: t('약한 방향의 오차 곡선은 s가 작아질수록 커집니다.', 'The weak-direction error grows as s becomes small.'),
  }],
  experiments: [
    {
      title: t('같은 오차를 다른 방향에 넣기', 'Apply the same error in two directions'),
      predict: t('관측 오차 0.01을 약한 방향에서 강한 방향으로 옮기면 복원은 얼마나 달라질까요?', 'How does moving error 0.01 from the weak to the strong direction change reconstruction?'),
      action: t('s=0.01에서 오차 방향을 바꾸세요.', 'At s=0.01, change the error direction.'),
      explain: t('약한 방향의 오차는 1이며 강한 방향의 오차는 0.01입니다.', 'Weak-direction error is one and strong-direction error is 0.01.'),
      preset: 0,
    },
    {
      title: t('특잇값을 0으로 두기', 'Set the singular value to zero'),
      predict: t('둘째 관측 오차가 있을 때와 없을 때 해의 개수는 어떻게 될까요?', 'How many exact solutions are there with and without second-reading error?'),
      action: t('s=0에서 오차 방향을 약한 방향과 강한 방향으로 바꾸세요.', 'At s=0, switch the error between weak and strong directions.'),
      explain: t('둘째 관측이 0이면 해가 무한히 많고, 0이 아니면 정확한 해가 없습니다.', 'There are infinitely many exact solutions for a zero second reading and no exact solution otherwise.'),
      preset: 1,
    },
  ],
};
