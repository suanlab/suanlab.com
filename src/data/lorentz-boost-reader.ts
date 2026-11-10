import type { LearningGuide, LearningUnit } from './interactive-book';

export const lorentzBoostId = 'special-relativity-lorentz-boosts' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const lorentzBoostUnit: LearningUnit = {
  id: lorentzBoostId,
  title: t('특수상대론과 로런츠 변환', 'Special Relativity and Lorentz Boosts'),
  subtitle: t('관성계를 바꿔 같은 사건의 좌표와 시공간 간격을 비교합니다.', 'Change inertial frames and compare event coordinates and spacetime intervals.'),
  prerequisites: t('벡터 좌표와 선형변환을 복습하세요.', 'Review vector coordinates and linear transformations.'),
  prerequisiteIds: ['linear-transforms'],
  eyebrow: 'SPECIAL RELATIVITY',
  formula: 'ct′ = γ(ct − βx),  x′ = γ(x − βct)',
  accent: 'block-order',
  questions: [
    {
      id: 'lorentz-simultaneity',
      question: t('S에서 동시에 일어난 두 사건에 Δx≠0과 β≠0을 적용하면 Δct′는?', 'For simultaneous events in S with Δx≠0 and β≠0, what is Δct′?'),
      options: [t('−γβΔx', '−γβΔx'), t('0', '0'), t('γΔx', 'γΔx')],
      answer: 0,
      explanation: t('시간 변환식에 Δct=0을 대입합니다.', 'Substitute Δct=0 into the time transformation.'),
    },
    {
      id: 'lorentz-light-ray',
      question: t('x=ct 위의 빛 사건은 부스트 뒤 어디에 놓이나요?', 'Where does a light event on x=ct lie after a boost?'),
      options: [t('x′=ct′', 'x′=ct′'), t('항상 x′=0', 'Always x′=0'), t('항상 ct′=0', 'Always ct′=0')],
      answer: 0,
      explanation: t('x=ct를 두 식에 대입하면 두 변환 성분이 같습니다.', 'Substituting x=ct makes the two transformed coordinates equal.'),
    },
  ],
};

export const lorentzBoostGuide: LearningGuide = {
  intuition: t('관성계가 달라도 같은 사건을 기록합니다. 좌표와 간격을 각각 비교하세요.', 'Both frames describe the same event. Compare its coordinates and its interval.'),
  steps: [{
    title: t('움직이는 원점', 'The moving origin'),
    explanation: t('S에서 x=βct인 선은 S′에서 x′=0입니다.', 'The line x=βct in S is the line x′=0 in S′.'),
    formula: 'x′ = γ(x − βct)',
    diagram: t('움직이는 원점의 좌표를 계산합니다.', 'Calculate the coordinate of the moving origin.'),
  }],
  experiments: [
    {
      title: t('동시성의 차이', 'Relativity of simultaneity'),
      predict: t('두 사건이 S에서 동시에 일어나면 S′에서도 동시에 일어날까요?', 'Will two simultaneous events in S remain simultaneous in S′?'),
      action: t('ct=0과 x=3에서 β를 0.6과 −0.6으로 바꾸고 ct′를 읽으세요.', 'At ct=0 and x=3, compare ct′ for β=0.6 and −0.6.'),
      explain: t('공간적으로 떨어진 두 사건의 시간차는 관성계에 따라 달라집니다.', 'The time difference for spatially separated events depends on the frame.'),
      preset: 0,
    },
    {
      title: t('빛의 경로와 간격', 'Light rays and the interval'),
      predict: t('부스트하면 빛 사건의 간격은 어떻게 바뀔까요?', 'What happens to the interval of a light event under a boost?'),
      action: t('빛 사건 예제를 선택하고 β를 바꿔 두 간격을 비교하세요.', 'Select the light-event example, change β, and compare both intervals.'),
      explain: t('빛의 두 좌표 성분은 함께 바뀌고 제곱 간격은 0을 유지합니다.', 'The two coordinates change together and the squared interval remains zero.'),
      preset: 1,
    },
  ],
};
