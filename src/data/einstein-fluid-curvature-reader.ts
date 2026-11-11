import type { LearningGuide, LearningUnit } from './interactive-book';

export const einsteinFluidCurvatureId = 'einstein-equations-trace-and-sources' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const einsteinFluidCurvatureUnit: LearningUnit = {
  id: einsteinFluidCurvatureId,
  title: t('아인슈타인 방정식의 구조와 물질 원천', 'The Structure and Matter Sources of the Einstein Equations'),
  subtitle: t('완전 유체의 대각합과 리치 성분을 구하고 먼지·복사·진공을 비교합니다.', 'Compute the trace and Ricci components of a perfect fluid and compare dust, radiation and vacuum.'),
  prerequisites: t('계량, 공변 미분과 곡률 수축을 복습하세요.', 'Review metrics, covariant derivatives, and curvature contractions.'),
  prerequisiteIds: ['curvature-covariant-derivatives-and-holonomy'],
  eyebrow: 'EINSTEIN FIELD EQUATIONS',
  formula: 'Gᵤᵥ + Λgᵤᵥ = κTᵤᵥ',
  accent: 'block-order',
  questions: [
    {
      id: 'einstein-radiation-trace',
      question: t('Λ=0인 복사에서 R=0이면 리치 텐서 전체도 0인가요?', 'At Λ=0, does R=0 for radiation imply that the whole Ricci tensor vanishes?'),
      options: [t('아니요. 개별 리치 성분은 남습니다.', 'No. Individual Ricci components remain.'), t('예. 대각합이 모든 성분을 정합니다.', 'Yes. The trace determines every component.'), t('압력이 음수일 때만 알 수 있습니다.', 'Only negative pressure allows a conclusion.')],
      answer: 0,
      explanation: t('복사는 T=0이지만 ρ가 양수이면 R₀₀=κρ입니다.', 'Radiation has T=0, yet R₀₀=κρ when the density is positive.'),
    },
    {
      id: 'einstein-vacuum-contraction',
      question: t('Λ=0인 진공에서 장방정식이 직접 정하는 조건은?', 'What does the field equation directly require in vacuum with Λ=0?'),
      options: [t('리치 텐서가 0이다.', 'The Ricci tensor vanishes.'), t('리만 텐서의 모든 성분이 0이다.', 'Every Riemann component vanishes.'), t('좌표에서 계량의 모든 도함수가 0이다.', 'All coordinate derivatives of the metric vanish.')],
      answer: 0,
      explanation: t('리치 텐서는 리만 텐서의 수축입니다. 진공 조건은 그 수축을 정합니다.', 'The Ricci tensor is a contraction of the Riemann tensor. The vacuum equation constrains the contraction.'),
    },
  ],
};

export const einsteinFluidCurvatureGuide: LearningGuide = {
  intuition: t('하나의 대각합은 모든 방향의 곡률 성분을 담지 못합니다.', 'A single trace cannot retain the curvature components in every direction.'),
  steps: [{
    title: t('계량으로 수축하기', 'Contract with the metric'),
    explanation: t('정규직교 기저에서 시간 성분의 부호를 바꿔 공간 성분과 더합니다.', 'In the orthonormal frame, reverse the time component’s sign before adding the spatial components.'),
    formula: 'R = −R₀₀ + 3Rᵢᵢ',
    diagram: t('시간 방향과 공간 방향의 리치 성분을 따로 계산합니다.', 'Calculate the time and spatial Ricci components separately.'),
  }],
  experiments: [
    {
      title: t('먼지와 복사', 'Dust and radiation'),
      predict: t('밀도가 같을 때 복사와 먼지의 스칼라 곡률은 어떻게 다를까요?', 'At equal density, how do scalar curvatures differ for dust and radiation?'),
      action: t('먼지와 복사 예제를 고르고 R과 R₀₀을 비교하세요.', 'Choose the dust and radiation examples and compare R and R₀₀.'),
      explain: t('복사에서는 R이 0이고 시간 방향 리치 성분은 남습니다.', 'Radiation has zero scalar curvature and nonzero time Ricci curvature.'),
      preset: 0,
    },
    {
      title: t('진공과 우주 상수', 'Vacuum and the cosmological constant'),
      predict: t('밀도가 0이어도 Λ를 바꾸면 어떤 곡률이 생길까요?', 'At zero density, what curvature appears when Λ changes?'),
      action: t('진공 예제에서 Λ를 바꾸고 R이 4Λ인지 확인하세요.', 'In the vacuum example, vary Λ and check that R=4Λ.'),
      explain: t('이 경우 리치 텐서는 Λ와 계량의 곱입니다.', 'The Ricci tensor equals Λ times the metric in this case.'),
      preset: 1,
    },
  ],
};
