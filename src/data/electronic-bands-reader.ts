import type { LearningGuide, LearningUnit } from './interactive-book';

export const electronicBandsId = 'electronic-bands-in-a-dimerized-chain' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const electronicBandsUnit: LearningUnit = {
  id: electronicBandsId,
  title: t('이합체 사슬의 전자 밴드', 'Electronic Bands in a Dimerized Chain'),
  subtitle: t('두 자리 셀에서 호핑과 자리 에너지가 밴드 간격을 어떻게 바꾸는지 살펴봅니다.', 'Explore how hopping and on-site energies change the band gap in a two-site cell.'),
  prerequisites: t('격자, 브릴루앙 영역과 고윳값을 복습하세요.', 'Review lattices, Brillouin zones, and eigenvalues.'),
  prerequisiteIds: ['lattice-vibrations-and-phonon-dispersion'],
  eyebrow: 'ELECTRONIC BANDS',
  formula: 'H(q)ψ = Eψ',
  accent: 'block-order',
  questions: [
    {
      id: 'bands-folded-crossing',
      question: t('두 호핑이 같고 자리 에너지도 같을 때 영역 경계에서 무슨 일이 일어나나요?', 'What happens at the zone edge when both hoppings and both on-site energies are equal?'),
      options: [t('두 에너지 가지가 만납니다.', 'The two bands touch.'), t('간격이 가장 커집니다.', 'The gap reaches its maximum.'), t('두 가지가 모두 평평해집니다.', 'Both bands become flat.')],
      answer: 0,
      explanation: t('경계에서 셀 안팎의 호핑 항이 상쇄됩니다.', 'The intercell and intracell hopping contributions cancel at the edge.'),
    },
    {
      id: 'bands-gap-ambiguity',
      question: t('간격 한 개를 측정하면 호핑 차이와 자리 에너지 차이를 각각 구할 수 있나요?', 'Can a single gap measurement determine both the hopping contrast and the on-site energy difference?'),
      options: [t('두 효과를 구분하려면 관측을 더 해야 합니다.', 'Additional observations are needed to distinguish the effects.'), t('항상 두 값을 모두 결정합니다.', 'It always fixes both values.'), t('간격은 자리 에너지와 관계가 없습니다.', 'The gap is independent of on-site energy.')],
      answer: 0,
      explanation: t('두 효과가 제곱합으로 들어갑니다. 자리별 가중치를 추가로 관측하면 후보를 더 좁힐 수 있습니다.', 'The two effects enter as a sum of squares. Site-resolved weights further constrain the candidates.'),
    },
  ],
};

export const electronicBandsGuide: LearningGuide = {
  intuition: t('두 자리 사이의 호핑은 파수에 따라 더해지거나 상쇄됩니다.', 'Hopping contributions reinforce or cancel according to the wave number.'),
  steps: [{
    title: t('영역 경계의 간격', 'The zone-edge gap'),
    explanation: t('영역 경계의 위상은 −1입니다. 호핑 차이와 자리 에너지 차이가 간격을 결정합니다.', 'The intercell phase is −1 at the zone edge. Hopping contrast and on-site energy difference set the gap.'),
    formula: 'E_g = 2√[Δ² + (t₁ − t₂)²]',
    diagram: t('두 가지가 영역 경계에서 만날 수 있습니다.', 'The two branches can meet at the zone edge.'),
  }],
  experiments: [
    {
      title: t('간격 닫기', 'Close the gap'),
      predict: t('두 호핑을 같게 두면 경계에서 어떤 변화가 생길까요?', 'What changes at the edge if both hoppings are equal?'),
      action: t('q/π=1에서 Δ=0을 유지하고 t₂를 1로 바꾸세요.', 'At q/π=1 keep Δ=0 and set t₂ to 1.'),
      explain: t('두 가지가 만나면 에너지로 정렬된 가지의 기울기와 각 고유상태의 자리 가중치는 하나로 정해지지 않습니다.', 'When the bands touch, the sorted branch slope and individual eigenstate weights are undefined.'),
      preset: 0,
    },
    {
      title: t('같은 간격, 다른 상태', 'Equal gap, different states'),
      predict: t('t₂=0.6, Δ=0과 같은 간격을 t₂=1에서 얻으려면 Δ를 얼마로 둘까요?', 'With t₂=1, which Δ gives the same gap as t₂=0.6 and Δ=0?'),
      action: t('경계에서 Δ를 0.4로 바꾸고 A 성분 가중치를 비교하세요.', 'At the edge set Δ to 0.4 and compare the A weight.'),
      explain: t('간격은 0.8로 돌아오며 A 가중치는 0.5에서 1로 바뀝니다.', 'The gap returns to 0.8 and the A weight changes from 0.5 to 1.'),
      preset: 1,
    },
  ],
};
