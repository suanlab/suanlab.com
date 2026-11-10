/** Lightweight bilingual experiment configuration; numerical imports are types only.
 * Reader and guide registration lives in src/data/interactive-book.ts;
 * this module supplies experiment presets and guided comparisons.
 */
import type { WeightedBoltzmannParameters } from './learning-weighted-boltzmann';
import type { FiniteGibbsParameters } from './learning-finite-gibbs';
import type { QuadraticGibbsParameters } from './learning-quadratic-gibbs';
export const boltzmannGibbsKinds = ['reference-measure-boltzmann-softmax-identifiability', 'finite-gibbs-conditionals-scan-support', 'quadratic-gaussian-gibbs-normalizability'] as const;
export type BoltzmannGibbsKind = typeof boltzmannGibbsKinds[number];
export type BoltzmannGibbsInput =
  | { kind: 'reference-measure-boltzmann-softmax-identifiability'; parameters: WeightedBoltzmannParameters }
  | { kind: 'finite-gibbs-conditionals-scan-support'; parameters: FiniteGibbsParameters }
  | { kind: 'quadratic-gaussian-gibbs-normalizability'; parameters: QuadraticGibbsParameters };
export interface BoltzmannGibbsText { ko: string; en: string }
export interface BoltzmannGibbsPreset { id: string; label: BoltzmannGibbsText; input: BoltzmannGibbsInput }
export interface BoltzmannGibbsGuide { title: BoltzmannGibbsText; predict: BoltzmannGibbsText; action: BoltzmannGibbsText; explain: BoltzmannGibbsText; presetIndices: readonly number[] }
const t = (ko: string, en: string): BoltzmannGibbsText => ({ ko, en });
const kB = 1.380649e-23, gap = kB * Math.log(2);
export const isBoltzmannGibbsKind = (v: string): v is BoltzmannGibbsKind => (boltzmannGibbsKinds as readonly string[]).includes(v);
export const boltzmannGibbsTitles: Record<BoltzmannGibbsKind, BoltzmannGibbsText> = {
  'reference-measure-boltzmann-softmax-identifiability': t('기준 측도와 Boltzmann 가중치', 'Reference measures and Boltzmann weights'),
  'finite-gibbs-conditionals-scan-support': t('조건부 Gibbs 갱신과 연결성', 'Conditional Gibbs updates and connectivity'),
  'quadratic-gaussian-gibbs-normalizability': t('연속 Gaussian과 정규화 가능성', 'Continuous Gaussians and normalizability'),
};
const weighted = (parameters: Partial<WeightedBoltzmannParameters> = {}): BoltzmannGibbsInput => ({ kind: boltzmannGibbsKinds[0], parameters: { energyLevels: [0, gap, 2 * gap], referenceWeights: [1, 2, 1], energyOffset: 0, temperature: 1, ...parameters } });
const gibbs = (parameters: Partial<FiniteGibbsParameters> = {}): BoltzmannGibbsInput => ({ kind: boltzmannGibbsKinds[1], parameters: { couplingEnergy: gap, firstFieldEnergy: 0, secondFieldEnergy: 0, energyOffset: 0, temperature: 1, support: 'full-product', ...parameters } });
const gaussian = (parameters: Partial<QuadraticGibbsParameters> = {}): BoltzmannGibbsInput => ({ kind: boltzmannGibbsKinds[2], parameters: { firstStiffness: 2 * kB, crossStiffness: kB, secondStiffness: 2 * kB, firstSourceEnergy: 0, secondSourceEnergy: 0, energyOffset: 0, temperature: 1, ...parameters } });
const preset = (id: string, ko: string, en: string, input: BoltzmannGibbsInput): BoltzmannGibbsPreset => ({ id, label: t(ko, en), input });
export const boltzmannGibbsPresets: Record<BoltzmannGibbsKind, readonly BoltzmannGibbsPreset[]> = {
  'reference-measure-boltzmann-softmax-identifiability': [
    preset('weighted-counting', '정수 미시상태 다중도', 'Integer microstate multiplicities', weighted()),
    preset('weighted-prior', '정규화 사전·같은 확률', 'Normalized prior, same probabilities', weighted({ referenceWeights: [.25, .5, .25] })),
    preset('weighted-real-reference', '비정수 기준 가중치', 'Noninteger reference weights', weighted({ referenceWeights: [.1, .2, .7] })),
    preset('weighted-common-offset', '큰 공통 이동·같은 간격', 'Large common offset, same gaps', weighted({ energyOffset: 2 ** 60 })),
    preset('weighted-common-scale', '에너지와 온도8배', 'Energy and temperature scaled by eight', weighted({ energyLevels: [0, 8 * gap, 16 * gap], temperature: 8 })),
    preset('weighted-reference-scale', '기준 가중치2의900승배', 'Reference weights scaled by 2^900', weighted({ referenceWeights: [2 ** 900, 2 ** 901, 2 ** 900] })),
    preset('weighted-forbidden-low', '아주 낮은 금지 준위', 'Very low forbidden level', weighted({ energyLevels: [0, -Number.MAX_VALUE, gap], referenceWeights: [1, 0, 1] })),
    preset('weighted-ground-limit', '가중 바닥 상태의0 K 극한', 'Weighted ground-state zero-K limit', weighted({ energyLevels: [0, 0, gap], referenceWeights: [1, 3, .25], temperature: 0 })),
    preset('weighted-negative-temperature', '유계 스펙트럼의 음의 온도', 'Negative temperature on a bounded spectrum', weighted({ temperature: -1 })),
  ],
  'finite-gibbs-conditionals-scan-support': [
    preset('gibbs-ferromagnetic', '순차 흐름의 비가역 반례', 'Nonreversible systematic flow example', gibbs()),
    preset('gibbs-independent', '독립 자리·한 sweep 정규화', 'Independent sites, one-sweep refresh', gibbs({ couplingEnergy: 0 })),
    preset('gibbs-antiferromagnetic', '음의 결합과 반대 흐름', 'Negative coupling and reversed flow', gibbs({ couplingEnergy: -gap })),
    preset('gibbs-zero-sum', '합0·단일 자리 갱신 정지', 'Zero sum, trapped coordinate updates', gibbs({ support: 'fixed-zero-sum' })),
    preset('gibbs-zero-sum-biased', '편향된 목표도 이동 불가', 'Biased target, still no coordinate moves', gibbs({ support: 'fixed-zero-sum', firstFieldEnergy: .5 * kB, secondFieldEnergy: -.25 * kB })),
    preset('gibbs-full-biased', '같은 장·전체 곱 지지집합', 'Same fields, full product support', gibbs({ firstFieldEnergy: .5 * kB, secondFieldEnergy: -.25 * kB })),
    preset('gibbs-weak-interaction', '화면에 안 보이는 비영 결합', 'Nonzero coupling hidden by display rounding', gibbs({ couplingEnergy: 1e-200 })),
    preset('gibbs-rare-transition', '작아도 양의 전이', 'Tiny but positive transitions', gibbs({ couplingEnergy: 400 * kB })),
    preset('gibbs-negative-temperature', '유계 음의 온도 조건부', 'Bounded negative-temperature conditionals', gibbs({ couplingEnergy: .8 * kB, firstFieldEnergy: .35 * kB, secondFieldEnergy: -.2 * kB, temperature: -1 })),
  ],
  'quadratic-gaussian-gibbs-normalizability': [
    preset('gaussian-correlated', '상관된 기본 타원', 'Baseline correlated ellipse', gaussian()),
    preset('gaussian-linear-source', '선형 source·중심 이동', 'Linear source shifts the center', gaussian({ firstSourceEnergy: kB })),
    preset('gaussian-uncoupled', '교차 에너지 없는 원형', 'Circular law without cross energy', gaussian({ crossStiffness: 0 })),
    preset('gaussian-positive-boundary', '엄밀히 양의 작은 행렬식', 'Strictly positive small determinant', gaussian({ firstStiffness: 1, crossStiffness: 1, secondStiffness: 1 + Number.EPSILON })),
    preset('gaussian-flat', '평평한 방향·정규화 불가', 'Flat direction, not normalizable', gaussian({ firstStiffness: 1, crossStiffness: 1, secondStiffness: 1 })),
    preset('gaussian-indefinite', '음의 곡률·정규화 불가', 'Negative curvature, not normalizable', gaussian({ firstStiffness: 1, crossStiffness: 1, secondStiffness: 1 - Number.EPSILON / 2 })),
    preset('gaussian-high-density', '1보다 큰 밀도 높이', 'Density height greater than one', gaussian({ firstStiffness: 100 * kB, crossStiffness: 0, secondStiffness: 100 * kB })),
    preset('gaussian-coordinate-orientation', '방향 반전·절댓값 Jacobian', 'Orientation reversal and absolute Jacobian', gaussian({ firstSourceEnergy: kB, coordinateScale: [-2, 3], coordinateShift: [1, -2] })),
    preset('gaussian-small-positive-temperature', '아주 작은 양의 온도·C 유지', 'Very small positive temperature, retained C', gaussian({ temperature: Number.MIN_VALUE })),
  ],
};
export const boltzmannGibbsGuides: Record<BoltzmannGibbsKind, readonly BoltzmannGibbsGuide[]> = {
  'reference-measure-boltzmann-softmax-identifiability': [
    { title: t('세 엔트로피 질문', 'Three entropy questions'), predict: t('준위 확률이 같으면 기준 엔트로피도 같을까요?', 'Do equal level probabilities imply equal reference entropy?'), action: t('정수 다중도와 정규화 사전을 비교하고 실수 기준도 확인하세요.', 'Compare integer multiplicities with the normalized prior, then inspect real weights.'), explain: t('다중도와 사전은 같은 p에도 다른 Z·상대 엔트로피를 줍니다. 정수일 때만 명시한 미시상태 수로 읽습니다.', 'Multiplicity and prior references can give the same p but different Z and relative entropy. Only the integer-counting interpretation counts the stated microstates.'), presetIndices: [0, 1, 2] },
    { title: t('영점·척도·기준의 모호성', 'Energy-origin, scale and reference ambiguities'), predict: t('같은 확률에서 어떤 양이 달라질까요?', 'Which quantities can differ at the same probabilities?'), action: t('공통 에너지 이동·에너지/온도 동시 확대·기준 재조정을 비교하세요.', 'Compare a common offset, joint energy/temperature scaling and reference rescaling.'), explain: t('영점 이동은 U,F를 옮기고, 기준 재조정은 엔트로피와F를 바꿉니다. 확률만으로 이 선택을 식별할 수 없습니다.', 'An energy shift moves U and F; reference rescaling changes entropy and F. Probabilities alone cannot identify those choices.'), presetIndices: [3, 4, 5] },
    { title: t('금지 상태와 저온 극한', 'Forbidden states and a low-temperature limit'), predict: t('낮은 에너지면 항상 바닥 상태이고 동률이면 균등할까요?', 'Is every low-energy label allowed, and are ties necessarily uniform?'), action: t('금지 준위·가중 바닥·유계 음의 온도를 차례로 확인하세요.', 'Inspect the forbidden level, weighted ground limit and bounded negative temperature.'), explain: t('m=0은 제외하고 바닥의 양의 기준 가중치로 극한을 정합니다. 작은 양수와 정확한0은 다릅니다.', 'Exclude m=0 and use positive ground weights for the limit. A tiny positive mass differs from an exact zero.'), presetIndices: [6, 7, 8] },
  ],
  'finite-gibbs-conditionals-scan-support': [
    { title: t('정상성과 상세평형', 'Stationarity and detailed balance'), predict: t('각 자리 갱신이 가역이면 순차 곱도 가역일까요?', 'If coordinate updates are reversible, is their ordered product reversible?'), action: t('양의 결합·독립·음의 결합의 흐름과 갱신 수를 비교하세요.', 'Compare flows and update counts for positive, zero and negative coupling.'), explain: t('순차 곱은 π를 보존하지만 일반적으로 상세평형은 깨집니다. random 한 단계와 sweep의 좌표 갱신 수도 다릅니다.', 'The product preserves pi but generally breaks detailed balance. A random step and a sweep also resample different numbers of coordinates.'), presetIndices: [0, 1, 2] },
    { title: t('제약이 끊는 연결', 'Connections removed by support'), predict: t('정상분포가 있어도 한 상태에 갇힐 수 있을까요?', 'Can a chain be trapped despite having the target as a stationary law?'), action: t('균등·편향 합0 지지집합과 같은 장의 전체 공간을 비교하세요.', 'Compare uniform and biased zero-sum support with full support under the same fields.'), explain: t('합0에서는 한 자리 변경이 금지되어 항등 kernel입니다. joint block은 두 좌표를 함께 뽑아 목표에 도달합니다.', 'Zero sum forbids single-coordinate changes, yielding identity kernels. A joint block resamples both coordinates and reaches the target.'), presetIndices: [3, 4, 5] },
    { title: t('작은 값과 수렴 증거', 'Small values and convergence evidence'), predict: t('확률이 같아 보이거나 전이가 안 보이면 정확히 같은 법칙일까요?', 'Do identical-looking probabilities or invisible transitions imply exact equality?'), action: t('미약한 결합의 수축·희귀 전이의 부호·음의 온도의 흐름을 보세요.', 'Inspect weak-coupling contraction, rare-edge signs and negative-temperature flows.'), explain: t('정확한 지지 그래프와 보존한 구간으로 판정합니다. 표시상0이나 유한 seed 빈도를 수학적0으로 바꾸지 않습니다.', 'Use the exact support graph and retained bounds. Display zeros and finite seeded counts are not mathematical zeros.'), presetIndices: [6, 7, 8] },
  ],
  'quadratic-gaussian-gibbs-normalizability': [
    { title: t('그릇과 확률 타원', 'The bowl and its probability ellipse'), predict: t('source는 중심과 폭을 모두 바꿀까요?', 'Does a linear source change both center and width?'), action: t('상관 타원·선형 source·교차항 없는 준비를 비교하세요.', 'Compare the correlated ellipse, linear source and uncoupled preparation.'), explain: t('고정 A,T에서 source는 평균을 옮기고 공분산은 τA inverse입니다. 교차항은 타원의 기울기와 상관을 바꿉니다.', 'At fixed A,T the source moves the mean while covariance is tau times A inverse. Cross energy changes tilt and correlation.'), presetIndices: [0, 1, 2] },
    { title: t('정규화의 엄격한 경계', 'The strict normalization boundary'), predict: t('작은 양의 행렬식과0과 음수를 같은 cutoff로 처리해도 될까요?', 'May one cutoff merge a tiny positive determinant, zero and a negative value?'), action: t('한 이진 간격 양쪽의 행렬과 평평한 경계를 비교하세요.', 'Compare matrices one binary spacing from either side of the flat boundary.'), explain: t('A11>0,det A>0을 정확히 검사합니다. 유사역행렬이나 유한 그림의 합1은 전체 평면 정규화를 대신하지 못합니다.', 'Test A11>0 and det A>0 exactly. A pseudoinverse or unit mass inside a finite plot cannot establish full-plane normalization.'), presetIndices: [3, 4, 5] },
    { title: t('밀도·기준 부피·저온', 'Density, reference volume and low temperature'), predict: t('밀도>1이나 작은 공분산은 잘못된 확률일까요?', 'Do density above one or tiny covariances make a probability law invalid?'), action: t('큰 밀도 높이·방향 반전 좌표·아주 작은 양의 온도를 확인하세요.', 'Inspect high density, orientation-reversing coordinates and a very small positive temperature.'), explain: t('확률은 면적 적분이고 좌표변환에는 절댓값 Jacobian이 필요합니다. 고전 양의 온도의 C/kB=1을 유한 간격의 저온 극한과 혼동하지 않습니다.', 'Probability is an area integral and coordinates require an absolute Jacobian. Classical C/kB=1 at positive T differs from a finite-gap low-temperature limit.'), presetIndices: [6, 7, 8] },
  ],
};
export function boltzmannGibbsPresetInput(kind: BoltzmannGibbsKind, index: number): BoltzmannGibbsInput {
  if (!isBoltzmannGibbsKind(kind) || !Number.isInteger(index) || index < 0 || !boltzmannGibbsPresets[kind][index]) throw new RangeError('Invalid Boltzmann/Gibbs preset');
  return structuredClone(boltzmannGibbsPresets[kind][index].input);
}
