/** Lightweight teaching configuration; numerical models stay in the worker. */
import { isEnsembleInput, snapshotEnsembleInput, type EnsembleInput } from './learning-ensemble-protocol';
import type { BinaryObservation } from './learning-binary-products';
export type EnsembleKind = EnsembleInput['kind'];
export type EnsembleText = { ko: string; en: string };
const t = (ko: string, en: string): EnsembleText => ({ ko, en });
// SI preset scale. Tests compare it with the model constant without importing
// a calculator into this client configuration module.
const kB = 1.380649e-23;
const reservoir = (n: number, b: number, K: number): EnsembleInput => snapshotEnsembleInput({ kind: 'finite-binary-reservoir-canonical-comparison', parameters: { subsystemSites: n, reservoirSites: b, totalExcitations: K } });
const canonical = (n: number, energyGap: number, temperature: number): EnsembleInput => snapshotEnsembleInput({ kind: 'canonical-binary-energy-fluctuations', parameters: { sites: n, energyGap, temperature } });
const grand = (M: number, energyGap: number, chemicalPotential: number, temperature: number): EnsembleInput => snapshotEnsembleInput({ kind: 'grand-canonical-finite-occupancy-fluctuations', parameters: { sites: M, energyGap, chemicalPotential, temperature } });
export const isEnsembleKind = (v: string): v is EnsembleKind => ['finite-binary-reservoir-canonical-comparison', 'canonical-binary-energy-fluctuations', 'grand-canonical-finite-occupancy-fluctuations'].includes(v);
export const ensembleTitles: Record<EnsembleKind, EnsembleText> = {
  'finite-binary-reservoir-canonical-comparison': t('유한 저장고와 정준 비교', 'Finite reservoir and canonical comparison'),
  'canonical-binary-energy-fluctuations': t('정준 에너지 분포와 요동', 'Canonical energy distributions and fluctuations'),
  'grand-canonical-finite-occupancy-fluctuations': t('입자 교환과 화학 응답', 'Particle exchange and chemical response'),
};
export const ensemblePresets: Record<EnsembleKind, { label: EnsembleText; input: EnsembleInput }[]> = {
  'finite-binary-reservoir-canonical-comparison': [
    { label: t('두 자리와 작은 저장고', 'Two sites and a small reservoir'), input: reservoir(2, 2, 2) },
    { label: t('밀도 1/3 비교', 'Density 1/3 comparison'), input: reservoir(2, 4, 2) },
    { label: t('같은 밀도의 큰 저장고', 'Larger reservoir at the same density'), input: reservoir(2, 98, 50) },
    { label: t('저장고 없이 고정한 여기', 'Fixed excitation without a reservoir'), input: reservoir(2, 0, 1) },
    { label: t('빈 부분계', 'Empty subsystem'), input: reservoir(0, 4, 2) },
    { label: t('함께 커지는 두 블록', 'Two growing blocks'), input: reservoir(128, 384, 192) },
  ],
  'canonical-binary-energy-fluctuations': [
    { label: t('네 자리·t=1', 'Four sites, t=1'), input: canonical(4, kB, 1) },
    { label: t('손계산 t=ln2', 'Worked example t=ln 2'), input: canonical(2, kB * Math.log(2), 1) },
    { label: t('같은 준비·열여섯 자리', 'Same preparation, sixteen sites'), input: canonical(16, kB, 1) },
    { label: t('유계 인구역전', 'Bounded population inversion'), input: canonical(4, kB, -1) },
    { label: t('양의 쪽 온도 0 극한', 'Positive-side zero-temperature limit'), input: canonical(4, kB, 0) },
    { label: t('정확한 간격 0', 'Exact zero gap'), input: canonical(4, 0, 0) },
    { label: t('확률 밖에서 복원되는 에너지', 'Energy rescued beyond probability range'), input: canonical(2, 1e280, 1e280 / (kB * 800)) },
    { label: t('표시 아래의 양의 열용량', 'Positive capacity below the display range'), input: canonical(2, Number.MIN_VALUE, Number.MAX_VALUE) },
    { label: t('첫 로그 밖의 극저온 꼬리', 'Cold tail beyond the first logarithm'), input: canonical(4, 1, Number.MIN_VALUE) },
  ],
  'grand-canonical-finite-occupancy-fluctuations': [
    { label: t('균형 점유 μ=ε', 'Balanced occupancy, mu=epsilon'), input: grand(2, kB, kB, 1) },
    { label: t('희박한 손계산 준비', 'Dilute worked preparation'), input: grand(2, kB, kB * (1 - Math.log(2)), 1) },
    { label: t('반대 편향의 준비', 'Opposite occupancy bias'), input: grand(2, kB, kB * (1 + Math.log(2)), 1) },
    { label: t('온도 0·가득 찬 배치', 'Zero temperature, fully occupied'), input: grand(2, 1, 2, 0) },
    { label: t('온도 0·임계점', 'Zero-temperature threshold'), input: grand(2, 1, 1, 0) },
    { label: t('온도 0·빈 배치', 'Zero temperature, empty pattern'), input: grand(2, 1, 0, 0) },
    { label: t('표시 확률 밖의 화학 응답', 'Chemical response beyond probability range'), input: grand(2, 1e-297, 0, 1e-297 / (kB * 800)) },
    { label: t('유계 음의 온도 응답', 'Bounded negative-temperature response'), input: grand(4, kB, 0, -1) },
    { label: t('큰 에너지의 정확한 상쇄', 'Exact cancellation of large energies'), input: grand(4, Number.MAX_VALUE, Number.MAX_VALUE, Number.MIN_VALUE) },
  ],
};
export interface EnsembleGuide { title: EnsembleText; predict: EnsembleText; action: EnsembleText; explain: EnsembleText; preset: number }
export const ensembleGuides: Record<EnsembleKind, EnsembleGuide[]> = {
  'finite-binary-reservoir-canonical-comparison': [
    { title: t('같은 평균과 다른 퍼짐', 'Same mean, different spread'), predict: t('평균이 1이면 두 분포도 같을까요?', 'Does a mean of 1 make the two laws identical?'), action: t('작은 저장고의 두 막대그림·분산·TV를 비교하세요.', 'Compare both histograms, their variances and TV for the small reservoir.'), explain: t('P=(1,4,1)/6과 Q=(1,2,1)/4는 평균이 같아도 분산은 1/3과 1/2이며 TV=1/6입니다.', 'P=(1,4,1)/6 and Q=(1,2,1)/4 share a mean but have variances 1/3 and 1/2, with TV=1/6.'), preset: 0 },
    { title: t('저장고 크기와 준비 밀도', 'Reservoir size and preparation density'), predict: t('K를 그대로 두고 저장고만 키워도 같은 비교일까요?', 'Is growing the reservoir at fixed K the same comparison?'), action: t('큰 저장고와 첫 예제를 비교하고 K/N을 확인하세요.', 'Compare the larger reservoir with the first example and check K/N.'), explain: t('n=2와 밀도 1/2를 유지하려면 K도 바꿉니다. b=98에서는 평균 1,분산 49/99,TV=1/198입니다.', 'Keeping n=2 and density 1/2 requires changing K too. At b=98 the mean is 1, variance 49/99 and TV=1/198.'), preset: 2 },
    { title: t('비교 방향과 금지된 결과', 'Comparison direction and forbidden outcomes'), predict: t('정방향 KL이 유한하면 역방향도 유한할까요?', 'Does finite forward KL imply finite reverse KL?'), action: t('저장고 없는 고정 여기에서 P와 Q의 지지집합을 보세요.', 'Inspect the supports of P and Q for fixed excitation without a reservoir.'), explain: t('P는 A=1에 집중하지만 Q는 0과2도 허용합니다. D(P∥Q)=ln2이고 D(Q∥P)는 무한대입니다.', 'P is concentrated at A=1 while Q also allows 0 and 2. D(P||Q)=ln 2 but D(Q||P) is infinite.'), preset: 3 },
  ],
  'canonical-binary-energy-fluctuations': [
    { title: t('정규화와 응답 단위', 'Normalization and response units'), predict: t('여기 수 분산과 열용량은 같은 물리량일까요?', 'Are count variance and heat capacity the same physical quantity?'), action: t('t=ln2의 분포·분산·C/kB를 비교하세요.', 'Compare the distribution, variance and C/kB at t=ln 2.'), explain: t('분포는 (4,4,1)/9이고 Var(K)=4/9입니다. C/kB=(ln2)²·4/9이며 물리적 C는 J/K 단위입니다.', 'The law is (4,4,1)/9 with Var(K)=4/9. C/kB=(ln 2)^2 times 4/9, while physical C is measured in J/K.'), preset: 1 },
    { title: t('총량과 상대 요동', 'Extensive quantities and relative fluctuation'), predict: t('자리가 늘면 절대 분산도 작아질까요?', 'Does adding sites reduce the absolute variance?'), action: t('같은 t=1에서 네 자리와 열여섯 자리를 비교하세요.', 'Compare four and sixteen sites at the same t=1.'), explain: t('평균·분산·엔트로피·열용량은 네 배,표준편차는 두 배,상대 요동은 절반입니다.', 'Mean, variance, entropy and heat capacity grow fourfold; standard deviation doubles and relative fluctuation halves.'), preset: 2 },
    { title: t('확률 표시와 에너지 척도', 'Probability display and energy scale'), predict: t('확률 숫자가 표시되지 않으면 에너지 분산도0일까요?', 'If probability cannot be displayed, must energy variance be zero?'), action: t('드문 여기의 부호·로그와 U·Var(E)를 확인하세요.', 'Inspect the rare tail sign/log and U and Var(E).'), explain: t('유한한 t≈800의 확률은 양수입니다. ε=10^280 J의 척도와 결합하면 U와 Var(E)는 표현 가능하며 온도0 극한과 다릅니다.', 'At finite t about 800 the probability is positive. Combining it with epsilon=10^280 J yields representable U and Var(E), unlike the zero-temperature limit.'), preset: 6 },
  ],
  'grand-canonical-finite-occupancy-fluctuations': [
    { title: t('입자 수와 조건부 배치', 'Particle count and conditional patterns'), predict: t('입자 하나인 배치01의 무조건부와 조건부 확률은 같을까요?', 'Does pattern 01 have the same unconditional and conditional probability?'), action: t('균형 점유에서 N=1의 확률과 그 안의 배치를 비교하세요.', 'Compare the probability of N=1 with its within-count patterns at balance.'), explain: t('P(N=1)=1/2,배치01의 확률은1/4,조건부는1/2입니다. Var(N)=1/2이며 μ 응답은1/(2kBT)입니다.', 'P(N=1)=1/2, pattern 01 has probability 1/4, and its conditional probability is 1/2. Var(N)=1/2 and the chemical response is 1/(2kBT).'), preset: 0 },
    { title: t('온도0의 임계점과 두 극값', 'Zero-temperature threshold and two extrema'), predict: t('온도0이면 입자 수 분산이 항상0일까요?', 'Must particle-count variance vanish at zero temperature?'), action: t('임계점을 빈 배치·가득 찬 배치 예제와 비교하세요.', 'Compare the threshold with the empty and fully occupied examples.'), explain: t('고정 μ=ε 경로의 임계점에는 분산1/2와 엔트로피2kB ln2가 남고 유한한 μ 미분은 없습니다. 가득 찬 경우에는 logΞ가 발산해도 Ω=-2 J입니다.', 'On the fixed mu=epsilon threshold path, variance 1/2 and entropy 2kB ln 2 remain, with no finite chemical derivative. In the full case log Xi diverges but Omega=-2 J.'), preset: 4 },
    { title: t('작은 점유의 유한 응답', 'Finite response from tiny occupancy'), predict: t('점유 확률이 선형 범위 밖이면 화학 응답도 사라질까요?', 'Does an out-of-range occupancy probability eliminate chemical response?'), action: t('희박한 예제의 양의 확률과 1/J 단위 응답을 확인하세요.', 'Inspect the dilute positive probability and response measured in 1/J.'), explain: t('μ=0,ε=10^-297 J,t≈800에서 확률은 양수지만 선형 범위 밖입니다. 1/(kBT) 인자로 응답은 표현 가능합니다.', 'At mu=0, epsilon=10^-297 J and t about 800, probability is positive outside the linear range. The factor 1/(kBT) makes the response representable.'), preset: 6 },
  ],
};
export function ensemblePresetInput(kind: EnsembleKind, index: number): EnsembleInput {
  if (!isEnsembleKind(kind) || !Number.isInteger(index) || index < 0 || !ensemblePresets[kind][index]) throw new RangeError('Invalid ensemble preset');
  return snapshotEnsembleInput(ensemblePresets[kind][index].input);
}
export const validEnsembleSavedInput = (kind: EnsembleKind, value: unknown): value is EnsembleInput => isEnsembleInput(value) && value.kind === kind;
export const ensembleSavedValidators = {
  'finite-binary-reservoir-canonical-comparison': (v: unknown): v is EnsembleInput => validEnsembleSavedInput('finite-binary-reservoir-canonical-comparison', v),
  'canonical-binary-energy-fluctuations': (v: unknown): v is EnsembleInput => validEnsembleSavedInput('canonical-binary-energy-fluctuations', v),
  'grand-canonical-finite-occupancy-fluctuations': (v: unknown): v is EnsembleInput => validEnsembleSavedInput('grand-canonical-finite-occupancy-fluctuations', v),
};
export function ensemblePlotValue(value: BinaryObservation | null): number | null {
  return value?.value !== null && value?.value !== undefined && Number.isFinite(value.value) ? value.value : null;
}
