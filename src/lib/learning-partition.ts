/** Lightweight bilingual teaching configuration. No numerical runtime imports. */
import { isPartitionInput, snapshotPartitionInput, type PartitionInput } from './learning-partition-protocol';
export type PartitionKind = PartitionInput['kind'];
export type PartitionText = { ko: string; en: string };
const t = (ko: string, en: string): PartitionText => ({ ko, en });
const kB = 1.380649e-23, g = kB * Math.log(2);
const spectrum = (energyLevels: number[], energyOffset = 0, temperature = 1) => snapshotPartitionInput({ kind: 'finite-spectrum-partition-cumulant-generators', parameters: { energyLevels, energyOffset, temperature } });
const pair = (couplingEnergy: number, fieldEnergy = 0, temperature = 1, support: 'full-product' | 'fixed-zero-sum' = 'full-product') => snapshotPartitionInput({ kind: 'coupled-binary-sites-partition-factorization', parameters: { couplingEnergy, fieldEnergy, temperature, support } });
const ladder = (gapEnergy: number, temperature: number, cutoff: number) => snapshotPartitionInput({ kind: 'infinite-energy-ladder-normalization-cutoff', parameters: { gapEnergy, temperature, cutoff } });
export const partitionKinds = ['finite-spectrum-partition-cumulant-generators', 'coupled-binary-sites-partition-factorization', 'infinite-energy-ladder-normalization-cutoff'] as const;
export const isPartitionKind = (v: string): v is PartitionKind => (partitionKinds as readonly string[]).includes(v);
export const partitionTitles: Record<PartitionKind, PartitionText> = {
  'finite-spectrum-partition-cumulant-generators': t('분배함수와 에너지 cumulant', 'Partition functions and energy cumulants'),
  'coupled-binary-sites-partition-factorization': t('결합·제약과 곱 분해', 'Coupling, constraints and factorization'),
  'infinite-energy-ladder-normalization-cutoff': t('무한 사다리와 절단 오차', 'Infinite ladders and cutoff error'),
};
export const partitionPresets: Record<PartitionKind, { label: PartitionText; input: PartitionInput }[]> = {
  'finite-spectrum-partition-cumulant-generators': [
    { label: t('세 준위 손계산', 'Three-level worked example'), input: spectrum([0, g, 2 * g]) },
    { label: t('큰 공통 기준·같은 간격', 'Large common offset, same gaps'), input: spectrum([0, g, 2 * g], 2 ** 60) },
    { label: t('입력 전에 소실된 간격', 'Gaps lost before input'), input: spectrum([2 ** 60, 2 ** 60 + g, 2 ** 60 + 2 * g]) },
    { label: t('같은 준위·음의 온도', 'Same levels, negative temperature'), input: spectrum([0, g, 2 * g], 0, -1) },
    { label: t('온도0의 두 바닥상태', 'Two ground states at zero temperature'), input: spectrum([0, 0, g], 0, 0) },
    { label: t('모든 준위가 같은 에너지', 'All states at equal energy'), input: spectrum([0, 0, 0]) },
    { label: t('양의 온도·음의 세 번째 cumulant', 'Positive temperature, negative third cumulant'), input: spectrum([0, g, g, g]) },
    { label: t('드문 확률과 유한 에너지', 'Rare probability and finite energy'), input: spectrum([0, 1e280], 0, 1e280 / (kB * 800)) },
    { label: t('정밀도 상한의 미해결', 'Unresolved at the precision ceiling'), input: spectrum([0, Number.MIN_VALUE, Number.MAX_VALUE], 0, Number.MAX_VALUE) },
  ],
  'coupled-binary-sites-partition-factorization': [
    { label: t('편향 없는 상관 쌍', 'Correlated unbiased pair'), input: pair(g) },
    { label: t('결합 없는 전체 공간', 'Uncoupled full support'), input: pair(0) },
    { label: t('결합 없이 합0 제약', 'Zero-sum constraint without coupling'), input: pair(0, 0, 1, 'fixed-zero-sum') },
    { label: t('반정렬 선호', 'Preference for disagreement'), input: pair(-g) },
    { label: t('공통 외부장의 편향', 'Bias from a common field'), input: pair(g, .7 * kB) },
    { label: t('유계 음의 온도', 'Bounded negative temperature'), input: pair(g, .7 * kB, -1) },
    { label: t('온도0의 세 상태 교차', 'Three-state crossing at zero temperature'), input: pair(-1, 1, 0) },
    { label: t('공분산0·미분 부재', 'Zero covariance, absent derivative'), input: pair(0, 0, 0) },
    { label: t('표시 범위 아래의 의존성', 'Dependence below display range'), input: pair(1e-200) },
  ],
  'infinite-energy-ladder-normalization-cutoff': [
    { label: t('q=1/2·세 상태 절단', 'q=1/2, three-state cutoff'), input: ladder(g, 1, 2) },
    { label: t('누락 질량1% 이하', 'Missing mass below one percent'), input: ladder(g, 1, 6) },
    { label: t('높은 온도·작은 절단', 'High temperature, small cutoff'), input: ladder(kB * .001, 1, 2) },
    { label: t('같은 온도·큰 절단', 'Same temperature, larger cutoff'), input: ladder(kB * .001, 1, 4096) },
    { label: t('간격0·무한 정규화 불가', 'Zero gap, no infinite normalization'), input: ladder(0, 1, 8) },
    { label: t('음의 온도·유한 준비만', 'Negative temperature, finite law only'), input: ladder(g, -1, 8) },
    { label: t('온도0의 바닥 극한', 'Ground limit at zero temperature'), input: ladder(g, 0, 8) },
    { label: t('q의1 반올림과 유한 열용량', 'q rounds to one, capacity stays finite'), input: ladder(Number.MIN_VALUE, Number.MAX_VALUE, 2) },
    { label: t('극저온의 양의 꼬리', 'Positive cold tail'), input: ladder(kB * 5000, 1, 2) },
  ],
};
export interface PartitionGuide { title: PartitionText; predict: PartitionText; action: PartitionText; explain: PartitionText; preset: number }
export const partitionGuides: Record<PartitionKind, PartitionGuide[]> = {
  'finite-spectrum-partition-cumulant-generators': [
    { title: t('가중치에서 모멘트로', 'From weights to moments'), predict: t('확률이 (4,2,1)/7이면 에너지 분산도 같은 숫자일까요?', 'Does the law (4,2,1)/7 make energy variance the same number?'), action: t('확률과 U·Var(E)·세 번째 cumulant의 단위를 비교하세요.', 'Compare probabilities and the units of U, Var(E) and the third cumulant.'), explain: t('지수 평균4/7·분산26/49·세 번째 cumulant114/343에 ε·ε²·ε³이 각각 곱해집니다.', 'Index mean 4/7, variance 26/49 and third cumulant 114/343 carry factors epsilon, epsilon² and epsilon³.'), preset: 0 },
    { title: t('기준 이동과 소실된 입력', 'Offset shifts and lost input'), predict: t('큰 공통 에너지를 더하면 분포가 균등해질까요?', 'Does adding a large common energy make the law uniform?'), action: t('큰 공통 기준 예제와 입력 전에 소실된 간격 예제를 비교하세요.', 'Compare the large-offset example with gaps lost before input.'), explain: t('별도 기준은 간격과 확률을 보존하고 U·F만 이동시킵니다. 이미 평평하게 반올림된 입력의 간격은 복구할 수 없습니다.', 'A separate offset preserves gaps and probabilities, shifting U and F. Gaps already rounded out of an input list cannot be recovered.'), preset: 1 },
    { title: t('온도 부호와 바닥 극한', 'Temperature sign and ground limit'), predict: t('음의 온도가 온도0의 다른 표기일까요?', 'Is negative temperature another name for zero temperature?'), action: t('음의 온도, 두 바닥상태, 모든 에너지 동일 예제를 비교하세요.', 'Compare negative temperature, two ground states and equal energies.'), explain: t('음의 온도는 높은 에너지에 더 큰 가중치를 줍니다. T=0은 양의 쪽 바닥 극한이며, 퇴화하면 여러 바닥상태가 확률을 나눕니다.', 'Negative temperature favors higher energies. T=0 is the positive-side ground limit, sharing probability between degenerate ground states.'), preset: 3 },
  ],
  'coupled-binary-sites-partition-factorization': [
    { title: t('평균0과 상관', 'Zero means and correlation'), predict: t('두 평균이0이면 두 스핀은 독립일까요?', 'Are two spins independent when both means vanish?'), action: t('편향 없는 쌍과 반정렬 예제의 네 배치를 비교하세요.', 'Compare the four configurations for agreement and disagreement preferences.'), explain: t('첫 예제는 평균0이어도 공분산3/5입니다. Δ=ln(5/4)와 상호정보량 약0.19274476 nat는 다른 비교입니다.', 'The first example has zero means but covariance 3/5. Delta=ln(5/4) differs from mutual information about 0.19274476 nats.'), preset: 0 },
    { title: t('지원 제약과 독립성', 'Support constraints and independence'), predict: t('J=0이면 어떤 상태공간에서도 곱 분해할까요?', 'Does J=0 imply factorization on every support?'), action: t('결합 없는 전체 공간을 합0 제약과 비교하세요.', 'Compare uncoupled full support with the zero-sum constraint.'), explain: t('곱 분해에는 가법 에너지와 곱 상태공간이 함께 필요합니다. 합0에서는 두 반정렬 배치만 남아 공분산−1·정보량ln2입니다.', 'Factorization needs additive energy and product support. Zero sum leaves two opposite configurations, covariance -1 and information ln2.'), preset: 2 },
    { title: t('source 곡률과 물리적 응답', 'Source curvature and physical response'), predict: t('교차 공분산0이면 온도0의 미분도0일까요?', 'Does zero cross covariance imply a zero derivative at T=0?'), action: t('공분산0·미분 부재와 세 상태 교차 예제를 확인하세요.', 'Inspect zero covariance with an absent derivative and the three-state crossing.'), explain: t('J=h=0,T=0에서 Cov(A,M)=0이지만 비영 h가 A=1을 골라 미분이 없습니다. 유한 source 공분산과 물리적 미분은 다릅니다.', 'At J=h=0,T=0, Cov(A,M)=0, but nonzero h selects A=1, so the derivative is absent. Source covariance and physical derivatives differ.'), preset: 7 },
  ],
  'infinite-energy-ladder-normalization-cutoff': [
    { title: t('정규화와 누락 질량', 'Normalization and missing mass'), predict: t('세 확률의 합이1이면 무한 꼬리도0일까요?', 'Does a three-probability sum of one imply a zero infinite tail?'), action: t('유한·무한 막대와 누락 질량·평균 편향을 비교하세요.', 'Compare finite/infinite bars, missing mass and mean bias.'), explain: t('q=1/2,L=2에서 빠진 질량은1/8,평균 지수 편향은3/7입니다. 유한 목록은 남은7/8을 조건화합니다.', 'At q=1/2,L=2, missing mass is 1/8 and mean-index bias 3/7. The finite list conditions the retained 7/8.'), preset: 0 },
    { title: t('오차 기준으로 절단 선택', 'Choose a cutoff by an error criterion'), predict: t('L=2가 누락 질량1% 조건을 만족할까요?', 'Does L=2 meet a one-percent missing-mass target?'), action: t('L=6 예제의 누락 질량과 두 KL 방향을 보세요.', 'Inspect missing mass and both KL directions at L=6.'), explain: t('L=6이면1/128로1%보다 작습니다. 순방향 KL은 유한하지만 무한 분포의 양의 꼬리 때문에 역방향은 무한대입니다.', 'At L=6 the tail is 1/128, below one percent. Forward KL is finite; reverse KL is infinite because the target has a positive tail.'), preset: 1 },
    { title: t('온도와 절단의 두 척도', 'Temperature and cutoff scales'), predict: t('높은 온도에서도 고정된 작은 절단이 충분할까요?', 'Does a fixed small cutoff suffice at high temperature?'), action: t('같은 x=0.001에서 작은 절단과4096 절단을 비교하세요.', 'Compare the small cutoff and cutoff 4096 at the same x=0.001.'), explain: t('고정 L의 C/kB는 높은 온도에서0,무한 사다리는1로 갑니다. 유한 목록의 정규화만으로 무한계 정확도를 주장할 수 없습니다.', 'At high temperature C/kB tends to zero for fixed L and one for the infinite ladder. Finite-list normalization does not establish infinite-target accuracy.'), preset: 2 },
  ],
};
export function partitionPresetInput(kind: PartitionKind, index: number): PartitionInput {
  if (!isPartitionKind(kind) || !Number.isInteger(index) || index < 0 || !partitionPresets[kind][index]) throw new RangeError('Invalid partition preset');
  return snapshotPartitionInput(partitionPresets[kind][index].input);
}
export const validPartitionSavedInput = (kind: PartitionKind, value: unknown): value is PartitionInput => isPartitionInput(value) && value.kind === kind;
export const partitionSavedValidators: Record<PartitionKind, (value: unknown) => value is PartitionInput> = {
  'finite-spectrum-partition-cumulant-generators': (v: unknown): v is PartitionInput => validPartitionSavedInput('finite-spectrum-partition-cumulant-generators', v),
  'coupled-binary-sites-partition-factorization': (v: unknown): v is PartitionInput => validPartitionSavedInput('coupled-binary-sites-partition-factorization', v),
  'infinite-energy-ladder-normalization-cutoff': (v: unknown): v is PartitionInput => validPartitionSavedInput('infinite-energy-ladder-normalization-cutoff', v),
};
