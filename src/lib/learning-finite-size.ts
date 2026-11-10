/** Bilingual teaching preparations; registration follows actual reader checks. */
import { finiteSizeKinds as kinds, isFiniteSizeInput, snapshotFiniteSizeInput, type FiniteSizeInput, type FiniteSizeKind } from './learning-finite-size-protocol';
export { finiteSizeKinds } from './learning-finite-size-protocol';
export type { FiniteSizeInput, FiniteSizeKind };
export interface FiniteSizeText { ko: string; en: string }
export const finiteSizeText = (ko: string, en: string): FiniteSizeText => ({ ko, en });
const t = finiteSizeText;
export const isFiniteSizeKind = (value: string): value is FiniteSizeKind => (kinds as readonly string[]).includes(value);
export interface FiniteSizePreset { id: string; label: FiniteSizeText; input: FiniteSizeInput }
export interface FiniteSizeGuide { title: FiniteSizeText; predict: FiniteSizeText; action: FiniteSizeText; explain: FiniteSizeText; presetIndices: readonly number[] }
export const finiteSizeTitles: Record<FiniteSizeKind, FiniteSizeText> = {
 [kinds[0]]: t('경계 성장·광역성·가법성', 'Boundary growth, extensivity and additivity'),
 [kinds[1]]: t('공간 밀도·상관·자기평균화', 'Spatial density, correlations and self-averaging'),
 [kinds[2]]: t('두 상의 유한 반올림·복소 영점', 'Two-phase finite rounding and complex zeros'),
};
type Boundary = Extract<FiniteSizeInput, { kind: typeof kinds[0] }>['parameters'];
type Spatial = Extract<FiniteSizeInput, { kind: typeof kinds[1] }>['parameters'];
type TwoPhase = Extract<FiniteSizeInput, { kind: typeof kinds[2] }>['parameters'];
const boundary = (p: Partial<Boundary> = {}): FiniteSizeInput => ({ kind: kinds[0], parameters: { dimension: 2, lengthOne: 2, lengthTwo: 2, width: 3, growthMode: 'uniform-dilation', scale: 1, spinOne: 1, spinTwo: -1, couplingNumerator: 1, couplingDenominator: 1, ...p } });
const spatial = (p: Partial<Spatial> = {}): FiniteSizeInput => ({ kind: kinds[1], parameters: { siteCount: 4, occupiedWeight: 1, emptyWeight: 1, commonNumerator: 1, commonDenominator: 2, commonScaling: 'fixed', epsilonNumerator: 1, epsilonDenominator: 2, ...p } });
const two = (p: Partial<TwoPhase> = {}): FiniteSizeInput => ({ kind: kinds[2], parameters: { siteCount: 4, activityNumerator: 2, activityDenominator: 1, sizePath: 'fixed', approachSide: 1, offsetNumerator: 1, offsetDenominator: 2, ...p } });
const preset = (id: string, ko: string, en: string, input: FiniteSizeInput): FiniteSizePreset => ({ id, label: t(ko, en), input: snapshotFiniteSizeInput(input) });
const rare: Partial<Spatial> = { siteCount: 512, occupiedWeight: 1, emptyWeight: 999999, commonNumerator: 1, commonDenominator: 1000000, commonScaling: 'inverse-size-squared', epsilonNumerator: 999999, epsilonDenominator: 1000000 };
export const finiteSizePresets: Record<FiniteSizeKind, readonly FiniteSizePreset[]> = {
 [kinds[0]]: [
  preset('boundary-base', '두 블록·반대 스핀', 'Two blocks, opposite spins', boundary()),
  preset('boundary-uniform-large', '모든 방향 네 배', 'Fourfold growth in every direction', boundary({ scale: 4 })),
  preset('boundary-slab', '두께 고정·시작 크기', 'Fixed thickness, initial size', boundary({ growthMode: 'fixed-thickness' })),
  preset('boundary-slab-large', '두께 고정·가로 네 배', 'Fixed thickness, fourfold transverse growth', boundary({ growthMode: 'fixed-thickness', scale: 4 })),
  preset('boundary-parallel', '평행 스핀·별도 정규화 비교', 'Parallel spins, separate normalization', boundary({ spinTwo: 1 })),
  preset('boundary-parallel-large', '평행 스핀·큰 계의 교차항', 'Parallel spins, cross term in a larger system', boundary({ spinTwo: 1, scale: 4 })),
  preset('boundary-unbalanced', '불균형 블록·작은 양의 에너지', 'Unbalanced blocks, small positive energy', boundary({ lengthTwo: 3 })),
  preset('boundary-unbalanced-large', '불균형 확대·에너지 부호 변경', 'Unbalanced growth, changed energy sign', boundary({ lengthTwo: 3, scale: 2 })),
  preset('boundary-two-sites', '한 차원·두 자리', 'One dimension, two sites', boundary({ dimension: 1, lengthOne: 1, lengthTwo: 1, width: 1 })),
  preset('boundary-zero', '결합0·두께 고정 기하 유지', 'Zero coupling, unchanged slab geometry', boundary({ couplingNumerator: 0, growthMode: 'fixed-thickness', scale: 4 })),
  preset('boundary-huge', '큰 정수 자리 수와 작은 결합', 'Huge exact site counts, small coupling', boundary({ dimension: 3, lengthOne: 1000000, lengthTwo: 1000000, width: 1000000, scale: 1000000, couplingDenominator: 1000000 })),
  preset('boundary-three-dimensions', '세 차원의 균일 확대', 'Uniform growth in three dimensions', boundary({ dimension: 3, scale: 4 })),
 ],
 [kinds[1]]: [
  preset('spatial-base', '네 자리·절반의 공통모드', 'Four sites, half collective mode', spatial()),
  preset('spatial-iid', '같은 주변분포·독립 준비', 'Same marginals, independent preparation', spatial({ commonNumerator: 0 })),
  preset('spatial-collective', '모든 자리 한 값 복사', 'One value copied to every site', spatial({ commonNumerator: 1, commonDenominator: 1 })),
  preset('spatial-inverse', '공통 가중치가 역크기로 감소', 'Collective weight decays as inverse size', spatial({ commonScaling: 'inverse-size' })),
  preset('spatial-inverse-square', '공통 가중치가 역제곱으로 감소', 'Collective weight decays as inverse square', spatial({ commonScaling: 'inverse-size-squared' })),
  preset('spatial-fixed-large', '고정 공통 가중치·64자리', 'Fixed collective weight, 64 sites', spatial({ siteCount: 64 })),
  preset('spatial-inverse-large', '역크기 경로·64자리', 'Inverse-size path, 64 sites', spatial({ siteCount: 64, commonScaling: 'inverse-size' })),
  preset('spatial-square-large', '역제곱 경로·64자리', 'Inverse-square path, 64 sites', spatial({ siteCount: 64, commonScaling: 'inverse-size-squared' })),
  preset('spatial-smaller-tail', '분산 증가·특정 꼬리 감소', 'Larger variance, smaller selected tail', spatial({ siteCount: 2, occupiedWeight: 1, emptyWeight: 9, epsilonNumerator: 3, epsilonDenominator: 20 })),
  preset('spatial-inclusive', '편차 임계값의 등호 포함', 'Include equality at the deviation threshold', spatial({ siteCount: 3, occupiedWeight: 1, emptyWeight: 2, epsilonNumerator: 1, epsilonDenominator: 3 })),
  preset('spatial-above', '등호보다 조금 큰 임계값', 'Threshold just above equality', spatial({ siteCount: 3, occupiedWeight: 1, emptyWeight: 2, epsilonNumerator: 333334, epsilonDenominator: 1000000 })),
  preset('spatial-deterministic', '확정 비점유·상관계수 미정의', 'Deterministic emptiness, undefined correlation', spatial({ occupiedWeight: 0 })),
  preset('spatial-singleton', '한 자리·서로 다른 쌍 없음', 'One site, no distinct pair', spatial({ siteCount: 1 })),
  preset('spatial-empty-event', '격자에서 불가능한 편차 사건', 'Deviation event impossible on the grid', spatial({ epsilonNumerator: 2, epsilonDenominator: 1 })),
  preset('spatial-rare', '같은 선도 분산·다른 극소 꼬리', 'Same leading variance, different rare tail', spatial(rare)),
  preset('spatial-rare-iid', '극소 꼬리의 독립 참조', 'Independent reference for the rare tail', spatial({ ...rare, commonNumerator: 0 })),
 ],
 [kinds[2]]: [
  preset('rounding-up', '네 자리·활동도2', 'Four sites, activity two', two()),
  preset('rounding-down', '역수 활동도·확률 교환', 'Reciprocal activity, exchanged probabilities', two({ activityNumerator: 1, activityDenominator: 2 })),
  preset('rounding-coexist', '공존점·네 자리', 'Coexistence, four sites', two({ activityNumerator: 1 })),
  preset('rounding-coexist-large', '공존점·64자리', 'Coexistence, 64 sites', two({ activityNumerator: 1, siteCount: 64 })),
  preset('rounding-inverse-up', '양의 역크기 접근', 'Positive inverse-size approach', two({ sizePath: 'inverse-size' })),
  preset('rounding-inverse-down', '음의 역크기 접근', 'Negative inverse-size approach', two({ sizePath: 'inverse-size', approachSide: -1 })),
  preset('rounding-square-up', '양의 역제곱 접근', 'Positive inverse-square approach', two({ sizePath: 'inverse-size-squared' })),
  preset('rounding-square-down', '음의 역제곱 접근', 'Negative inverse-square approach', two({ sizePath: 'inverse-size-squared', approachSide: -1 })),
  preset('rounding-inverse-large', '역크기 접근·512자리', 'Inverse-size approach, 512 sites', two({ sizePath: 'inverse-size', siteCount: 512 })),
  preset('rounding-square-large', '역제곱 접근·512자리', 'Inverse-square approach, 512 sites', two({ sizePath: 'inverse-size-squared', siteCount: 512 })),
  preset('rounding-odd-roots', '홀수 차수·음의 실수 근', 'Odd degree, negative real root', two({ siteCount: 3 })),
  preset('rounding-even-roots', '짝수 차수·단위원 근', 'Even degree, unit-circle roots', two({ siteCount: 8 })),
  preset('rounding-tiny-log', '큰 활동도·극소 양의 압력 보정', 'Large activity, tiny positive pressure correction', two({ siteCount: 512, activityNumerator: 1000000 })),
  preset('rounding-tiny-reciprocal', '작은 활동도·같은 압력 보정', 'Small activity, same pressure correction', two({ siteCount: 512, activityNumerator: 1, activityDenominator: 1000000 })),
  preset('rounding-zero-offset', '접근 가중치0·정확한 공존점', 'Zero approach offset, exact coexistence', two({ sizePath: 'inverse-size', offsetNumerator: 0, siteCount: 512 })),
  preset('rounding-small-offset', '작은 접근 가중치·양의 장 유지', 'Small approach offset, retained positive field', two({ sizePath: 'inverse-size', offsetDenominator: 1000000, siteCount: 512 })),
 ],
};
export function finiteSizePresetInput(kind: FiniteSizeKind, index: number): FiniteSizeInput {
 const value = finiteSizePresets[kind]?.[index];
 if (!Number.isInteger(index) || !value || !isFiniteSizeInput(value.input)) throw new RangeError('Unknown finite-size preset');
 return snapshotFiniteSizeInput(value.input);
}
/** Nonnegative integer drafts only; signed spins and sides use explicit choices. */
export function parseFiniteSizeInteger(draft: string): number | null {
 const text = draft.trim(); if (!/^[+]?\d+$/.test(text)) return null;
 const value = Number(text); return Number.isSafeInteger(value) && value <= 1000000 ? value + 0 : null;
}
const guide = (title: FiniteSizeText, predict: FiniteSizeText, action: FiniteSizeText, explain: FiniteSizeText, presetIndices: number[]): FiniteSizeGuide => ({ title, predict, action, explain, presetIndices });
export const finiteSizeGuides: Record<FiniteSizeKind, readonly FiniteSizeGuide[]> = {
 [kinds[0]]: [
  guide(t('어느 방향을 키웠나요?', 'Which directions grew?'), t('자리 수가 커지면 경계 비율도 반드시 사라질까요?', 'Must a boundary ratio vanish when the site count grows?'), t('균일 확대와 두께 고정의 시작·큰 크기, 세 차원 준비를 비교하세요.', 'Compare initial and larger sizes for uniform and fixed-thickness growth, then the three-dimensional case.'), t('균일 확대에서는 접합/자리 비가 줄지만 두께 고정에서는1/(a+b)가 남습니다. 자리 수 증가와 모든 방향의 증가는 다릅니다.', 'Uniform growth reduces interface per site; fixed thickness retains 1/(a+b). More sites and growth in every direction are different conditions.'), [0, 1, 2, 3, 11]),
  guide(t('광역성과 절단을 구별', 'Separate extensivity from cutting'), t('전체 결합을 N으로 나누면 두 블록이 가법적일까요?', 'Does dividing the coupling by N make two blocks additive?'), t('세 상호작용 그래프와 평행 블록의 별도 정규화, 두 자리·결합0을 비교하세요.', 'Compare the three interaction graphs, separate normalization for parallel blocks, two sites and zero coupling.'), t('Kac 최악 에너지 크기는 선형이어도 교차항/자리는 남습니다. 별도 정규화는 Hamiltonian을 바꾸므로 보정항을 유지합니다.', 'The Kac energy norm is linear while its cross term per site remains. Separate normalization changes the Hamiltonian and requires the counterterm.'), [0, 4, 5, 8, 9]),
  guide(t('취소와 큰 정수', 'Cancellation and huge integers'), t('선택 배치의 작은 에너지가 전체 Hamiltonian의 선형 성장을 증명할까요?', 'Does a small energy of one configuration prove linear Hamiltonian growth?'), t('불균형 블록의 크기별 부호와 큰 정수 자리 수의 작은 보정항을 살펴보세요.', 'Inspect size-dependent signs for unbalanced blocks and small corrections at huge exact site counts.'), t('선택 배치의 취소와 최악 에너지 크기를 분리합니다. 큰 정수는 문자열로 유지하며 표현 가능한 작은 보정을 underflow로 부르지 않습니다.', 'Separate cancellation in a selected configuration from the worst energy scale. Keep huge counts as strings and do not call representable small corrections underflow.'), [6, 7, 10]),
 ],
 [kinds[1]]: [
  guide(t('같은 주변분포·다른 요동', 'Same marginals, different fluctuations'), t('한 자리 확률이 같으면 평균 밀도 분산도 같을까요?', 'Do equal one-site probabilities imply equal density variance?'), t('독립·절반 공통·완전 공통과 한 자리·확정 준비를 비교하세요.', 'Compare independent, half-collective and fully collective laws, then singleton and deterministic cases.'), t('서로 다른 쌍의 공분산이 N(N−1)번 들어갑니다. 한 자리의 쌍 부재와 확정 준비의 상관계수 미정의를0과 구별합니다.', 'Distinct-pair covariance contributes N(N−1) times. Distinguish absent singleton pairs and undefined deterministic correlation from zero.'), [0, 1, 2, 11, 12]),
  guide(t('자기평균화의 경로', 'Paths to self-averaging'), t('공통 가중치가 감소하면 선도 분산 계수도 항상 독립과 같을까요?', 'Does decreasing collective weight always recover the IID leading variance coefficient?'), t('고정·역크기·역제곱 가중치를4자리와64자리에서 비교하세요.', 'Compare fixed, inverse-size and inverse-square weights at four and 64 sites.'), t('고정 양수 가중치는 밀도 분산을 남기고 역크기는 v(1+c), 역제곱은 v의 N배 분산 극한을 줍니다. 크기 축은 시간이 아닙니다.', 'Fixed positive weight retains density variance; inverse size gives limiting scaled variance v(1+c), inverse square gives v. The size axis is not time.'), [0, 3, 4, 5, 6, 7]),
  guide(t('꼬리·상한·경계', 'Tails, bounds and boundaries'), t('더 큰 분산이나 같은 선도 분산만으로 꼬리를 비교할 수 있을까요?', 'Can larger variance or equal leading variance alone compare tails?'), t('분산 증가/꼬리 감소, 임계값 등호, 빈 사건과 극소 꼬리의 독립 참조를 비교하세요.', 'Compare increased variance with decreased tail, threshold equality, an empty event and rare-tail IID references.'), t('정확 사건 확률과 상한을 분리하고 독립 참조를 혼합 상한으로 쓰지 않습니다. 선도 분산이 같아도 드문 공통 사건은 꼬리를 바꿉니다.', 'Separate exact event probability from bounds and do not reuse the IID reference as a mixture bound. Rare collective events can change tails despite matching leading variance.'), [8, 9, 10, 13, 14, 15]),
 ],
 [kinds[2]]: [
  guide(t('두 상태와 공존점', 'Two states and coexistence'), t('압력 오차가 줄면 밀도 요동도 사라질까요?', 'Must density fluctuations vanish when pressure error shrinks?'), t('활동도와 역수, 공존점4자리·64자리의 압력·요동·응답을 비교하세요.', 'Compare activity and its reciprocal, then pressure, fluctuation and response at coexistence for four and 64 sites.'), t('공존점의 밀도 분산은1/4이고 응답은 N/4지만 압력 보정은 ln2/N입니다. 두 허용 상태를 독립 자리 합으로 바꾸지 않습니다.', 'At coexistence the density variance is 1/4 and response N/4 while pressure correction is ln2/N. Do not replace the two allowed states by an independent-site sum.'), [0, 1, 2, 3]),
  guide(t('접근 경로와 극한 순서', 'Approach paths and order of limits'), t('양의 장으로 공존점에 접근하면 항상 점유가 선택될까요?', 'Does approaching coexistence through positive fields always select occupancy?'), t('역크기와 역제곱의 두 방향, 큰 크기,0과 작은 오프셋을 비교하세요.', 'Compare both sides of inverse-size and inverse-square paths, larger sizes, zero and small offsets.'), t('Ng의 극한이 ±c이면 밀도는 logistic(±c),0이면1/2로 갑니다. 입력 변화가 크기 경로인지 고정 N의 장 미분인지 구별합니다.', 'Limiting Ng=±c gives logistic(±c) density; zero gives one half. Distinguish changing a size path from differentiating the field at fixed N.'), [4, 5, 6, 7, 8, 9, 14, 15]),
  guide(t('복소 영점과 작은 로그', 'Complex zeros and tiny logarithms'), t('유한계의 복소 근이 양의 실수 축에 놓일까요?', 'Can a finite-size complex zero lie on the positive real axis?'), t('홀짝 차수의 회전수·근접 거리와 큰/작은 활동도의 압력 보정을 확인하세요.', 'Inspect root turns and distances for odd/even degree, then pressure corrections for large/small activity.'), t('근은 단위원에 있고 가장 가까운 각은 π/N입니다. 좌표는 그림용 근사이며 회전수가 정확 검산입니다. 표시상1+t=1이어도 작은 로그 보정은 양수입니다.', 'Roots lie on the unit circle with nearest angle pi/N. Coordinates are drawing approximations; exact turns verify the roots. A displayed 1+t=1 can coexist with a positive logarithmic correction.'), [10, 11, 12, 13]),
 ],
};
