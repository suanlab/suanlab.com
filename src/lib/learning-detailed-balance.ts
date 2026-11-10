/** Teaching preparations only; readers are registered after visual verification. */
import { detailedBalanceKinds as kinds, isDetailedBalanceInput, snapshotDetailedBalanceInput, type DetailedBalanceInput, type DetailedBalanceKind } from './learning-detailed-balance-protocol';
export { detailedBalanceKinds } from './learning-detailed-balance-protocol';
export type { DetailedBalanceInput, DetailedBalanceKind };
export interface DetailedBalanceText { ko: string; en: string }
export const detailedBalanceText = (ko: string, en: string): DetailedBalanceText => ({ ko, en });
const t = detailedBalanceText;
export const isDetailedBalanceKind = (value: string): value is DetailedBalanceKind => (kinds as readonly string[]).includes(value);
export interface DetailedBalancePreset { id: string; label: DetailedBalanceText; input: DetailedBalanceInput }
export interface DetailedBalanceGuide { title: DetailedBalanceText; predict: DetailedBalanceText; action: DetailedBalanceText; explain: DetailedBalanceText; presetIndices: readonly number[] }
export const detailedBalanceTitles: Record<DetailedBalanceKind, DetailedBalanceText> = {
 [kinds[0]]: t('정상분포 속의 순환 흐름', 'Circulating flow within a stationary distribution'),
 [kinds[1]]: t('제안 비대칭과 Hastings 보정', 'Proposal asymmetry and the Hastings correction'),
 [kinds[2]]: t('가역성·주기성과 혼합', 'Reversibility, periodicity and mixing'),
};
const cycle = (f: number, b: number, s: number, initial: readonly number[] = [1, 0, 0], n = 3): DetailedBalanceInput => ({ kind: kinds[0], parameters: { forwardWeight: f, backwardWeight: b, stayWeight: s, initialWeights: initial, stepCount: n } });
const hastings = (target: readonly number[], f = 2, b = 1, s = 1): DetailedBalanceInput => ({ kind: kinds[1], parameters: { targetWeights: target, forwardWeight: f, backwardWeight: b, stayWeight: s } });
const two = (a: number, b: number, d: number, u = 0, v = 1, n = 8): DetailedBalanceInput => ({ kind: kinds[2], parameters: { forwardNumerator: a, backwardNumerator: b, transitionDenominator: d, initialOneNumerator: u, initialDenominator: v, stepCount: n } });
const p = (id: string, ko: string, en: string, input: DetailedBalanceInput): DetailedBalancePreset => ({ id, label: t(ko, en), input: snapshotDetailedBalanceInput(input) });
export const detailedBalancePresets: Record<DetailedBalanceKind, readonly DetailedBalancePreset[]> = {
 [kinds[0]]: [
  p('cycle-driven', '비대칭 순환·집중 초기값', 'Biased circulation, concentrated start', cycle(2, 1, 1)),
  p('cycle-stationary', '균일 초기값·전류 유지', 'Uniform start, persistent current', cycle(2, 1, 1, [1, 1, 1])),
  p('cycle-reversible', '양방향 대칭·무체류', 'Symmetric moves, no holding', cycle(1, 1, 0)),
  p('cycle-clockwise', '한 방향·주기3', 'One direction, period three', cycle(1, 0, 0)),
  p('cycle-clockwise-held', '한 방향·체류 추가', 'One direction with holding', cycle(1, 0, 1)),
  p('cycle-identity', '항등·비균일 정상 초기값', 'Identity, nonuniform stationary start', cycle(0, 0, 1, [1, 2, 3])),
  p('cycle-identity-uniform', '항등·균일 참조', 'Identity, uniform reference', cycle(0, 0, 1, [1, 1, 1])),
  p('cycle-reversed', '반대 방향의 정상전류', 'Reversed stationary current', cycle(1, 2, 1)),
  p('cycle-bound', '수축 상한과 실제 거리', 'Contraction bound and actual distance', cycle(7, 2, 5, [1, 0, 0], 1)),
  p('cycle-tiny', '표시 아래 남은 비영 편차', 'Nonzero deviation below display range', cycle(1000000, 999999, 1000000, [1, 0, 0], 64)),
  p('cycle-initial', '전이 전·t=0', 'Before transition, t=0', cycle(2, 1, 1, [1, 0, 0], 0)),
  p('cycle-one-step', '첫 전이·t=1', 'First transition, t=1', cycle(2, 1, 1, [1, 0, 0], 1)),
 ],
 [kinds[1]]: [
  p('hastings-asymmetric', '비대칭 제안·목표1:2:4', 'Asymmetric proposal, target 1:2:4', hastings([1, 2, 4])),
  p('hastings-symmetric', '대칭 제안·같은 목표', 'Symmetric proposal, same target', hastings([1, 2, 4], 1, 1, 1)),
  p('hastings-accidental', '우연한 목표 보존·상세평형 실패', 'Accidental invariance, failed detailed balance', hastings([1, 1, 4])),
  p('hastings-uniform', '균일 목표·비대칭 제안', 'Uniform target, asymmetric proposal', hastings([1, 1, 1])),
  p('hastings-one-way', '역제안 없음·보정 후 항등', 'Missing reverse proposal, corrected identity', hastings([1, 2, 4], 1, 0, 0)),
  p('hastings-one-way-uniform', '한 방향·균일 목표', 'One direction, uniform target', hastings([1, 1, 1], 1, 0, 0)),
  p('hastings-self-only', '자기 제안만·이동0', 'Self proposals only, zero movement', hastings([1, 2, 4], 0, 0, 1)),
  p('hastings-high-holding', '높은 수락률·낮은 이동률', 'High acceptance, low movement', hastings([1, 2, 4], 2, 1, 1000000)),
  p('hastings-no-self', '자기 제안 없음·거부 체류', 'No self proposal, rejection holding', hastings([1, 2, 4], 2, 1, 0)),
  p('hastings-small-drift', '여섯 자리 표시가 숨긴 편류', 'Drift hidden by six-decimal display', hastings([1000000, 999999, 1000000], 1000000, 999999, 1000000)),
 ],
 [kinds[2]]: [
  p('two-positive', '양의 고유값·단조 접근', 'Positive eigenvalue, monotone approach', two(1, 2, 4)),
  p('two-negative', '음의 고유값·교대 접근', 'Negative eigenvalue, alternating approach', two(3, 3, 4)),
  p('two-flip-even', '완전 교대·짝수 종료', 'Exact alternation, even final step', two(1, 1, 1, 0, 1, 2)),
  p('two-flip-odd', '완전 교대·홀수 종료', 'Exact alternation, odd final step', two(1, 1, 1, 0, 1, 3)),
  p('two-flip-stationary', '교대 커널·정상 초기값', 'Alternating kernel, stationary start', two(1, 1, 1, 1, 2)),
  p('two-identity', '항등·자기 초기값에 머묾', 'Identity, retains its initial law', two(0, 0, 1, 1, 3)),
  p('two-absorbing', '흡수·가약이어도 유일 극한', 'Absorption, unique limit despite reducibility', two(0, 1, 2, 1, 1)),
  p('two-reset', '한 번에 정상분포', 'Stationary after one transition', two(1, 3, 4)),
  p('two-quarter', '거리1/4의 포함 경계', 'Inclusive quarter-distance threshold', two(1, 1, 4, 0, 1, 1)),
  p('two-slow', '기록 내 미도달·극한 존재', 'Unreached in record, limit exists', two(1, 1, 1000000, 0, 1, 512)),
  p('two-near-flip', '거의 교대·느린 혼합', 'Near alternation, slow mixing', two(999999, 999999, 1000000, 0, 1, 512)),
  p('two-tiny-positive', '작은 양의 고유값·비영 오차', 'Tiny positive eigenvalue, nonzero error', two(499999, 500000, 1000000, 0, 1, 64)),
  p('two-tiny-negative', '작은 음의 고유값·비영 오차', 'Tiny negative eigenvalue, nonzero error', two(500001, 500000, 1000000, 0, 1, 64)),
  p('two-stationary-zero', '정상 초기값·최악 거리와 구별', 'Stationary start versus worst-case distance', two(1, 1, 4, 1, 2, 0)),
 ],
};
export function detailedBalancePresetInput(kind: DetailedBalanceKind, index: number): DetailedBalanceInput {
 const preset = detailedBalancePresets[kind]?.[index];
 if (!Number.isInteger(index) || !preset || !isDetailedBalanceInput(preset.input)) throw new RangeError('Unknown detailed-balance preset');
 return snapshotDetailedBalanceInput(preset.input);
}
/** Parse a nonnegative integer draft without converting tiny nonzero text to zero. */
export function parseDetailedBalanceInteger(draft: string): number | null {
 const value = draft.trim();
 if (!/^[+]?\d+$/.test(value)) return null;
 const number = Number(value);
 return Number.isSafeInteger(number) && number <= 1000000 ? number + 0 : null;
}
const g = (title: DetailedBalanceText, predict: DetailedBalanceText, action: DetailedBalanceText, explain: DetailedBalanceText, presetIndices: number[]): DetailedBalanceGuide => ({ title, predict, action, explain, presetIndices });
export const detailedBalanceGuides: Record<DetailedBalanceKind, readonly DetailedBalanceGuide[]> = {
 [kinds[0]]: [
  g(t('멈춘 확률과 움직이는 질량', 'Fixed probabilities, moving mass'), t('정상분포이면 모든 순전류가0일까요?', 'Must every net current vanish at stationarity?'), t('집중 초기값과 균일 초기값에서 정상 흐름·현재 흐름을 비교하세요.', 'Compare reference and current flows for concentrated and uniform starts.'), t('정상성은 정점별 유입과 유출의 합이 같은 조건입니다. 상세평형은 각 쌍의 흐름까지 같도록 요구합니다. 균일 초기값에서도 순환전류1/12가 남습니다.', 'Stationarity balances total inflow and outflow at each vertex. Detailed balance also balances every pair. A uniform start still carries circulating current 1/12.'), [0, 1, 7]),
  g(t('주기와 정상 참조', 'Period and stationary reference'), t('방향이 하나이면 반드시 수렴하지 않을까요?', 'Does one-direction movement always prevent convergence?'), t('무체류·체류·대칭·항등 준비의 분류를 비교하세요.', 'Compare no holding, holding, symmetric movement and identity.'), t('한 방향 무체류는 주기3이고 체류를 더하면 주기1입니다. 대칭 무체류 삼각형도 길이2와3의 귀환 때문에 주기1입니다. 항등에서는 모든 분포가 정상이며 균일 참조가 유일하지 않습니다.', 'A directed cycle without holding has period three; holding gives period one. A symmetric no-holding triangle also has period one through returns of length two and three. Every law is stationary under identity, so the uniform reference is not unique.'), [2, 3, 4, 5, 6]),
  g(t('경계·상한·작은 편차', 'Time boundary, bound and tiny deviation'), t('반올림 확률이 균일하면 정확히 정상일까요?', 'Do rounded uniform probabilities establish exact stationarity?'), t('t=0과 첫 전이, 수축 상한, 작은 비영 편차를 비교하세요. 시간을 바꾸면 해당 시점의 흐름을 다시 계산하세요.', 'Compare t=0 and the first transition, a contraction bound and tiny nonzero deviations. Recompute flow when changing the requested time.'), t('상한은 실제 거리보다 클 수 있습니다. 정확한 유리수 편차는 선형 표시 범위 아래에서도0이 아닙니다. 과거 확률과 마지막 시점의 흐름을 섞지 않습니다.', 'The bound may exceed actual distance. An exact rational deviation below linear display range remains nonzero. Never combine past probabilities with final-step flow.'), [8, 9, 10, 11]),
 ],
 [kinds[1]]: [
  g(t('목표·제안·보정', 'Target, proposal and correction'), t('목표 비율만 사용하면 항상 목표가 보존될까요?', 'Does using only the target ratio always preserve the target?'), t('비대칭·대칭·우연한 보존·균일 목표의 두 커널을 비교하세요.', 'Compare both kernels for asymmetric, symmetric, accidentally invariant and uniform targets.'), t('Hastings 비율은 역제안까지 포함하여 쌍별 흐름을 맞춥니다. 보정 전에도 특정 목표가 우연히 보존될 수 있으므로 정상성의 실패와 상세평형의 실패를 구별합니다.', 'The Hastings ratio includes reverse proposals to balance pair flows. An uncorrected kernel may accidentally preserve a particular target; distinguish failed invariance from failed detailed balance.'), [0, 1, 2, 3]),
  g(t('없는 제안과 거부 체류', 'Missing proposals and rejection holding'), t('거부를 버리고 이동 확률을 다시 정규화해도 같은 법칙일까요?', 'Does discarding rejections and renormalizing moves preserve the same law?'), t('한 방향 제안·자기 제안·거부 체류와 점프 법칙을 확인하세요.', 'Inspect one-way proposals, self proposals, rejection holding and the jump law.'), t('없는 제안의 조건부 수락은 미제공입니다. 있는 제안의 역제안이 없으면 수락률은0입니다. 거부는 대각 전이에 쌓이며 이동만 남긴 점프 법칙은 상태별 이동률로 재가중됩니다.', 'Conditional acceptance is unavailable for absent proposals. An existing proposal without its reverse has zero acceptance. Rejections accumulate on the diagonal; retaining only moves reweights the jump law by state-specific movement rates.'), [0, 4, 5, 6, 8]),
  g(t('수락·이동·표시 정밀도', 'Acceptance, movement and display precision'), t('수락률이1에 가까우면 효율적으로 이동할까요?', 'Does acceptance near one establish effective movement?'), t('기본·높은 체류·자기 제안만·작은 편류를 비교하세요.', 'Compare baseline, high holding, self-only and small-drift preparations.'), t('자기 제안은 수락되어도 상태가 바뀌지 않습니다. 여섯 자리 표시가 숨긴 편류는 이중정밀도 underflow와 다릅니다. 이 실험은 고정 커널 분석이며 시간이나 표본 경로가 아닙니다.', 'Accepted self proposals do not change state. Drift hidden by six decimals differs from binary64 underflow. This experiment analyzes fixed kernels, not time evolution or sample paths.'), [0, 6, 7, 9]),
 ],
 [kinds[2]]: [
  g(t('부호 있는 오차와 거리', 'Signed error and distance'), t('음의 고유값이면 거리가 증가할까요?', 'Does a negative eigenvalue make distance grow?'), t('양·음 고유값과 한 단계 reset, 작은 비영 오차를 비교하세요.', 'Compare positive and negative eigenvalues, one-step reset and tiny nonzero errors.'), t('편차는 λ의 부호에 따라 교대할 수 있지만 TV는 절댓값을 사용합니다. λ=0이면 한 전이 후 정확히 정상입니다. 작은 비영 편차를0으로 바꾸거나 로그 축에서 정확한0을 양수로 바꾸지 않습니다.', 'Deviation may alternate with the sign of lambda; TV uses its absolute value. Lambda zero reaches stationarity after one transition. Do not replace tiny nonzero errors by zero or place exact zero at a positive log-axis value.'), [0, 1, 7, 11, 12]),
  g(t('가역성만으로 충분할까', 'Is reversibility sufficient?'), t('상세평형을 만족하면 모든 초기값에서 혼합할까요?', 'Does detailed balance imply mixing from every initial law?'), t('완전 교대·정상 교대·항등·흡수를 비교하세요.', 'Compare alternation, stationary alternation, identity and absorption.'), t('완전 교대는 가역이어도 최악 거리 기준으로 혼합하지 않습니다. 항등은 자기 초기값에 수렴하지만 유일한 정상 참조가 없습니다. 흡수 커널은 가약이어도 모든 초기값에서 유일한 정상분포로 수렴할 수 있습니다.', 'Exact alternation is reversible without worst-case mixing. Identity converges to its own initial law but has no unique stationary reference. An absorbing kernel can be reducible yet converge from every initial law to a unique stationary law.'), [2, 3, 4, 5, 6]),
  g(t('평균·기록·혼합 기준', 'Averages, record and mixing criterion'), t('기록 내에1/4에 도달하지 않으면 영원히 혼합하지 않을까요?', 'Does failing to reach one quarter in the record imply never mixing?'), t('교대의 짝홀 Cesaro 평균, 포함 경계, 느린 혼합과 정상 초기값을 비교하세요.', 'Compare even/odd Cesaro means, the inclusive threshold, slow mixing and a stationary start.'), t('Cesaro 평균은 t=0부터N까지 N+1개 분포의 평균입니다. 기록 내 미도달과 주기적 영구 미도달은 다릅니다. 혼합 기준은 선택 초기값의 거리가 아니라 최악 초기값의 거리≤1/4입니다.', 'The Cesaro mean averages N+1 laws from t=0 through N. Unreached within the record differs from permanent periodic failure. The mixing criterion is worst-initial distance at most one quarter, not distance for the selected start.'), [2, 3, 8, 9, 10, 13]),
 ],
};
