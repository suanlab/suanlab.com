/** Exact-input editing and discrete clocks; no numerical model runs here. */
import { parseDetailedBalanceInteger, detailedBalanceText as t, type DetailedBalanceText } from './learning-detailed-balance';
import { isDetailedBalanceInput, snapshotDetailedBalanceInput as snapshot, type DetailedBalanceInput } from './learning-detailed-balance-protocol';
export interface DetailedBalanceField { id: string; parameter: string; index?: number; value: number; label: DetailedBalanceText }
export type DetailedBalanceTemporalInput = Exclude<DetailedBalanceInput, { kind: 'metropolis-hastings-proposal-balance' }>;
export const isDetailedBalanceTemporalInput = (input: DetailedBalanceInput): input is DetailedBalanceTemporalInput => input.kind !== 'metropolis-hastings-proposal-balance';
export function detailedBalanceFields(input: DetailedBalanceInput): DetailedBalanceField[] {
 const fields: DetailedBalanceField[] = [];
 const scalar = (parameter: string, value: number, ko: string, en: string) => fields.push({ id: parameter, parameter, value, label: t(ko, en) });
 if (input.kind === 'reversible-two-state-periodicity-mixing') {
  const p = input.parameters;
  scalar('forwardNumerator', p.forwardNumerator, '전이 a의 분자 A · 0≤A≤D', 'Transition a numerator A · 0≤A≤D');
  scalar('backwardNumerator', p.backwardNumerator, '전이 b의 분자 B · 0≤B≤D', 'Transition b numerator B · 0≤B≤D');
  scalar('transitionDenominator', p.transitionDenominator, '공통 전이 분모 D · 1–1,000,000', 'Transition denominator D · 1–1,000,000');
  scalar('initialOneNumerator', p.initialOneNumerator, '초기 상태1 확률 분자 U · 0≤U≤V', 'Initial state-one numerator U · 0≤U≤V');
  scalar('initialDenominator', p.initialDenominator, '초기 확률 분모 V · 1–1,000,000', 'Initial denominator V · 1–1,000,000');
 } else {
  const p = input.parameters;
  scalar('forwardWeight', p.forwardWeight, '정방향 가중치 f · 0–1,000,000', 'Forward weight f · 0–1,000,000');
  scalar('backwardWeight', p.backwardWeight, '역방향 가중치 b · 0–1,000,000', 'Backward weight b · 0–1,000,000');
  scalar('stayWeight', p.stayWeight, '체류 가중치 s · 0–1,000,000', 'Holding weight s · 0–1,000,000');
  const parameter = input.kind === 'stationary-flux-three-state-cycle' ? 'initialWeights' : 'targetWeights';
  const values = input.kind === 'stationary-flux-three-state-cycle' ? input.parameters.initialWeights : input.parameters.targetWeights;
  values.forEach((value, index) => fields.push({ id: `${parameter}.${index}`, parameter, index, value, label: parameter === 'initialWeights' ? t(`초기 상태${index} 가중치 · 0–1,000,000`, `Initial state ${index} weight · 0–1,000,000`) : t(`목표 상태${index} 가중치 · 1–1,000,000`, `Target state ${index} weight · 1–1,000,000`) }));
 }
 if (isDetailedBalanceTemporalInput(input)) scalar('stepCount', input.parameters.stepCount, '이산 전이 횟수 N · 정수0–512', 'Discrete transition count N · integer 0–512');
 return fields;
}
export function editDetailedBalanceDraft(input: DetailedBalanceInput, fieldId: string, draft: string): DetailedBalanceInput | null {
 if (!isDetailedBalanceInput(input)) return null;
 const field = detailedBalanceFields(input).find(field => field.id === fieldId), value = parseDetailedBalanceInteger(draft);
 if (!field || value === null) return null;
 const parameters: Record<string, unknown> = { ...input.parameters };
 if (field.index === undefined) parameters[field.parameter] = value;
 else { const values = [...parameters[field.parameter] as readonly number[]]; values[field.index] = value; parameters[field.parameter] = values; }
 const next = { kind: input.kind, parameters };
 return isDetailedBalanceInput(next) ? snapshot(next) : null;
}
export function advanceDetailedBalanceTime(input: DetailedBalanceInput): DetailedBalanceTemporalInput | null {
 if (!isDetailedBalanceInput(input) || !isDetailedBalanceTemporalInput(input) || input.parameters.stepCount >= 512) return null;
 return snapshot({ ...input, parameters: { ...input.parameters, stepCount: input.parameters.stepCount + 1 } } as DetailedBalanceTemporalInput) as DetailedBalanceTemporalInput;
}
export function rewindDetailedBalanceTime(input: DetailedBalanceTemporalInput): DetailedBalanceTemporalInput {
 return snapshot({ ...input, parameters: { ...input.parameters, stepCount: 0 } } as DetailedBalanceTemporalInput) as DetailedBalanceTemporalInput;
}
