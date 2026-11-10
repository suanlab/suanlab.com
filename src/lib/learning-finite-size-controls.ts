/** Scalar editing and spatial-size scans only; no numerical model imports. */
import { finiteSizeText as t, parseFiniteSizeInteger, type FiniteSizeText } from './learning-finite-size';
import { isFiniteSizeInput, snapshotFiniteSizeInput as snapshot, type FiniteSizeInput } from './learning-finite-size-protocol';
export interface FiniteSizeField { id: string; value: number; label: FiniteSizeText; minimum: number; maximum: number; active: boolean }
export interface FiniteSizeChoice { id: string; value: number | string; label: FiniteSizeText; options: { value: number | string; label: FiniteSizeText }[] }
export function finiteSizeFields(input: FiniteSizeInput): FiniteSizeField[] {
 const fields: FiniteSizeField[] = [];
 const add = (id: string, value: number, label: FiniteSizeText, minimum = 1, maximum = 1000000, active = true) => fields.push({ id, value, label, minimum, maximum, active });
 const p = input.parameters;
 if ('dimension' in p) {
  add('lengthOne', p.lengthOne, t('첫 블록 길이 a₀', 'First block length a₀')); add('lengthTwo', p.lengthTwo, t('둘째 블록 길이 b₀', 'Second block length b₀')); add('width', p.width, t('가로 폭 w₀', 'Transverse width w₀')); add('scale', p.scale, t('공간 확대 배율 λ', 'Spatial growth scale λ'));
  add('couplingNumerator', p.couplingNumerator, t('결합 J/E0의 분자', 'Numerator of coupling J/E0'), 0); add('couplingDenominator', p.couplingDenominator, t('결합 J/E0의 분모', 'Denominator of coupling J/E0'));
 } else if ('occupiedWeight' in p) {
  add('siteCount', p.siteCount, t('공간 자리 수 N', 'Spatial site count N'), 1, 512); add('occupiedWeight', p.occupiedWeight, t('점유 무게 A', 'Occupied weight A'), 0); add('emptyWeight', p.emptyWeight, t('비점유 무게 B', 'Empty weight B'), 0);
  add('commonNumerator', p.commonNumerator, t('기본 공통 가중치 c의 분자 C', 'Base collective-weight numerator C'), 0); add('commonDenominator', p.commonDenominator, t('기본 공통 가중치 c의 분모 D', 'Base collective-weight denominator D'));
  add('epsilonNumerator', p.epsilonNumerator, t('편차 임계값 ε의 분자', 'Deviation-threshold numerator')); add('epsilonDenominator', p.epsilonDenominator, t('편차 임계값 ε의 분모', 'Deviation-threshold denominator'));
 } else {
  const fixed = p.sizePath === 'fixed'; add('siteCount', p.siteCount, t('공간 자리 수 N', 'Spatial site count N'), 1, 512);
  add('activityNumerator', p.activityNumerator, t('고정 활동도 z의 분자 A', 'Fixed-activity numerator A'), 1, 1000000, fixed); add('activityDenominator', p.activityDenominator, t('고정 활동도 z의 분모 B', 'Fixed-activity denominator B'), 1, 1000000, fixed);
  add('offsetNumerator', p.offsetNumerator, t('접근 가중치 c의 분자 C', 'Approach-offset numerator C'), 0, 1000000, !fixed); add('offsetDenominator', p.offsetDenominator, t('접근 가중치 c의 분모 D', 'Approach-offset denominator D'), 1, 1000000, !fixed);
 }
 return fields;
}
export function finiteSizeChoices(input: FiniteSizeInput): FiniteSizeChoice[] {
 const p = input.parameters, options = (values: (number | string)[], labels: FiniteSizeText[]) => values.map((value, i) => ({ value, label: labels[i] }));
 if ('dimension' in p) return [
  { id: 'dimension', value: p.dimension, label: t('격자 차원', 'Lattice dimension'), options: options([1, 2, 3], [t('1차원', 'One dimension'), t('2차원', 'Two dimensions'), t('3차원', 'Three dimensions')]) },
  { id: 'growthMode', value: p.growthMode, label: t('공간 성장 경로', 'Spatial growth path'), options: options(['uniform-dilation', 'fixed-thickness'], [t('모든 방향 확대', 'Grow every direction'), t('두께 고정·가로 확대', 'Fixed thickness, transverse growth')]) },
  ...(['spinOne', 'spinTwo'] as const).map((id, i) => ({ id, value: p[id], label: i ? t('둘째 블록 스핀', 'Second block spin') : t('첫 블록 스핀', 'First block spin'), options: options([1, -1], [t('양의 스핀 +1', 'Positive spin +1'), t('음의 스핀 −1', 'Negative spin −1')]) })),
 ];
 if ('occupiedWeight' in p) return [{ id: 'commonScaling', value: p.commonScaling, label: t('공통 가중치의 크기 경로', 'Collective-weight size path'), options: options(['fixed', 'inverse-size', 'inverse-size-squared'], [t('고정 c', 'Fixed c'), t('역크기 c/N', 'Inverse size c/N'), t('역제곱 c/N²', 'Inverse square c/N²')]) }];
 return [
  { id: 'sizePath', value: p.sizePath, label: t('활동도 크기 경로', 'Activity size path'), options: options(['fixed', 'inverse-size', 'inverse-size-squared'], [t('고정 z', 'Fixed z'), t('역크기 접근', 'Inverse-size approach'), t('역제곱 접근', 'Inverse-square approach')]) },
  { id: 'approachSide', value: p.approachSide, label: t('공존점 접근 방향', 'Side of approach to coexistence'), options: options([1, -1], [t('위쪽 · 양의 로그장', 'From above, positive log field'), t('역수 · 음의 로그장', 'Reciprocal, negative log field')]) },
 ];
}
export function editFiniteSizeDraft(input: FiniteSizeInput, fieldId: string, draft: string): FiniteSizeInput | null {
 if (!isFiniteSizeInput(input)) return null;
 const descriptor = finiteSizeFields(input).find(field => field.id === fieldId), value = parseFiniteSizeInteger(draft);
 if (!descriptor || !descriptor.active || value === null || value < descriptor.minimum || value > descriptor.maximum) return null;
 const next = { kind: input.kind, parameters: { ...input.parameters, [fieldId]: value } };
 return isFiniteSizeInput(next) ? snapshot(next) : null;
}
export function editFiniteSizeChoice(input: FiniteSizeInput, choiceId: string, raw: string): FiniteSizeInput | null {
 if (!isFiniteSizeInput(input)) return null;
 if (input.kind === 'two-phase-partition-zeros-rounding' && input.parameters.sizePath === 'fixed' && choiceId === 'approachSide') return null;
 const descriptor = finiteSizeChoices(input).find(choice => choice.id === choiceId), option = descriptor?.options.find(option => String(option.value) === raw);
 if (!option) return null;
 const next = { kind: input.kind, parameters: { ...input.parameters, [choiceId]: option.value } };
 return isFiniteSizeInput(next) ? snapshot(next) : null;
}
export const finiteSizeSize = (input: FiniteSizeInput) => input.kind === 'boundary-growth-extensivity-additivity' ? input.parameters.scale : input.parameters.siteCount;
export const finiteSizeMaximum = (input: FiniteSizeInput) => input.kind === 'boundary-growth-extensivity-additivity' ? 1000000 : 512;
/** Double the spatial control, with an explicitly bounded final scan step. */
export function advanceFiniteSize(input: FiniteSizeInput): FiniteSizeInput | null {
 if (!isFiniteSizeInput(input)) return null;
 const size = finiteSizeSize(input), maximum = finiteSizeMaximum(input); if (size >= maximum) return null;
 return editFiniteSizeDraft(input, input.kind === 'boundary-growth-extensivity-additivity' ? 'scale' : 'siteCount', String(Math.min(maximum, 2 * size)));
}
export function rewindFiniteSize(input: FiniteSizeInput): FiniteSizeInput {
 if (!isFiniteSizeInput(input)) throw new RangeError('Invalid finite-size input');
 const next = editFiniteSizeDraft(input, input.kind === 'boundary-growth-extensivity-additivity' ? 'scale' : 'siteCount', '1');
 if (!next) throw new RangeError('Invalid finite-size rewind'); return next;
}
