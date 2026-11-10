/** Reader-safe spin experiments: inputs only, never numerical engines. */
import { createSpinChainParameters, spinChainParameterKeys, spinChainEdges, spinChainAxes } from './learning-spin-chain-input';
import { snapshotManyBodyInput, readManyBodyInput, type ManyBodyInput } from './learning-many-body-input';
export const spinChainLearningId = 'finite-spin-chain-local-hamiltonians' as const;
export type SpinTeachingText = { ko: string; en: string };
export const spinText = (ko: string, en: string): SpinTeachingText => ({ ko, en });
function chain(ghz: boolean, transverse: boolean): ManyBodyInput<'spin-chain'> {
  const p = { ...createSpinChainParameters(4) }; p.hDenominator = 2;
  for (let i = 0; i < 4; i++) p[`fieldX${i}`] = transverse ? -1 : 0;
  for (let i = 0; i < 3; i++) p[`couplingZZ${i}${i + 1}`] = -2;
  if (ghz) p.stateReal15 = 1;
  return snapshotManyBodyInput({ kind: 'spin-chain', parameters: p });
}
function triangle(ground: boolean): ManyBodyInput<'spin-chain'> {
  const p = { ...createSpinChainParameters(3, 1) }; p.stateReal0 = ground ? 0 : 1; p.stateReal1 = ground ? 1 : 0;
  for (const [i,j] of spinChainEdges(3,1)) p[`couplingZZ${i}${j}`] = 1;
  return snapshotManyBodyInput({ kind: 'spin-chain', parameters: p });
}
function complexY(): ManyBodyInput<'spin-chain'> {
  const p = { ...createSpinChainParameters(1) }; p.stateImag1 = 1; p.fieldY0 = -1;
  return snapshotManyBodyInput({ kind: 'spin-chain', parameters: p });
}
export const spinChainTeachingPresets = [
  { id: 'product-transverse', label: spinText('곱 상태 · 횡장', 'Product state · transverse field'), input: chain(false,true) },
  { id: 'ghz-transverse', label: spinText('GHZ · 횡장', 'GHZ · transverse field'), input: chain(true,true) },
  { id: 'ghz-diagonal', label: spinText('GHZ · 대각 Ising', 'GHZ · diagonal Ising'), input: chain(true,false) },
  { id: 'triangle-excited', label: spinText('삼각형 · 높은 고유상태', 'Triangle · excited eigenstate'), input: triangle(false) },
  { id: 'triangle-ground', label: spinText('삼각형 · 바닥 고유상태', 'Triangle · ground eigenstate'), input: triangle(true) },
  { id: 'complex-y', label: spinText('복소 위상 · Y=+1', 'Complex phase · Y=+1'), input: complexY() },
] as const;
export function spinChainTeachingInput(index = 0): ManyBodyInput<'spin-chain'> {
  if (!Number.isInteger(index) || index < 0 || index >= spinChainTeachingPresets.length) throw new RangeError('Unknown spin teaching preset');
  return snapshotManyBodyInput(spinChainTeachingPresets[index].input);
}
export function editSpinTeachingField(input: ManyBodyInput, key: string, draft: string): ManyBodyInput<'spin-chain'> | null {
  if (input.kind !== 'spin-chain' || !spinChainParameterKeys.includes(key) || !/^-?(?:0|[1-9]\d*)$/.test(draft) || draft === '-0' || draft.length > 8) return null;
  const value = Number(draft); if (!Number.isSafeInteger(value) || Math.abs(value)>1000000) return null;
  const next = readManyBodyInput({kind:'spin-chain',parameters:{...input.parameters,[key]:value}});return next?.kind === 'spin-chain' ? next as ManyBodyInput<'spin-chain'> : null;
}
export function spinTeachingFields(input: ManyBodyInput<'spin-chain'>) {
  const p = input.parameters, n = p.sites, allowed = new Set(spinChainEdges(n,p.graph).map(([i,j])=>`${i}${j}`));
  return spinChainParameterKeys.filter(key => {
    const field=/^field[XYZ](\d)$/.exec(key), bond=/^coupling[XYZ][XYZ](\d\d)$/.exec(key), state=/^state(?:Real|Imag)(\d+)$/.exec(key);
    return field ? Number(field[1])<n : bond ? allowed.has(bond[1]) : state ? Number(state[1])<2**n : !['sites','graph'].includes(key);
  }).map(key => {
    const field=/^field([XYZ])(\d)$/.exec(key),bond=/^coupling([XYZ])([XYZ])(\d)(\d)$/.exec(key),state=/^state(Real|Imag)(\d+)$/.exec(key);
    const label = field ? spinText(`위치 ${field[2]}의 ${field[1]} 장 분자`,`Site ${field[2]}: ${field[1]} field numerator`) : bond ? spinText(`${bond[3]}–${bond[4]}의 ${bond[1]}${bond[2]} 결합 분자`,`${bond[3]}–${bond[4]}: ${bond[1]}${bond[2]} coupling numerator`) : state ? spinText(`|${Number(state[2]).toString(2).padStart(n,'0')}〉 ${state[1]==='Real'?'실수':'허수'} 진폭`,`|${Number(state[2]).toString(2).padStart(n,'0')}〉 ${state[1]==='Real'?'real':'imaginary'} amplitude`) : key==='offset'?spinText('상수 에너지 분자','Constant energy numerator'):spinText('Hamiltonian 공통 양의 분모','Positive common Hamiltonian denominator');
    return {key,label,value:p[key]};
  });
}
export { spinChainAxes };
