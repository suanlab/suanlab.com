/** Pure-state C2 closed-system preparation in declared E0,T0 units. */
export const spectralEvolutionKeys=['stateReal0','stateImag0','stateReal1','stateImag1','axisReal0','axisImag0','axisReal1','axisImag1','centerNumerator','centerDenominator','gapNumerator','gapDenominator','timeNumerator','timeDenominator','hbarNumerator','hbarDenominator','sampleSteps','eulerSteps'] as const;
export type SpectralEvolutionParameters=Record<typeof spectralEvolutionKeys[number],number>;
export function isSpectralEvolutionParameters(value:unknown):value is SpectralEvolutionParameters{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==spectralEvolutionKeys.length)return false;
 const p=value as SpectralEvolutionParameters;
 if(!spectralEvolutionKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000))return false;
 return p.timeNumerator>=0&&p.timeDenominator>0&&p.hbarNumerator>0&&p.hbarDenominator>0&&p.centerDenominator>0&&p.gapDenominator>0&&p.sampleSteps>=1&&p.sampleSteps<=16&&p.eulerSteps>=1&&p.eulerSteps<=64&&p.eulerSteps%p.sampleSteps===0&&['state','axis'].every(prefix=>[0,1].some(i=>p[`${prefix}Real${i}` as keyof SpectralEvolutionParameters]!==0||p[`${prefix}Imag${i}` as keyof SpectralEvolutionParameters]!==0));
}
export function snapshotSpectralEvolutionParameters(p:SpectralEvolutionParameters):Readonly<SpectralEvolutionParameters>{if(!isSpectralEvolutionParameters(p))throw new RangeError('Eighteen owned bounded integers; nonzero state/axis,positive denominators and hbar,nonnegative time;1..16 samples and1..64 Euler steps divisible by sample count');return Object.freeze(Object.fromEntries(spectralEvolutionKeys.map(k=>[k,p[k]+0]))) as SpectralEvolutionParameters;}
