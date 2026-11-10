/** Rational Bloch ball and a random-I/Z channel parameter, not elapsed time. */
export const densitySpectrumKeys=['blochX','blochY','blochZ','blochDenominator','probeReal0','probeImag0','probeReal1','probeImag1','etaNumerator','etaDenominator','sampleSteps'] as const;
export type DensitySpectrumParameters=Record<typeof densitySpectrumKeys[number],number>;
export function isDensitySpectrumParameters(x:unknown):x is DensitySpectrumParameters{
 if(!x||typeof x!=='object'||Object.getPrototypeOf(x)!==Object.prototype||Reflect.ownKeys(x).length!==11)return false;const p=x as DensitySpectrumParameters;
 if(!densitySpectrumKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000)||p.blochDenominator<=0||p.etaDenominator<=0||Math.abs(p.etaNumerator)>p.etaDenominator||p.sampleSteps<1||p.sampleSteps>16)return false;
 return ['probeReal0','probeImag0','probeReal1','probeImag1'].some(k=>p[k as keyof DensitySpectrumParameters]!==0)&&[p.blochX,p.blochY,p.blochZ].reduce((s,x)=>s+BigInt(x)**BigInt(2),BigInt(0))<=BigInt(p.blochDenominator)**BigInt(2);
}
export function snapshotDensitySpectrumParameters(x:DensitySpectrumParameters):Readonly<DensitySpectrumParameters>{if(!isDensitySpectrumParameters(x))throw new RangeError('11 owned integers |x|<=1e6; positive denominators,exact Bloch ball,nonzero probe,eta in[-1,1],1..16 sample intervals');return Object.freeze(Object.fromEntries(densitySpectrumKeys.map(k=>[k,x[k]+0]))) as DensitySpectrumParameters;}
