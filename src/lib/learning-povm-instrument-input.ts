/** Rational Bloch state, two Pauli directions and outcome-specific SU(2) feedback. */
export const povmInstrumentKeys=['blochX','blochY','blochZ','blochDenominator','axisReal0','axisImag0','axisReal1','axisImag1','followReal0','followImag0','followReal1','followImag1','strengthNumerator','strengthDenominator','minusReal','minusX','minusY','minusZ','plusReal','plusX','plusY','plusZ'] as const;
export type PovmInstrumentParameters=Record<typeof povmInstrumentKeys[number],number>;
export function isPovmInstrumentParameters(x:unknown):x is PovmInstrumentParameters{
 if(!x||typeof x!=='object'||Object.getPrototypeOf(x)!==Object.prototype||Reflect.ownKeys(x).length!==22)return false;const p=x as PovmInstrumentParameters;
 if(!povmInstrumentKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000)||p.blochDenominator<=0||p.strengthDenominator<=0||p.strengthNumerator<0||p.strengthNumerator>p.strengthDenominator)return false;
 return [p.blochX,p.blochY,p.blochZ].reduce((s,v)=>s+BigInt(v)**BigInt(2),BigInt(0))<=BigInt(p.blochDenominator)**BigInt(2)&&(['axis','follow']as const).every(prefix=>(['Real0','Imag0','Real1','Imag1']as const).some(k=>p[`${prefix}${k}`]!==0))&&(['minus','plus']as const).every(prefix=>(['Real','X','Y','Z']as const).some(k=>p[`${prefix}${k}`]!==0));
}
export function snapshotPovmInstrumentParameters(p:PovmInstrumentParameters):Readonly<PovmInstrumentParameters>{if(!isPovmInstrumentParameters(p))throw new RangeError('22 owned integers |x|<=1e6; positive denominators,exact Bloch ball,strength in[0,1],nonzero axis/follow spinors and feedback quaternions');return Object.freeze(Object.fromEntries(povmInstrumentKeys.map(k=>[k,p[k]+0])))as PovmInstrumentParameters;}
