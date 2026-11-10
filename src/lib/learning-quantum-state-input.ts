/** Raw integer representatives, not probabilities or normalized amplitudes. */
export const quantumStateKeys=['stateReal0','stateImag0','stateReal1','stateImag1','referenceReal0','referenceImag0','referenceReal1','referenceImag1','basisReal0','basisImag0','basisReal1','basisImag1','phaseA','phaseB','scaleNumerator','scaleDenominator']as const;
export type QuantumStateParameters=Record<typeof quantumStateKeys[number],number>;
export function isQuantumStateParameters(value:unknown):value is QuantumStateParameters{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==quantumStateKeys.length)return false;const p=value as QuantumStateParameters;
 if(!quantumStateKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000))return false;
 return p.scaleDenominator>0&&p.scaleNumerator!==0&&(p.phaseA!==0||p.phaseB!==0)&&['state','reference','basis'].every(s=>[s+'Real0',s+'Imag0',s+'Real1',s+'Imag1'].some(k=>p[k as keyof QuantumStateParameters]!==0));
}
export function snapshotQuantumStateParameters(p:QuantumStateParameters):Readonly<QuantumStateParameters>{if(!isQuantumStateParameters(p))throw new RangeError('Sixteen owned integers with magnitudes<=1e6; nonzero state,reference,basis,phase pair and scale; positive scale denominator');return Object.freeze(Object.fromEntries(quantumStateKeys.map(k=>[k,p[k]+0])))as QuantumStateParameters;}
