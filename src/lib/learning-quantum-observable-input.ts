/** Raw bounded integers; probabilities and normalized coordinates are derived. */
const vectorKeys=(prefix:string,n:number)=>Array.from({length:n},(_,i)=>[`${prefix}Real${i}`,`${prefix}Imag${i}`]).flat();
export const hermitianKeys=['stateReal0','stateImag0','stateReal1','stateImag1','axisReal0','axisImag0','axisReal1','axisImag1','basisReal0','basisImag0','basisReal1','basisImag1','centerNumerator','centerDenominator','gapNumerator','gapDenominator'] as const;
export const sequentialKeys=['stateReal0','stateImag0','stateReal1','stateImag1','stateReal2','stateImag2','axisAReal0','axisAImag0','axisAReal1','axisAImag1','axisAReal2','axisAImag2','axisBReal0','axisBImag0','axisBReal1','axisBImag1','axisBReal2','axisBImag2','aPositiveNumerator','aPositiveDenominator','aNegativeNumerator','aNegativeDenominator','bPositiveNumerator','bPositiveDenominator','bNegativeNumerator','bNegativeDenominator'] as const;
export type HermitianParameters=Record<typeof hermitianKeys[number],number>;
export type SequentialObservationParameters=Record<typeof sequentialKeys[number],number>;
function valid(value:unknown,keys:readonly string[],prefixes:string[],dimension:number):value is Record<string,number>{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)return false;
 const p=value as Record<string,number>;
 return keys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000&&(!k.endsWith('Denominator')||p[k]>0))&&prefixes.every(prefix=>vectorKeys(prefix,dimension).some(k=>p[k]!==0));
}
export const isHermitianParameters=(p:unknown):p is HermitianParameters=>valid(p,hermitianKeys,['state','axis','basis'],2);
export const isSequentialObservationParameters=(p:unknown):p is SequentialObservationParameters=>valid(p,sequentialKeys,['state','axisA','axisB'],3);
function snapshot(p:unknown,keys:readonly string[],validator:(p:unknown)=>p is Record<string,number>){if(!validator(p))throw new RangeError('Owned integers with magnitudes<=1e6, positive denominators and nonzero raw vectors required');return Object.freeze(Object.fromEntries(keys.map(k=>[k,p[k]+0])));}
export const snapshotHermitianParameters=(p:HermitianParameters)=>snapshot(p,hermitianKeys,isHermitianParameters) as Readonly<HermitianParameters>;
export const snapshotSequentialObservationParameters=(p:SequentialObservationParameters)=>snapshot(p,sequentialKeys,isSequentialObservationParameters) as Readonly<SequentialObservationParameters>;
