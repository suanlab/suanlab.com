/** General two-qubit Hermitian numerator plus four local binary projective tests. */
import {mixedInputMatrix,mixedHermitianInvariants} from './learning-mixed-entanglement-algebra';
export const mixedEntanglementKeys=[
 ...Array.from({length:4},(_,i)=>Array.from({length:4-i},(_,j)=>`hReal${i}${i+j}`)).flat(),
 ...Array.from({length:3},(_,i)=>Array.from({length:3-i},(_,j)=>`hImag${i}${i+j+1}`)).flat(),
 ...['a0','a1','b0','b1'].flatMap(s=>[`${s}Real0`,`${s}Imag0`,`${s}Real1`,`${s}Imag1`]),
] as readonly string[];
export type MixedEntanglementParameters=Record<string,number>;
export function isMixedEntanglementParameters(x:unknown):x is MixedEntanglementParameters {
 if(!x||typeof x!=='object'||Object.getPrototypeOf(x)!==Object.prototype||Reflect.ownKeys(x).length!==32)return false;
 const p=x as MixedEntanglementParameters;
 if(!mixedEntanglementKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000))return false;
 if(p.hReal00+p.hReal11+p.hReal22+p.hReal33<=0)return false;
 if(!['a0','a1','b0','b1'].every(s=>['Real0','Imag0','Real1','Imag1'].some(k=>p[`${s}${k}`]!==0)))return false;
 return mixedHermitianInvariants(mixedInputMatrix(p)).positiveSemidefinite;
}
export function snapshotMixedEntanglementParameters(x:MixedEntanglementParameters):Readonly<MixedEntanglementParameters> {
 if(!isMixedEntanglementParameters(x))throw new RangeError('32 owned integers of magnitude<=1e6; positive trace,PSD Hermitian numerator and four nonzero local spinors required');
 return Object.freeze(Object.fromEntries(mixedEntanglementKeys.map(k=>[k,x[k]+0])));
}
