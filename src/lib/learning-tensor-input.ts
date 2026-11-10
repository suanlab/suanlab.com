/** Reader-safe canonical preparations. Mixed PSD is checked with small exact minors. */
import {isTensorPureParameters,snapshotTensorPureParameters,type TensorPureParameters} from './learning-tensor-pure-input';
import type {MixedEntanglementParameters} from './learning-mixed-entanglement-input';
import {mixedMatrix,invariants} from './learning-tensor-wire-math';
export const tensorKinds=['bipartite-pure-schmidt-partial-trace','two-qubit-ppt-correlations-chsh'] as const;
export type TensorInput={kind:typeof tensorKinds[0];parameters:TensorPureParameters}|{kind:typeof tensorKinds[1];parameters:MixedEntanglementParameters};
export const tensorMixedKeys=[...Array.from({length:4},(_,i)=>Array.from({length:4-i},(_,j)=>`hReal${i}${i+j}`)).flat(),...Array.from({length:3},(_,i)=>Array.from({length:3-i},(_,j)=>`hImag${i}${i+j+1}`)).flat(),...['a0','a1','b0','b1'].flatMap(s=>[`${s}Real0`,`${s}Imag0`,`${s}Real1`,`${s}Imag1`])] as readonly string[];
export const tensorRecord=(x:unknown):x is Record<string,unknown>=>!!x&&typeof x==='object'&&Object.getPrototypeOf(x)===Object.prototype;
export function isTensorMixedParameters(x:unknown):x is MixedEntanglementParameters{
 if(!tensorRecord(x)||Reflect.ownKeys(x).length!==32)return false;const p=x as MixedEntanglementParameters;
 if(!tensorMixedKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000)||p.hReal00+p.hReal11+p.hReal22+p.hReal33<=0||!['a0','a1','b0','b1'].every(s=>['Real0','Imag0','Real1','Imag1'].some(k=>p[s+k]!==0)))return false;
 return invariants(mixedMatrix(p)).psd;
}
export function isTensorInput(x:unknown):x is TensorInput{return tensorRecord(x)&&Reflect.ownKeys(x).length===2&&Object.hasOwn(x,'kind')&&Object.hasOwn(x,'parameters')&&(x.kind===tensorKinds[0]?isTensorPureParameters(x.parameters):x.kind===tensorKinds[1]&&isTensorMixedParameters(x.parameters));}
export function snapshotTensorInput(x:TensorInput):TensorInput{
 if(!isTensorInput(x))throw new RangeError('Invalid tensor preparation');
 return Object.freeze(x.kind===tensorKinds[0]?{kind:x.kind,parameters:snapshotTensorPureParameters(x.parameters)}:{kind:x.kind,parameters:Object.freeze(Object.fromEntries(tensorMixedKeys.map(k=>[k,x.parameters[k]+0])))});
}
export const tensorInputKey=(x:TensorInput)=>JSON.stringify(snapshotTensorInput(x));
