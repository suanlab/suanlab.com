/** Reader-safe preparations:exact small Gram rank,without spectral engines. */
import {isRayleighParameters,snapshotRayleighParameters,rayleighKeys,type RayleighParameters}from'./learning-rayleigh-input';
import type{RitzParameters}from'./learning-ritz-input';
import{int,c,adjoint,product,determinant}from'./learning-tensor-wire-math';
export const quantumVariationalKinds=['finite-state-rayleigh-energy-variance','nonorthogonal-ritz-subspace-certificates']as const;
export type QuantumVariationalInput={kind:typeof quantumVariationalKinds[0];parameters:RayleighParameters}|{kind:typeof quantumVariationalKinds[1];parameters:RitzParameters};
export const quantumVariationalRitzKeys=['dimension','columns','hDenominator',...Array.from({length:4},(_,i)=>Array.from({length:4-i},(_,k)=>{const j=i+k;return i===j?[`hReal${i}${j}`]:[`hReal${i}${j}`,`hImag${i}${j}`];}).flat()).flat(),...Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>[`basisReal${i}${j}`,`basisImag${i}${j}`])).flat(2),...Array.from({length:4},(_,i)=>[`coefficientReal${i}`,`coefficientImag${i}`]).flat(),'shearReal','shearImag','scaleNumerator','scaleDenominator']as readonly string[];
export const quantumVariationalRecord=(x:unknown):x is Record<string,unknown>=>!!x&&typeof x==='object'&&Object.getPrototypeOf(x)===Object.prototype;
export function quantumVariationalDataRecord(x:unknown,keys:readonly string[]):x is Record<string,unknown>{if(!quantumVariationalRecord(x)||Reflect.ownKeys(x).length!==keys.length)return false;const ds=Object.getOwnPropertyDescriptors(x);return keys.every(k=>Object.hasOwn(ds,k)&&Object.hasOwn(ds[k],'value'));}
export function isQuantumVariationalRitzParameters(x:unknown):x is RitzParameters{
 if(!quantumVariationalDataRecord(x,quantumVariationalRitzKeys))return false;const p=x as RitzParameters;
 if(!quantumVariationalRitzKeys.every(k=>Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000))return false;
 const n=p.dimension,k=p.columns;if(n<1||n>4||k<1||k>n||p.hDenominator<1||p.scaleDenominator<1||p.scaleNumerator===0||k===1&&(p.shearReal!==0||p.shearImag!==0))return false;
 for(let i=0;i<4;i++){
  if(i>=k&&(p[`coefficientReal${i}`]!==0||p[`coefficientImag${i}`]!==0))return false;
  for(let j=0;j<4;j++)if((i>=n||j>=k)&&(p[`basisReal${i}${j}`]!==0||p[`basisImag${i}${j}`]!==0))return false;
  for(let j=i;j<4;j++)if((i>=n||j>=n)&&(p[`hReal${i}${j}`]!==0||i!==j&&p[`hImag${i}${j}`]!==0))return false;
 }
 if(!Array.from({length:k},(_,i)=>p[`coefficientReal${i}`]!==0||p[`coefficientImag${i}`]!==0).some(Boolean))return false;
 const B=Array.from({length:n},(_,i)=>Array.from({length:k},(_,j)=>c(int(p[`basisReal${i}${j}`]),int(p[`basisImag${i}${j}`])))),det=determinant(product(adjoint(B),B));return det.im.n===BigInt(0)&&det.re.n>BigInt(0);
}
export function isQuantumVariationalInput(x:unknown):x is QuantumVariationalInput{return quantumVariationalDataRecord(x,['kind','parameters'])&&(x.kind===quantumVariationalKinds[0]?isRayleighParameters(x.parameters):x.kind===quantumVariationalKinds[1]&&isQuantumVariationalRitzParameters(x.parameters));}
export function snapshotQuantumVariationalInput(x:QuantumVariationalInput):QuantumVariationalInput{
 if(!isQuantumVariationalInput(x))throw new RangeError('Invalid quantum variational preparation');
 return Object.freeze(x.kind===quantumVariationalKinds[0]?{kind:x.kind,parameters:snapshotRayleighParameters(x.parameters)}:{kind:x.kind,parameters:Object.freeze(Object.fromEntries(quantumVariationalRitzKeys.map(k=>[k,x.parameters[k]+0])))});
}
export const quantumVariationalInputKey=(x:QuantumVariationalInput)=>JSON.stringify(snapshotQuantumVariationalInput(x));
export {rayleighKeys};
