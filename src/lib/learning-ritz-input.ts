/** Full-rank general complex trial bases; no floating rank tolerance. */
import {f,c,type C} from './learning-quantum-hilbert-core';
import {ritzGram,ritzInverseGram} from './learning-ritz-exact';
export const ritzKeys=[
 'dimension','columns','hDenominator',
 ...Array.from({length:4},(_,i)=>Array.from({length:4-i},(_,k)=>{const j=i+k;return i===j?[`hReal${i}${j}`]:[`hReal${i}${j}`,`hImag${i}${j}`];}).flat()).flat(),
 ...Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>[`basisReal${i}${j}`,`basisImag${i}${j}`])).flat(2),
 ...Array.from({length:4},(_,i)=>[`coefficientReal${i}`,`coefficientImag${i}`]).flat(),
 'shearReal','shearImag','scaleNumerator','scaleDenominator',
]as readonly string[];
export type RitzParameters=Record<string,number>;
export function ritzInputBasis(p:RitzParameters):C[][]{return Array.from({length:p.dimension},(_,i)=>Array.from({length:p.columns},(_,j)=>c(f(BigInt(p[`basisReal${i}${j}`])),f(BigInt(p[`basisImag${i}${j}`])))));}
export function isRitzParameters(value:unknown):value is RitzParameters{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==ritzKeys.length)return false;
 const ds=Object.getOwnPropertyDescriptors(value);if(!ritzKeys.every(k=>Object.hasOwn(ds,k)&&Object.hasOwn(ds[k],'value')&&Number.isSafeInteger(ds[k].value)&&Math.abs(ds[k].value)<=1000000))return false;
 const p=value as RitzParameters,n=p.dimension,k=p.columns;if(n<1||n>4||k<1||k>n||p.hDenominator<1||p.scaleDenominator<1||p.scaleNumerator===0)return false;
 for(let i=0;i<4;i++){
  if(i>=k&&(p[`coefficientReal${i}`]!==0||p[`coefficientImag${i}`]!==0))return false;
  for(let j=0;j<4;j++)if((i>=n||j>=k)&&(p[`basisReal${i}${j}`]!==0||p[`basisImag${i}${j}`]!==0))return false;
  for(let j=i;j<4;j++)if((i>=n||j>=n)&&(p[`hReal${i}${j}`]!==0||i!==j&&p[`hImag${i}${j}`]!==0))return false;
 }
 if(k===1&&(p.shearReal!==0||p.shearImag!==0))return false;
 if(!Array.from({length:k},(_,i)=>p[`coefficientReal${i}`]!==0||p[`coefficientImag${i}`]!==0).some(Boolean))return false;
 try{ritzInverseGram(ritzGram(ritzInputBasis(p)));}catch{return false;}
 return true;
}
export function snapshotRitzParameters(value:RitzParameters):Readonly<RitzParameters>{
 if(!isRitzParameters(value))throw new RangeError('63 owned bounded integers; dimension1..4,1<=columns<=dimension,positive denominators,nonzero scale/coefficient vector,independent basis and zero inactive coordinates required');
 return Object.freeze(Object.fromEntries(ritzKeys.map(k=>[k,value[k]+0])));
}
