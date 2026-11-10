/** Reader-only projections of an owned, independently validated response.
 * Packed intervals are kept in their common binary scale; no BigInt, exact
 * algebra, numerical producer or checker runs on the reader thread.
 */
import type {RealTimePhaseValidatedSuccess} from './learning-real-time-phase-protocol';
import {scalarFieldProjection} from './learning-scalar-field-view';
export type RealTimeMatrixKind='hessian'|'inverse'|'formal-connected'|'regulated-connected'|'map'|'initial-covariance'|'final-covariance';
export interface RealTimeMatrixDisplay {kind:RealTimeMatrixKind;component:'real'|'imaginary';scale:'linear'|'signed-log';selectedCell:number;}
export const defaultRealTimeMatrixDisplay:RealTimeMatrixDisplay={kind:'map',component:'real',scale:'linear',selectedCell:0};
type Range=Readonly<{lower:string;upper:string}>;
export interface RealTimeMatrixCell {
  row:number;column:number;real:Range;imaginary:Range;coefficientBits:number;scalePowerOfTwo:number;multiplier:string;
  midpoint:number;exactZero:boolean;unresolved:boolean;projectionAvailable:boolean;
}
export function realTimeMatrixKinds(response:RealTimePhaseValidatedSuccess):RealTimeMatrixKind[]{
  const r=response.result;
  return r.kind==='chain'?['hessian',...(r.inverse?['inverse','formal-connected'] as const:[]),...(r.regulatedConnectedResponse?['regulated-connected'] as const:[]),'map']:
    r.kind==='packet'?['map','initial-covariance','final-covariance']:[];
}
const zero:Range=Object.freeze({lower:'0',upper:'0'});
function projection(range:Range,bits:number){
  const lower=bits?Number(range.lower)*2**(-bits):scalarFieldProjection(range.lower);
  const upper=bits?Number(range.upper)*2**(-bits):scalarFieldProjection(range.upper);
  return lower===null||upper===null||!Number.isFinite(lower)||!Number.isFinite(upper)?null:lower/2+upper/2;
}
export function realTimePhaseMatrixView(response:RealTimePhaseValidatedSuccess,display:RealTimeMatrixDisplay){
  const r=response.result;if(r.kind==='canonical-fold'||!realTimeMatrixKinds(response).includes(display.kind))return null;
  let n=2,offset=0,labels=['q','p'],complex=false,bits=0,exponent=0,multiplier='1';
  let entry:(i:number,j:number)=>{real:Range;imaginary:Range};
  const rational=(value:string):Range=>({lower:value,upper:value});
  if(display.kind==='map'||r.kind==='packet'){
    const matrix=display.kind==='initial-covariance'&&r.kind==='packet'?r.exact.initialCovariance:display.kind==='final-covariance'&&r.kind==='packet'?r.exact.finalCovariance:r.exact.map;
    entry=(i,j)=>({real:rational(matrix[i][j]),imaginary:zero});
  }else{
    n=r.exact.dimension;offset=1;labels=Array.from({length:n},(_,i)=>String(i+offset));
    if(display.kind==='hessian')entry=(i,j)=>({real:rational(i===j?r.exact.hessian.diagonal:Math.abs(i-j)===1?r.exact.hessian.offDiagonal:'0'),imaginary:zero});
    else{
      const packed=display.kind==='regulated-connected'?r.regulatedConnectedResponse:r.inverse;if(!packed)return null;
      bits=packed.coefficientBits;exponent=packed.scalePowerOfTwo;
      complex=display.kind==='regulated-connected'||display.kind==='formal-connected';
      if(display.kind==='formal-connected')multiplier=r.formalConnectedResponse.inverseMultiplier;
      entry=(i,j)=>{
        const a=Math.min(i,j),b=Math.max(i,j),cell=packed.upperRows[a][b-a];
        if('real' in cell)return {real:cell.real,imaginary:cell.imaginary};
        return display.kind==='formal-connected'?{real:zero,imaginary:cell}:{real:cell,imaginary:zero};
      };
    }
  }
  const selected=Number.isInteger(display.selectedCell)?Math.max(0,Math.min(n*n-1,display.selectedCell)):0;
  const source:RealTimeMatrixCell[][]=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>{
    const value=entry(i,j),range=value[display.component],midpoint=projection(range,bits),isZero=range.lower==='0'&&range.upper==='0';
    const unresolved=!isZero&&(range.lower==='0'||range.lower.startsWith('-'))&&!range.upper.startsWith('-');
    return {row:i,column:j,...value,coefficientBits:bits,scalePowerOfTwo:exponent,multiplier,midpoint:midpoint??0,exactZero:isZero,unresolved,projectionAvailable:midpoint!==null};
  }));
  const projectionAvailable=source.every(row=>row.every(cell=>cell.projectionAvailable));
  const values=source.map(row=>row.map(cell=>cell.midpoint));
  const transform=(v:number)=>display.scale==='signed-log'?Math.sign(v)*Math.log1p(1e6*Math.abs(v))/Math.log1p(1e6):v;
  const heights=values.map(row=>row.map(transform)),maximum=Math.max(...heights.flat().map(Math.abs))||1;
  const points=heights.flatMap((row,i)=>row.map((value,j)=>{
    const cell=source[i][j],y=2.5*value/maximum;
    const marker=!cell.projectionAvailable?'unavailable':cell.exactZero?'zero':cell.unresolved?'unresolved':Math.abs(y)<1e-6?'tiny':'bar';
    return {x:n===1?0:6*i/(n-1)-3,z:n===1?0:6*j/(n-1)-3,y,value,selected:i*n+j===selected,marker};
  }));
  return {n,offset,labels,selected,row:Math.floor(selected/n),column:selected%n,source,values,heights,points,maximum,projectionAvailable,complex,
    kind:display.kind,component:display.component,scale:display.scale,coefficientBits:bits,scalePowerOfTwo:exponent,multiplier,
    selectedEntry:source[Math.floor(selected/n)][selected%n],unresolved:source.flat().filter(x=>x.unresolved).length,
    displayTransform:display.scale==='linear'?'f':'sign(f) log(1+10^6|f|)/log(1+10^6)'};
}
export type RealTimePhaseMatrixView=NonNullable<ReturnType<typeof realTimePhaseMatrixView>>;
