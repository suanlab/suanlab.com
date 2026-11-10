/** Floating display projections only; exact verified strings remain authoritative. */
import type {NongaussianFieldResult} from './learning-nongaussian-field-protocol';
import {readNongaussianFieldInput,type NongaussianFieldInput} from './learning-nongaussian-field-input';
import {scalarFieldProjection as project} from './learning-scalar-field-view';
export type NongaussianMatrixKind='connectedTwo'|'rawTwo'|'susceptibility'|'logZHessian';
export type NongaussianScale='linear'|'signed-log';
export interface NongaussianFieldDraft{sites:string;boundary:NongaussianFieldInput['boundary'];pairFactor:string;fourBodyFactor:string;temperature:string;sourceText:string;probesText:string;}
export const nongaussianFieldDraft=(input:NongaussianFieldInput):NongaussianFieldDraft=>({sites:String(input.sites),boundary:input.boundary,pairFactor:input.pairFactor,fourBodyFactor:input.fourBodyFactor,temperature:input.temperature,sourceText:input.sourceFactors.join(', '),probesText:input.probes.join(', ')});
export function parseNongaussianFieldDraft(d:NongaussianFieldDraft){
 if(!/^[4-8]$/.test(d.sites)||d.sourceText.length>512||d.probesText.length>64)return null;const probes=d.probesText.split(',').map(s=>s.trim());if(probes.length!==4||probes.some(s=>!/^(?:0|[1-9]\d?)$/.test(s)))return null;
 return readNongaussianFieldInput({sites:Number(d.sites),boundary:d.boundary,pairFactor:d.pairFactor,fourBodyFactor:d.fourBodyFactor,temperature:d.temperature,sourceFactors:d.sourceText.split(',').map(s=>s.trim()),probes:probes.map(Number)});
}
const clamp=(v:number,count:number)=>Number.isInteger(v)?Math.max(0,Math.min(count-1,v)):0;
export function nongaussianFieldView(r:NongaussianFieldResult,kind:NongaussianMatrixKind,scale:NongaussianScale,selected:number,stateIndex=0){
 const n=r.input.sites,index=clamp(selected,n*n),row=Math.floor(index/n),column=index%n,state=clamp(stateIndex,r.stateCount),source=kind==='logZHessian'?r.sourceDerivatives.logZHessian:r[kind],transform=(v:number)=>scale==='signed-log'?Math.sign(v)*Math.log1p(Math.abs(v))/Math.LN10:v;
 let projectionAvailable=true;const finite=(s:string)=>{const v=project(s);if(v===null){projectionAvailable=false;return 0;}return v;};
 const values=source.map(a=>a.map(finite)),mean=r.mean.map(finite),probabilities=r.states.map(s=>finite(s.probability)),histogram=r.magnetization.map(bin=>finite(bin.probability)),wick=r.fourPoint.wickPairProducts.map(finite),four=[finite(r.fourPoint.centered),wick.reduce((a,b)=>a+b,0),finite(r.fourPoint.connectedCumulant)],heights=values.map(a=>a.map(transform)),maximum=Math.max(...heights.flat().map(Math.abs))||1;
 return{projectionAvailable,kind,scale,n,selected:index,row,column,state,configuration:r.states[state],source,values,mean,probabilities,histogram,wick,four,heights,maximum,transform,columnValues:values.map(a=>a[column]),points:heights.flatMap((a,i)=>a.map((v,j)=>({x:6*i/(n-1)-3,y:2.5*v/maximum,z:6*j/(n-1)-3,value:v,selected:i===row&&j===column})))};
}
export type NongaussianFieldView=ReturnType<typeof nongaussianFieldView>;
