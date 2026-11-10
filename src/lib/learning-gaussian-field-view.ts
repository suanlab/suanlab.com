/** Reader projections only. Exact strings in the verified result remain authoritative. */
import type {GaussianFieldResult} from './learning-gaussian-field-protocol';
import {readGaussianFieldInput,type GaussianFieldInput} from './learning-gaussian-field-input';
import {scalarFieldProjection as project} from './learning-scalar-field-view';
export type GaussianMatrixKind='connectedTwo'|'rawTwo'|'susceptibility'|'precision';
export type GaussianScale='linear'|'signed-log';
export interface GaussianFieldDraft {sites:string;boundary:GaussianFieldInput['boundary'];constraint:GaussianFieldInput['constraint'];massSquared:string;stiffness:string;temperature:string;sourceText:string;probesText:string;}
export const gaussianFieldDraft=(input:GaussianFieldInput):GaussianFieldDraft=>({...input,sites:String(input.sites),sourceText:input.source.join(', '),probesText:input.probes.join(', ')});
export function parseGaussianFieldDraft(d:GaussianFieldDraft){
 if(!/^[3-8]$/.test(d.sites)||d.sourceText.length>512||d.probesText.length>64)return null;const probes=d.probesText.split(',').map(s=>s.trim());if(probes.length!==4||probes.some(s=>!/^(?:0|[1-9]\d?)$/.test(s)))return null;
 return readGaussianFieldInput({sites:Number(d.sites),boundary:d.boundary,constraint:d.constraint,massSquared:d.massSquared,stiffness:d.stiffness,temperature:d.temperature,source:d.sourceText.split(',').map(s=>s.trim()),probes:probes.map(Number)});
}
export function gaussianFieldView(r:GaussianFieldResult,kind:GaussianMatrixKind,scale:GaussianScale,selected:number){
 const n=r.input.sites,index=Math.max(0,Math.min(n*n-1,selected)),row=Math.floor(index/n),column=index%n;
 const transform=(value:number)=>scale==='signed-log'?Math.sign(value)*Math.log1p(Math.abs(value))/Math.LN10:value;
 let projectionAvailable=true;const finite=(s:string)=>{const v=project(s);if(v===null){projectionAvailable=false;return 0;}return v;};
 const values=r[kind].map(a=>a.map(finite)),mean=r.mean.map(finite),wick=r.fourPoint.wickPairProducts.map(finite),heights=values.map(a=>a.map(transform)),maximum=Math.max(...heights.flat().map(Math.abs))||1;
 return {projectionAvailable,kind,scale,n,selected:index,row,column,values,heights,maximum,transform,source:r[kind],mean,columnValues:values.map(a=>a[column]),four:r.fourPoint,wick,points:heights.flatMap((a,i)=>a.map((value,j)=>({x:6*i/(n-1)-3,y:2.5*value/maximum,z:6*j/(n-1)-3,value,selected:i===row&&j===column})))};
}
export type GaussianFieldView=ReturnType<typeof gaussianFieldView>;
