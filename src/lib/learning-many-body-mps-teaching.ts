/** Reader-safe tensor coordinates and fixed teaching inputs; no MPS engine. */
import {createMpsParameters,mpsParameterKeys} from './learning-mps-input';
import {readManyBodyInput,snapshotManyBodyInput,type ManyBodyInput} from './learning-many-body-input';
import type {ManyBodySuccess} from './learning-many-body-protocol';
export const mpsLearningId='matrix-product-state-contraction-gauge' as const;
export type MpsTeachingResult=Extract<ManyBodySuccess,{kind:'mps'}>['result'];
export const mpsText=(ko:string,en:string)=>({ko,en});
const prepare=(sites:number,bond:number,entries:Record<string,number>,gauge:Record<string,number>={})=>{
 const p={...createMpsParameters(sites,bond)};for(const key of mpsParameterKeys)if(/^t\d(?:Real|Imag)/.test(key))p[key]=0;
 return snapshotManyBodyInput({kind:'mps',parameters:{...p,...entries,...gauge}});
};
const ghz={t0Real000:1,t0Real011:1,t1Real000:1,t1Real111:1,t2Real000:1,t2Real110:1};
export const mpsTeachingPresets=[
 {id:'ghz-real-gauge',label:mpsText('GHZ · 실수 scale/shear','GHZ · real scale/shear'),input:prepare(3,2,ghz,{gaugeShearDenominator:2})},
 {id:'ghz-complex-gauge',label:mpsText('GHZ · 복소 gauge','GHZ · complex gauge'),input:prepare(3,2,ghz,{gaugeScaleNumerator:-3,gaugeScaleDenominator:2,gaugeShearReal:2,gaugeShearImag:-3,gaugeShearDenominator:5})},
 {id:'complex-cancellation',label:mpsText('복소 상쇄 · 전체 영벡터','Complex cancellation · global zero vector'),input:prepare(2,2,{t0Real000:1,t0Imag001:1,t1Real000:1,t1Imag100:1},{gaugeShearDenominator:2})},
 {id:'real-cancellation',label:mpsText('실수 상쇄 · 비영 tensor','Real cancellation · nonzero tensors'),input:prepare(2,2,{t0Real000:1,t0Real001:1,t1Real000:1,t1Real100:-1},{gaugeShearDenominator:2})},
 {id:'right-y-product',label:mpsText('오른쪽 Y+ · 복소 위상','Right Y+ · complex phase'),input:prepare(2,1,{t0Real000:1,t1Real000:1,t1Imag010:1})},
 {id:'redundant-bond-four',label:mpsText('Bond4 · rank1 곱 상태','Bond4 · rank1 product state'),input:prepare(3,4,{t0Real000:1,t1Real000:1,t2Real000:1},{gaugeScaleNumerator:1,gaugeShearReal:0})},
 {id:'single-complex-site',label:mpsText('한 위치 · 가상 bond 없음','One site · no virtual bond'),input:prepare(1,1,{t0Real000:1,t0Imag010:1})},
 {id:'four-site-unequal-bonds',label:mpsText('네 위치 · bond2/3/2','Four sites · bonds2/3/2'),input:prepare(4,2,{t0Real000:1,t0Real011:1,t1Real000:1,t1Real111:1,t2Real000:1,t2Real111:1,t3Real000:1,t3Real110:1},{bond1:3,gaugeBond:1,gaugeColumn:2,gaugeScaleNumerator:-2,gaugeShearReal:0,gaugeShearImag:1,gaugeShearDenominator:2})},
] as const;
export function mpsTeachingInput(index=0):ManyBodyInput<'mps'>{if(!Number.isInteger(index)||index<0||index>=mpsTeachingPresets.length)throw new RangeError('Unknown MPS preset');return snapshotManyBodyInput(mpsTeachingPresets[index].input);}
export function editMpsTeachingField(input:ManyBodyInput,key:string,raw:string):ManyBodyInput<'mps'>|null{
 if(input.kind!=='mps'||!mpsParameterKeys.includes(key)||!(/^-?(?:0|[1-9]\d*)$/).test(raw)||raw==='-0'||raw.length>8)return null;const value=Number(raw);if(!Number.isSafeInteger(value)||Math.abs(value)>1000000)return null;return readManyBodyInput({kind:'mps',parameters:{...input.parameters,[key]:value}})as ManyBodyInput<'mps'>|null;
}
export function mpsTeachingFields(input:ManyBodyInput<'mps'>){const p=input.parameters;
 return mpsParameterKeys.filter(key=>{
  if(['sites','bond0','bond1','bond2'].includes(key))return false;
  const tensor=/^t(\d)(Real|Imag)(\d)(\d)(\d)$/.exec(key);
  if(!tensor)return p.sites>1||!key.startsWith('gauge');const site=Number(tensor[1]),l=Number(tensor[3]),r=Number(tensor[5]);return site<p.sites&&l<(site===0?1:p[`bond${site-1}`])&&r<(site===p.sites-1?1:p[`bond${site}`]);
 }).map(key=>{const tensor=/^t(\d)(Real|Imag)(\d)(\d)(\d)$/.exec(key),label=tensor?mpsText(`A${tensor[1]}[s=${tensor[4]}][${tensor[3]},${tensor[5]}] ${tensor[2]==='Real'?'실수':'허수'} 분자`,`A${tensor[1]}[s=${tensor[4]}][${tensor[3]},${tensor[5]}] ${tensor[2]==='Real'?'real':'imaginary'} numerator`):key==='tensorDenominator'?mpsText('모든 tensor 성분의 공통 양의 분모','Positive common denominator of tensor entries'):mpsText(`${key} · gauge 좌표`,`${key} · gauge coordinate`);return{key,label,value:p[key]};});
}
/** A selected prefix row is already a verified contraction. Never replace it
 * with a probability or recompute exact arithmetic in a render callback. */
export function mpsPrefixRows(result:MpsTeachingResult,step:number,gauged=false){if(!Number.isInteger(step)||step<0||step>result.sites)throw new RangeError('Active prefix cut required');return(gauged?result.gauged:result.original).prefixes[step].map((row,index)=>({physicalPrefix:step===0?'∅':index.toString(2).padStart(step,'0'),row}));}
