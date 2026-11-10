/** Reader-safe Schmidt inputs; algebra,spectral roots and logarithms stay in Workers. */
import {createSchmidtParameters,schmidtParameterKeys} from './learning-schmidt-input';
import {readManyBodyInput,snapshotManyBodyInput,type ManyBodyInput} from './learning-many-body-input';
import type {ManyBodySuccess} from './learning-many-body-protocol';
export const schmidtLearningId='schmidt-cuts-bond-rank-truncation' as const;
export type SchmidtTeachingResult=Extract<ManyBodySuccess,{kind:'schmidt'}>['result'];
export type SchmidtTeachingCut=SchmidtTeachingResult['cuts'][number];
export const schmidtText=(ko:string,en:string)=>({ko,en});
const prepare=(sites:number,cut:number,budget:number,entries:Record<string,number>)=>{const p={...createSchmidtParameters(sites,budget)};for(const key of schmidtParameterKeys)if(key.startsWith('state'))p[key]=0;return snapshotManyBodyInput({kind:'schmidt',parameters:{...p,selectedCut:cut,...entries}});};
const unequal={stateReal0:2,stateReal3:1},diagonal={stateReal0:3,stateReal5:2,stateReal10:2,stateReal15:1};
export const schmidtTeachingPresets=[
 {id:'unequal-budget-one',label:schmidtText('비균등2/1 · budget1','Unequal2/1 · budget1'),input:prepare(2,1,1,unequal)},
 {id:'unequal-budget-zero',label:schmidtText('비균등2/1 · budget0','Unequal2/1 · budget0'),input:prepare(2,1,0,unequal)},
 {id:'unequal-full-support',label:schmidtText('비균등2/1 · budget4','Unequal2/1 · budget4'),input:prepare(2,1,4,unequal)},
 {id:'bell-degenerate-boundary',label:schmidtText('Bell · 양의 축퇴 경계','Bell · positive degenerate boundary'),input:prepare(2,1,1,{stateReal0:1,stateReal3:1})},
 ...[0,1,2,3,4].map(budget=>({id:`diagonal-budget-${budget}`,label:schmidtText(`대각3/2/2/1 · budget${budget}`,`Diagonal3/2/2/1 · budget${budget}`),input:prepare(4,2,budget,diagonal)})),
 {id:'right-complex-projector',label:schmidtText('복소 오른쪽 · Pᵀ의 방향','Complex right side · orientation of Pᵀ'),input:prepare(3,2,1,{stateReal0:2,stateImag1:2,stateReal2:1,stateImag3:-1})},
 {id:'tiny-positive-weight',label:schmidtText('아주 작은 양의 두 번째 weight','Tiny positive second weight'),input:prepare(2,1,1,{stateReal0:1000000,stateReal1:999999,stateReal2:999999,stateReal3:999998})},
 {id:'adjacent-bell-pairs',label:schmidtText('인접 Bell pairs · cut rank2/1/2','Adjacent Bell pairs · cut ranks2/1/2'),input:prepare(4,2,4,{stateReal0:1,stateReal3:1,stateReal12:1,stateReal15:1})},
 {id:'crossing-bell-pairs',label:schmidtText('교차 Bell pairs · cut rank2/4/2','Crossing Bell pairs · cut ranks2/4/2'),input:prepare(4,2,4,{stateReal0:1,stateReal5:1,stateReal10:1,stateReal15:1})},
] as const;
export function schmidtTeachingInput(index=0):ManyBodyInput<'schmidt'>{if(!Number.isInteger(index)||index<0||index>=schmidtTeachingPresets.length)throw new RangeError('Unknown Schmidt preset');return snapshotManyBodyInput(schmidtTeachingPresets[index].input);}
export function editSchmidtTeachingField(input:ManyBodyInput,key:string,raw:string):ManyBodyInput<'schmidt'>|null{
 if(input.kind!=='schmidt'||!schmidtParameterKeys.includes(key)||!(/^-?(?:0|[1-9]\d*)$/).test(raw)||raw==='-0'||raw.length>8)return null;const value=Number(raw);if(!Number.isSafeInteger(value)||Math.abs(value)>1000000)return null;return readManyBodyInput({kind:'schmidt',parameters:{...input.parameters,[key]:value}})as ManyBodyInput<'schmidt'>|null;
}
export function schmidtTeachingFields(input:ManyBodyInput<'schmidt'>){const p=input.parameters;return schmidtParameterKeys.filter(key=>{const m=/^state(?:Real|Imag)(\d+)$/.exec(key);return m&&Number(m[1])<2**p.sites;}).map(key=>{const m=/^state(Real|Imag)(\d+)$/.exec(key)!,bits=Number(m[2]).toString(2).padStart(p.sites,'0');return{key,value:p[key],label:schmidtText(`|${bits}〉 ${m[1]==='Real'?'실수':'허수'} 진폭`,`|${bits}〉 ${m[1]==='Real'?'real':'imaginary'} amplitude`)};});}
/** Render one certified group per distinct weight; never invent eigenvectors
 * or subdivide a repeated positive group into chosen physical directions. */
export function schmidtSpectrumGroups(cut:SchmidtTeachingCut){return cut.spectrum.map((group,i)=>({index:i,weight:group.weight,multiplicity:group.multiplicity,zeroWeight:group.zeroWeight,kept:cut.truncation.allocation[i].keptMultiplicity,discarded:cut.truncation.allocation[i].discardedMultiplicity}));}
