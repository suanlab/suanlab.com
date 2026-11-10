/** Reader-safe boson controls and basis-only ladder illustrations. */
import {createBosonFockParameters as raw,bosonFockParameterKeys} from './learning-boson-fock-input';
import {createBosonCoherentParameters as coherent,bosonCoherentParameterKeys} from './learning-boson-coherent-input';
import {readManyBodyInput,snapshotManyBodyInput,type ManyBodyInput} from './learning-many-body-input';
import type {ManyBodySuccess} from './learning-many-body-protocol';
export const bosonLearningId='boson-fock-cutoff-boundary-algebra' as const;
export type BosonTeachingKind='boson-fock'|'boson-coherent';
export type BosonTeachingResponse=Extract<ManyBodySuccess,{kind:BosonTeachingKind}>;
export type BosonFiniteResult=Extract<ManyBodySuccess,{kind:'boson-fock'}>['result']|Extract<ManyBodySuccess,{kind:'boson-coherent'}>['result']['normalizedProjection'];
export const bosonText=(ko:string,en:string)=>({ko,en});
const input=(kind:BosonTeachingKind,parameters:Readonly<Record<string,number>>)=>snapshotManyBodyInput({kind,parameters});
export const bosonTeachingPresets=[
 {id:'raw-vacuum',label:bosonText('자유 상태 · 진공','Raw state · vacuum'),input:input('boson-fock',{...raw(),onsite0:1})},
 {id:'raw-top',label:bosonText('자유 상태 · 최상위 점유','Raw state · top occupation'),input:input('boson-fock',{...raw(),stateReal0:0,stateReal3:1,onsite0:1})},
 {id:'raw-boundary-cat',label:bosonText('자유 상태 · 진공/경계 중첩','Raw state · vacuum/boundary superposition'),input:input('boson-fock',{...raw(),stateReal3:1,onsite0:1})},
 {id:'raw-cutoff-zero',label:bosonText('최대 점유0 · 차원1','Maximum occupation0 · dimension1'),input:input('boson-fock',{...raw(1,0),onsite0:1})},
 {id:'raw-negative-onsite',label:bosonText('음의 onsite · 유한 바닥만','Negative onsite · finite ground only'),input:input('boson-fock',{...raw(),stateReal0:0,stateReal3:1,onsite0:-1})},
 {id:'raw-two-mode-hopping',label:bosonText('두 모드 · 직사각 cutoff','Two modes · rectangular cutoff'),input:input('boson-fock',{...raw(2,2,1),stateReal0:0,stateReal2:1,hoppingReal:1})},
 {id:'coherent-cutoff-one',label:bosonText('Coherent α=1 · cutoff1','Coherent α=1 · cutoff1'),input:input('boson-coherent',{...coherent(1,1),onsite0:1})},
 {id:'coherent-cutoff-three',label:bosonText('Coherent α=1 · cutoff3','Coherent α=1 · cutoff3'),input:input('boson-coherent',{...coherent(),onsite0:1})},
 {id:'coherent-imaginary',label:bosonText('Coherent α=i · 복소 위상','Coherent α=i · complex phase'),input:input('boson-coherent',{...coherent(),alphaReal0:0,alphaImag0:1,onsite0:1})},
 {id:'coherent-large-tail',label:bosonText('Coherent μ=16 · 큰 버림','Coherent μ=16 · large discarded mass'),input:input('boson-coherent',{...coherent(),alphaReal0:4,onsite0:1})},
 {id:'coherent-tiny-tail',label:bosonText('작지만 양수인 Poisson tail','Tiny positive Poisson tail'),input:input('boson-coherent',{...coherent(),alphaDenominator0:1000000,onsite0:1})},
 {id:'coherent-two-mode',label:bosonText('Coherent 곱 · 두 cutoff','Coherent product · two cutoffs'),input:input('boson-coherent',{...coherent(2,1,2),alphaImag1:1,onsite0:1,onsite1:1})},
] as const;
export function bosonTeachingInput(index=0):ManyBodyInput<BosonTeachingKind>{if(!Number.isInteger(index)||index<0||index>=bosonTeachingPresets.length)throw new RangeError('Unknown boson preset');return snapshotManyBodyInput(bosonTeachingPresets[index].input);}
export function editBosonTeachingField(input:ManyBodyInput,key:string,raw:string):ManyBodyInput<BosonTeachingKind>|null{
 if(!['boson-fock','boson-coherent'].includes(input.kind)||!(/^-?(?:0|[1-9]\d*)$/).test(raw)||raw==='-0'||raw.length>8)return null;const keys=input.kind==='boson-fock'?bosonFockParameterKeys:bosonCoherentParameterKeys;if(!keys.includes(key))return null;const n=Number(raw);if(!Number.isSafeInteger(n)||Math.abs(n)>1000000)return null;const next=readManyBodyInput({kind:input.kind,parameters:{...input.parameters,[key]:n}});return next as ManyBodyInput<BosonTeachingKind>|null;
}
export function bosonTeachingFields(input:ManyBodyInput<BosonTeachingKind>){const p=input.parameters,keys=input.kind==='boson-fock'?bosonFockParameterKeys:bosonCoherentParameterKeys,dimension=(p.cutoff0+1)*(p.cutoff1+1);
 return keys.filter(key=>{if(['modes','cutoff0','cutoff1','leftMode','rightMode','leftCreation','rightCreation'].includes(key))return false;const state=/^state(?:Real|Imag)(\d+)$/.exec(key);if(state)return Number(state[1])<dimension;if(p.modes===1&&(['onsite1','selfDensity1','driveReal1','driveImag1','hoppingReal','hoppingImag','crossDensity','alphaReal1','alphaImag1','alphaDenominator1'].includes(key)))return false;return true;}).map(key=>{
  const state=/^state(Real|Imag)(\d+)$/.exec(key),alpha=/^alpha(Real|Imag|Denominator)([01])$/.exec(key),drive=/^drive(Real|Imag)([01])$/.exec(key),onsite=/^(onsite|selfDensity)([01])$/.exec(key);
  const index=state?Number(state[2]):0,labelBasis=p.modes===1?String(index):`${Math.floor(index/(p.cutoff1+1))},${index%(p.cutoff1+1)}`;
  const label=state?bosonText(`|${labelBasis}〉 ${state[1]==='Real'?'실수':'허수'} 진폭`,`|${labelBasis}〉 ${state[1]==='Real'?'real':'imaginary'} amplitude`):alpha?bosonText(`모드 ${alpha[2]} α ${alpha[1]==='Denominator'?'양의 분모':alpha[1]==='Real'?'실수 분자':'허수 분자'}`,`Mode ${alpha[2]} α ${alpha[1]==='Denominator'?'positive denominator':alpha[1]==='Real'?'real numerator':'imaginary numerator'}`):drive?bosonText(`모드 ${drive[2]} drive ${drive[1]==='Real'?'실수':'허수'} 분자`,`Mode ${drive[2]} drive ${drive[1]==='Real'?'real':'imaginary'} numerator`):onsite?bosonText(`모드 ${onsite[2]} ${onsite[1]} 분자`,`Mode ${onsite[2]} ${onsite[1]} numerator`):key==='hDenominator'?bosonText('Hamiltonian 공통 양의 분모','Positive common Hamiltonian denominator'):key==='offset'?bosonText('상수 에너지 분자','Constant energy numerator'):bosonText(`${key} 분자`,`${key} numerator`);return{key,label,value:p[key]};
 });
}
export type BosonLadderStep={basis:number|null;squaredCoefficient:number;mode:number|null;creation:boolean|null};
/** A basis illustration omits the complex input coefficient. Operator targets
 * come from validated matrices; integer occupations give exact sqrt factors. */
export function bosonBasisTrace(r:BosonFiniteResult,p:Readonly<Record<string,number>>,basis:number,reverse=false):BosonLadderStep[]{
 if(!Number.isInteger(basis)||basis<0||basis>=r.dimension)throw new RangeError('Active boson basis required');
 const actions=reverse?[[p.leftMode,p.leftCreation],[p.rightMode,p.rightCreation]]:[[p.rightMode,p.rightCreation],[p.leftMode,p.leftCreation]],steps:BosonLadderStep[]=[{basis,squaredCoefficient:1,mode:null,creation:null}];
 for(const[mode,creation]of actions){const prior=steps.at(-1)!;if(prior.basis===null){steps.push({basis:null,squaredCoefficient:0,mode,creation:!!creation});continue;}const a=r.operators[mode].annihilation,target=creation?a[prior.basis].findIndex(v=>v.real.exactSign!==0):a.findIndex(row=>row[prior.basis!].real.exactSign!==0),index=target<0?null:target,n=r.basisOccupations[prior.basis][mode];steps.push({basis:index,squaredCoefficient:index===null?0:prior.squaredCoefficient*(creation?n+1:n),mode,creation:!!creation});}return steps;
}
export const bosonFiniteResult=(response:BosonTeachingResponse):BosonFiniteResult=>response.kind==='boson-fock'?response.result:response.result.normalizedProjection;
