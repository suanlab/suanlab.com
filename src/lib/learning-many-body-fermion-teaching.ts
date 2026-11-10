/** Reader-safe Fock experiments and display traces from verified operator columns. */
import {createFermionFockParameters as create,fermionFockParameterKeys} from './learning-fermion-fock-input';
import {snapshotManyBodyInput,readManyBodyInput,type ManyBodyInput} from './learning-many-body-input';
import type {ManyBodySuccess} from './learning-many-body-protocol';
export const fermionLearningId='fermion-fock-parity-operator-algebra' as const;
export const fermionText=(ko:string,en:string)=>({ko,en});
export type FermionTeachingResult=Extract<ManyBodySuccess,{kind:'fermion-fock'}>['result'];
const input=(p:Record<string,number>)=>snapshotManyBodyInput({kind:'fermion-fock',parameters:p});
const vacuum=create();
const one={...create(2,2),stateReal1:0,stateReal2:1,hoppingReal01:1};
const paired={...create(),pairingReal01:1};
const evenCat={...paired,stateReal3:1};
const formal={...create(1,0),stateReal1:1,onsite0:1};
const sectorGround={...create(),stateReal0:0,stateReal3:1,onsite0:-2,onsite1:1};
const complex={...create(2,2),stateImag2:1,hoppingImag01:1};
const prefix={...create(4,2),stateReal1:0,stateReal14:1,leftMode:0,leftCreation:0,rightMode:3,rightCreation:1,onsite0:1};
export const fermionTeachingPresets=[
 {id:'vacuum-order',label:fermionText('진공 · 생성 순서','Vacuum · creation order'),input:input(vacuum)},
 {id:'number-hopping',label:fermionText('한 입자 · hopping','One particle · hopping'),input:input(one)},
 {id:'pairing-vacuum',label:fermionText('진공 · pairing','Vacuum · pairing'),input:input(paired)},
 {id:'even-number-coherence',label:fermionText('짝수 · 다른 입자 수의 중첩','Even · number superposition'),input:input(evenCat)},
 {id:'formal-parity-coherence',label:fermionText('형식적 parity 중첩','Formal parity superposition'),input:input(formal)},
 {id:'sector-ground',label:fermionText('짝수 바닥과 전체 바닥','Even-sector versus full ground'),input:input(sectorGround)},
 {id:'complex-hopping',label:fermionText('복소 hopping 위상','Complex hopping phase'),input:input(complex)},
 {id:'long-parity-prefix',label:fermionText('네 모드 · 긴 parity string','Four modes · long parity string'),input:input(prefix)},
] as const;
export function fermionTeachingInput(i=0):ManyBodyInput<'fermion-fock'>{if(!Number.isInteger(i)||i<0||i>=fermionTeachingPresets.length)throw new RangeError('Unknown fermion teaching preset');return snapshotManyBodyInput(fermionTeachingPresets[i].input);}
export function editFermionTeachingField(value:ManyBodyInput,key:string,raw:string):ManyBodyInput<'fermion-fock'>|null{
 if(value.kind!=='fermion-fock'||!fermionFockParameterKeys.includes(key)||!/^(-?)(0|[1-9]\d*)$/.test(raw)||raw==='-0'||raw.length>8)return null;
 const n=Number(raw);if(!Number.isSafeInteger(n)||Math.abs(n)>1000000)return null;const next=readManyBodyInput({kind:'fermion-fock',parameters:{...value.parameters,[key]:n}});return next?.kind==='fermion-fock'?next as ManyBodyInput<'fermion-fock'>:null;
}
export function fermionTeachingFields(value:ManyBodyInput<'fermion-fock'>){const n=value.parameters.modes;
 return fermionFockParameterKeys.filter(key=>{const site=/^onsite(\d)$/.exec(key),pair=/^(?:hopping|pairing)(?:Real|Imag)(\d)(\d)$/.exec(key)||/^density(\d)(\d)$/.exec(key),state=/^state(?:Real|Imag)(\d+)$/.exec(key);return site?Number(site[1])<n:pair?Number(pair[2])<n:state?Number(state[1])<2**n:!['modes','sector','leftMode','rightMode','leftCreation','rightCreation'].includes(key);}).map(key=>{
  const site=/^onsite(\d)$/.exec(key),pair=/^(hopping|pairing)(Real|Imag)(\d)(\d)$/.exec(key),density=/^density(\d)(\d)$/.exec(key),state=/^state(Real|Imag)(\d+)$/.exec(key);
  const label=site?fermionText(`모드 ${site[1]} onsite 분자`,`Mode ${site[1]} onsite numerator`):pair?fermionText(`${pair[3]}–${pair[4]} ${pair[1]} ${pair[2]==='Real'?'실수':'허수'} 분자`,`${pair[3]}–${pair[4]} ${pair[1]} ${pair[2]==='Real'?'real':'imaginary'} numerator`):density?fermionText(`${density[1]}–${density[2]} 점유 상호작용 분자`,`${density[1]}–${density[2]} density interaction numerator`):state?fermionText(`|${Number(state[2]).toString(2).padStart(n,'0')}〉 ${state[1]==='Real'?'실수':'허수'} 진폭`,`|${Number(state[2]).toString(2).padStart(n,'0')}〉 ${state[1]==='Real'?'real':'imaginary'} amplitude`):key==='offset'?fermionText('상수 에너지 분자','Constant energy numerator'):fermionText('Hamiltonian 공통 양의 분모','Positive common Hamiltonian denominator');return{key,label,value:value.parameters[key]};
 });
}
export type FermionTraceStep={basis:number|null;sign:number;mode:number|null;creation:boolean|null;prefixOnes:number|null;blocked:boolean};
/** Creation is the transpose of the already-verified annihilation map.
 * This illustrates a selected basis component, not a qubit partial trace.
 */
export function fermionBasisTrace(r:FermionTeachingResult,basis:number,reverse=false):FermionTraceStep[]{
 if(!Number.isInteger(basis)||basis<0||basis>=r.dimension)throw new RangeError('Active basis index required');
 const p=r.parameters,operators=reverse?[[p.leftMode,p.leftCreation],[p.rightMode,p.rightCreation]]:[[p.rightMode,p.rightCreation],[p.leftMode,p.leftCreation]];
 const steps:FermionTraceStep[]=[{basis,sign:1,mode:null,creation:null,prefixOnes:null,blocked:false}];
 for(const[mode,creation]of operators){const prior=steps.at(-1)!,columns=r.operators[mode].annihilationColumns;
  if(prior.basis===null){steps.push({basis:null,sign:0,mode,creation:!!creation,prefixOnes:null,blocked:true});continue;}
  const count=[...r.basisLabels[prior.basis].slice(0,mode)].filter(b=>b==='1').length;
  const row=creation?columns.findIndex(v=>v.row===prior.basis):columns[prior.basis].row;
  const index=row===null||row<0?null:row,sign=index===null?0:creation?columns[index].sign:columns[prior.basis].sign;
  steps.push({basis:index,sign:index===null?0:prior.sign*sign,mode,creation:!!creation,prefixOnes:count,blocked:index===null});
 }
 return steps;
}
