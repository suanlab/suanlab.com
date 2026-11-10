/** Reader-safe preparations and teaching IDs. No numerical or data-layer imports. */
import { snapshotQubitInput,readQubitInput,parseQubitInputDraft,parseQubitStoredInput,type QubitInput,type QubitWireScalar,type QubitWireMatrix,type QubitWireMap } from './learning-qubit-input';
export const qubitTeachingIds=['finite-qubit-circuit-order-interference','qubit-channel-kraus-choi-verification','system-environment-stinespring-reference','quantum-instruments-records-feedforward'] as const;
export type QubitTeachingId=typeof qubitTeachingIds[number];
export type QubitText={ko:string;en:string};
export const qubitText=(ko:string,en:string):QubitText=>({ko,en});
export const isQubitTeachingId=(value:string):value is QubitTeachingId=>(qubitTeachingIds as readonly string[]).includes(value);
export type QubitTeachingPreset={id:string;label:QubitText;input:QubitInput};
const w=(re:string,im='0',rootReal='0',rootImaginary='0'):QubitWireScalar=>({real:[re,rootReal],imaginary:[im,rootImaginary]});
const Z=w('0'),O=w('1'),I:QubitWireMatrix=[[O,Z],[Z,O]],X:QubitWireMatrix=[[Z,O],[O,Z]],PauliZ:QubitWireMatrix=[[O,Z],[Z,w('-1')]];
const zero:QubitWireMatrix=[[O,Z],[Z,Z]],one:QubitWireMatrix=[[Z,Z],[Z,O]],plus:QubitWireMatrix=[[w('1/2'),w('1/2')],[w('1/2'),w('1/2')]],plusI:QubitWireMatrix=[[w('1/2'),w('0','-1/2')],[w('0','1/2'),w('1/2')]],mixed:QubitWireMatrix=[[w('2/3'),w('0','1/6')],[w('0','-1/6'),w('1/3')]];
const K0:QubitWireMatrix=[[O,Z],[Z,w('3/5')]],K1:QubitWireMatrix=[[Z,w('4/5')],[Z,Z]];
const Lplus:QubitWireMatrix=[[w('0','0','1/2'),w('0','0','2/5')],[Z,w('0','0','3/10')]],Lminus:QubitWireMatrix=[[w('0','0','1/2'),w('0','0','-2/5')],[Z,w('0','0','3/10')]];
const term=(operator:QubitWireMatrix,weight='1')=>({weight,operator});
const operatorMap=(operators:QubitWireMatrix[]):QubitWireMap=>({kind:'operator-sum',terms:operators.map(a=>term(a))});
const preset=(id:string,ko:string,en:string,input:QubitInput):QubitTeachingPreset=>Object.freeze({id,label:qubitText(ko,en),input:snapshotQubitInput(input)});
const circuit=(names:string[])=>names.map(gate=>({gate,target:0})) as Extract<QubitInput,{kind:'circuit'}>['gates'];
const sequence:Extract<QubitInput,{kind:'circuit'}>['gates']=[{gate:'H',target:0},{gate:'T',target:2},{gate:'CX',control:2,target:0},{gate:'Y',target:1},{gate:'SWAP',target:0,other:1},{gate:'CZ',control:0,target:2},{gate:'Sdg',target:2},{gate:'Tdg',target:0}];
const rawThree=[w('-1'),w('0','1'),O,w('-1','1'),Z,w('1','1'),w('-1'),w('0','1')];
export const qubitTeachingPresets:Record<QubitTeachingId,readonly QubitTeachingPreset[]>={
 'finite-qubit-circuit-order-interference':[
  preset('order-hzh','H → Z → H','H → Z → H',{kind:'circuit',sites:1,gates:circuit(['H','Z','H']),input:[O,Z]}),
  preset('order-zhh','Z → H → H','Z → H → H',{kind:'circuit',sites:1,gates:circuit(['Z','H','H']),input:[O,Z]}),
  preset('phase-hth','H → T → H · 간섭','H → T → H · interference',{kind:'circuit',sites:1,gates:circuit(['H','T','H']),input:[O,Z]}),
  preset('bell','Bell · 공동 상태','Bell · joint state',{kind:'circuit',sites:2,gates:[{gate:'H',target:0},{gate:'CX',control:0,target:1}],input:[O,Z,Z,Z]}),
  preset('ghz','GHZ · 세 큐비트','GHZ · three qubits',{kind:'circuit',sites:3,gates:[{gate:'H',target:0},{gate:'CX',control:0,target:1},{gate:'CX',control:1,target:2}],input:[O,Z,Z,Z,Z,Z,Z,Z]}),
  preset('global-phase','|1〉에 T · 전체 위상','T on |1〉 · global phase',{kind:'circuit',sites:1,gates:[{gate:'T',target:0}],input:[Z,O]}),
  preset('positive-y','복소 위상 · Y=+1','Complex phase · Y=+1',{kind:'circuit',sites:1,gates:[],input:[O,w('0','1')]}),
  preset('irrational-norm','무리수 노름의 원시 입력','Raw input with irrational squared norm',{kind:'circuit',sites:1,gates:circuit(['H','T','Sdg']),input:[w('1','0','1'),w('0','1')]}),
  preset('long-circuit','16게이트 · 순서와 역회로','16 gates · order and inverse',{kind:'circuit',sites:3,gates:[...sequence,...sequence],input:rawThree}),
 ],
 'qubit-channel-kraus-choi-verification':[
  preset('identity','항등 · |+i〉','Identity · |+i〉',{kind:'channel',map:operatorMap([I]),state:plusI}),
  preset('damping','감쇠 γ=16/25 · |+i〉','Damping γ=16/25 · |+i〉',{kind:'channel',map:operatorMap([K0,K1]),state:plusI}),
  preset('damping-excited','감쇠 γ=16/25 · |1〉','Damping γ=16/25 · |1〉',{kind:'channel',map:operatorMap([K0,K1]),state:one}),
  preset('transpose','전치 · 참조계 검사','Transpose · reference-system test',{kind:'channel',map:{kind:'transpose'},state:plusI}),
  preset('phase-half','위상 뒤집기 p=1/2','Phase flip p=1/2',{kind:'channel',map:{kind:'operator-sum',terms:[term(I,'1/2'),term(PauliZ,'1/2')]},state:plusI}),
  preset('phase-one','위상 뒤집기 p=1','Phase flip p=1',{kind:'channel',map:{kind:'operator-sum',terms:[term(I,'0'),term(PauliZ)]},state:plusI}),
  preset('impossible-selection','|1〉에서 결과0만 선택','Select outcome0 from |1〉',{kind:'channel',map:operatorMap([zero]),state:one}),
  preset('trace-growing','가중치2 · trace 보존 실패','Weight2 · trace preservation fails',{kind:'channel',map:{kind:'operator-sum',terms:[term(I,'2')]},state:plusI}),
 ],
 'system-environment-stinespring-reference':[
  preset('no-decay','t=0 · |+i〉','t=0 · |+i〉',{kind:'environment',parameter:'0',state:plusI}),
  preset('partial-decay','t=1/2 · |+i〉','t=1/2 · |+i〉',{kind:'environment',parameter:'1/2',state:plusI}),
  preset('excitation','t=1/2 · |1〉','t=1/2 · |1〉',{kind:'environment',parameter:'1/2',state:one}),
  preset('ground','t=1/2 · |0〉','t=1/2 · |0〉',{kind:'environment',parameter:'1/2',state:zero}),
  preset('complete-transfer','t=1 · 정보의 환경 이동','t=1 · information moves to the environment',{kind:'environment',parameter:'1',state:plusI}),
  preset('mixed-input','t=1/2 · 혼합 입력','t=1/2 · mixed input',{kind:'environment',parameter:'1/2',state:mixed}),
  preset('larger-decay','t=2/3 · γ=144/169','t=2/3 · γ=144/169',{kind:'environment',parameter:'2/3',state:plusI}),
 ],
 'quantum-instruments-records-feedforward':[
  preset('projective-plus','Z 사영 · |+〉','Z projection · |+〉',{kind:'instrument',outcomes:[[term(zero)],[term(one)]],state:plus,correctZ:false}),
  preset('corrected-plus','Z 사영 + 결과별 보정','Z projection + conditional correction',{kind:'instrument',outcomes:[[term(zero)],[term(one)]],state:plus,correctZ:true}),
  preset('reset-plus','같은 POVM · 측정 후 초기화','Same POVM · measure and reset',{kind:'instrument',outcomes:[[term(zero)],[term([[Z,O],[Z,Z]])]],state:plus,correctZ:false}),
  preset('zero-branch','|0〉 · 확률0인 결과1','|0〉 · zero-probability outcome1',{kind:'instrument',outcomes:[[term(zero)],[term(one)]],state:zero,correctZ:false}),
  preset('damping-record','환경 계산 기저의 기록','Computational environment record',{kind:'instrument',outcomes:[[term(K0)],[term(K1)]],state:zero,correctZ:false}),
  preset('mixed-record','환경 ± 기저의 기록','Plus/minus environment record',{kind:'instrument',outcomes:[[term(Lplus)],[term(Lminus)]],state:zero,correctZ:false}),
  preset('coarse-record','여러 연산자를 묶은 결과','A coarse outcome with multiple operators',{kind:'instrument',outcomes:[[term(I,'1/2'),term(PauliZ,'1/4')],[term(X,'1/4')]],state:plusI,correctZ:false}),
  preset('deferred','결맞는 기록과 마지막 dephasing','Coherent record and final dephasing',{kind:'deferred',state:plusI}),
  preset('corrected-phase','|+i〉의 측정·보정','Measurement and correction of |+i〉',{kind:'instrument',outcomes:[[term(zero)],[term(one)]],state:plusI,correctZ:true}),
 ],
};
export const qubitExperimentPresets:Record<QubitTeachingId,readonly(readonly number[])[]>={
 'finite-qubit-circuit-order-interference':[[0,1],[2,5],[3,4]],
 'qubit-channel-kraus-choi-verification':[[0,1,2],[0,3],[4,5,6,7]],
 'system-environment-stinespring-reference':[[0,1,4],[2,3,5],[3]],
 'quantum-instruments-records-feedforward':[[0,3,6],[0,1,2],[4,5],[7,8]],
};
export function qubitTeachingInput(id:QubitTeachingId,index=0):QubitInput {const item=qubitTeachingPresets[id]?.[index];if(!Number.isInteger(index)||!item)throw new RangeError('Unknown qubit teaching preset');return snapshotQubitInput(item.input);}
export function readQubitTeachingInput(id:QubitTeachingId,value:unknown):QubitInput|null {
  const input=readQubitInput(value),allowed=id===qubitTeachingIds[0]?['circuit']:id===qubitTeachingIds[1]?['channel']:id===qubitTeachingIds[2]?['environment']:id===qubitTeachingIds[3]?['instrument','deferred']:[];
  return input&&allowed.includes(input.kind)?input:null;
}
export const parseQubitTeachingDraft=(id:QubitTeachingId,raw:string)=>readQubitTeachingInput(id,parseQubitInputDraft(raw));
export const parseQubitTeachingStoredInput=(id:QubitTeachingId,raw:string)=>readQubitTeachingInput(id,parseQubitStoredInput(raw));
