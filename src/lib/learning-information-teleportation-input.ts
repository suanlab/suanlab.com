/** Reader-safe examples for two protocols with separate messages and branch views. */
import {quantumInformationPresets} from './learning-quantum-information-presets';
import {readQuantumInformationInput,snapshotQuantumInformationInput,type InformationState,type QuantumInformationInput} from './learning-quantum-information-input';
export type TeleportationInput=Extract<QuantumInformationInput,{kind:'teleportation'}>;
const s=(real:string,imaginary='0')=>({real:[real,'0'],imaginary:[imaginary,'0']}as const);
const complex:InformationState=[s('3/5'),s('0','4/5')];
const extra=(id:string,ko:string,en:string,input:InformationState=complex,a=0,b=0)=>Object.freeze({id,label:Object.freeze({ko,en}),input:snapshotQuantumInformationInput({kind:'teleportation',input,a,b,order:'x-then-z'})as TeleportationInput});
export const teleportationExamples=Object.freeze([
 ...quantumInformationPresets['teleportation-superdense-coding-resources'].map(p=>({...p,input:p.input as TeleportationInput})),
 extra('message-00','메시지00 · Bell 부호','Message00 · Bell encoding'),
 extra('message-01','메시지01 · X 부호','Message01 · X encoding',complex,0,1),
 extra('message-10','메시지10 · Z 부호','Message10 · Z encoding',complex,1,0),
 extra('basis-one','입력1 · 네 가지 복원','Input1 · recovery in four branches',[s('0'),s('1')]),
 extra('plus-input','동일 중첩 입력 · 위상 복원','Equal superposition · phase recovery',[s('1'),s('1')]),
 extra('y-input','Y 순수 상태 · 허수 위상','Pure Y state · imaginary phase',[s('1'),s('0','1')]),
 extra('global-phase','전체 위상 i · 같은 밀도행렬','Global phase i · same density matrix',[s('0','3/5'),s('-4/5')],1,1),
]);
export const teleportationStateDraft=(state:InformationState)=>state.flatMap(z=>[...z.real,...z.imaginary]);
export function readTeleportationDraft(input:readonly string[],a:string,b:string,order:string):TeleportationInput|null{
 if(input.length!==8||!['0','1'].includes(a)||!['0','1'].includes(b))return null;
 const value=readQuantumInformationInput({kind:'teleportation',input:[0,4].map(i=>({real:[input[i],input[i+1]],imaginary:[input[i+2],input[i+3]]})),a:Number(a),b:Number(b),order});return value?.kind==='teleportation'?value:null;
}
