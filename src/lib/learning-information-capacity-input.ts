/** Reader-safe channel-family examples and a separately selected Bloch input. */
import {quantumInformationPresets} from './learning-quantum-information-presets';
import {readQuantumInformationInput,snapshotQuantumInformationInput,type QuantumInformationInput} from './learning-quantum-information-input';
export type CapacityInput=Extract<QuantumInformationInput,{kind:'capacity'}>;
const extra=(id:string,ko:string,en:string,probability:string,input:readonly[string,string,string]=['0','0','0'])=>Object.freeze({id,label:Object.freeze({ko,en}),input:snapshotQuantumInformationInput({kind:'capacity',probability,input})as CapacityInput});
export const capacityExamples=Object.freeze([
 ...quantumInformationPresets['quantum-channel-capacities-coding-limits'].map(p=>({...p,input:p.input as CapacityInput})),
 extra('complementary-noise','p=9/10 · 알려진 위상 반전 보정','p=9/10 · relabel the known phase flip','9/10'),
 extra('pure-identity','항등 채널·순수 입력 Ic=0','Identity channel · pure-input Ic=0','0',['0','0','1']),
 extra('mixed-z','편향 혼합 입력 · 환경 결맞음','Biased mixed input · environment coherence','1/10',['0','0','3/5']),
 extra('mixed-y','복소 Y 혼합 입력 · 입력별 정보','Complex Y mixed input · selected information','1/10',['0','3/5','1/5']),
 extra('pure-z','순수 Z 입력 · 무변화와 Ic=0','Pure Z input · unchanged state and Ic=0','1/10',['0','0','1']),
 extra('rotated-unitary','p=1 · 혼합 입력의 위상 회전','p=1 · phase rotation of a mixed input','1',['0','3/5','1/5']),
 extra('mixed-complete','p=1/2 · 혼합 입력의 결맞음 소실','p=1/2 · coherence loss in a mixed input','1/2',['0','3/5','1/5']),
 extra('tiny-below','1/2 아래 · 같은 작은 양자 용량','Below1/2 · the same tiny quantum capacity','999999999999999999999999999999/2000000000000000000000000000000'),
]);
export function readCapacityDraft(input:readonly string[],probability:string):CapacityInput|null{const value=readQuantumInformationInput({kind:'capacity',input:Array.from(input),probability});return value?.kind==='capacity'?value:null;}
