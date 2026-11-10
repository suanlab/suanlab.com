/** Reader-safe examples and draft ownership for the discrimination lesson. */
import {quantumInformationPresets} from './learning-quantum-information-presets';
import {readQuantumInformationInput,snapshotQuantumInformationInput,type QuantumInformationInput} from './learning-quantum-information-input';
export type DiscriminationInput=Extract<QuantumInformationInput,{kind:'discrimination'}>;
const extra=(id:string,ko:string,en:string,bloch0:readonly[string,string,string],bloch1:readonly[string,string,string])=>Object.freeze({id,label:Object.freeze({ko,en}),input:snapshotQuantumInformationInput({kind:'discrimination',bloch0,bloch1,prior:'1/2'})as DiscriminationInput});
export const discriminationExamples=Object.freeze([
 ...quantumInformationPresets['binary-qubit-state-discrimination'].map(p=>({...p,input:p.input as DiscriminationInput})),
 extra('orthogonal','직교 순수 상태 · 완전 판별','Orthogonal pure states · perfect discrimination',['0','0','1'],['0','0','-1']),
 extra('complex-y','복소 Y 상태 · 측정축 회전','Complex Y states · rotate the measurement axis',['0','1','0'],['0','-1','0']),
 extra('mixed-pair','혼합 상태 · 일부 정보만','Mixed states · partial information',['0','0','1/5'],['0','0','-1/5']),
]);
export function readDiscriminationDraft(bloch0:readonly string[],bloch1:readonly string[],prior:string):DiscriminationInput|null {
 const input=readQuantumInformationInput({kind:'discrimination',bloch0:Array.from(bloch0),bloch1:Array.from(bloch1),prior});return input?.kind==='discrimination'?input:null;
}
