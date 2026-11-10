/** Reader-safe inputs for three distinct repetition-code experiments. */
import {quantumInformationPresets} from './learning-quantum-information-presets';
import {readQuantumInformationInput,snapshotQuantumInformationInput,type InformationState,type QuantumInformationInput} from './learning-quantum-information-input';
export type ErrorCorrectionInput=Extract<QuantumInformationInput,{kind:'error-correction'}>;
const s=(real:string,imaginary='0')=>({real:[real,'0'],imaginary:[imaginary,'0']}as const);
const complex:InformationState=[s('3/5'),s('0','4/5')],family=['III','XII','IXI','IIX'];
const extra=(id:string,ko:string,en:string,selectedError:string,physicalXProbability='1/10',errors:readonly string[]=family,input:InformationState=complex)=>Object.freeze({id,label:Object.freeze({ko,en}),input:snapshotQuantumInformationInput({kind:'error-correction',input,selectedError,physicalXProbability,errors})as ErrorCorrectionInput});
export const errorCorrectionExamples=Object.freeze([
 ...quantumInformationPresets['quantum-error-correction-syndrome-knill-laflamme'].map(p=>({...p,input:p.input as ErrorCorrectionInput})),
 extra('middle-x','중간 X · syndrome11','Middle X · syndrome11','IXI'),
 extra('last-x','마지막 X · syndrome01','Last X · syndrome01','IIX'),
 extra('known-z','알려진 Z 하나 · 가족 조건 통과','One known Z · family condition passes','ZII','1/10',['ZII']),
 extra('identity-added','퇴화 Z 가족에 항등 추가','Add identity to the degenerate Z family','ZII','1/10',['III','ZII','IZI','IIZ']),
 extra('y-error','Y 오류 · 위상이 남는 복구','Y error · phase remains after recovery','YII'),
 extra('high-noise','물리 X 확률3/4 · 부호가 더 나쁨','Physical X probability3/4 · code does worse','III','3/4'),
 extra('complete-noise','물리 X 확률1 · 논리 X','Physical X probability1 · logical X','III','1'),
 extra('tiny-noise','아주 작은 오류 · 논리 확률 보존','Tiny noise · preserve logical probability','III','1/1000000000000000000000000000000'),
 extra('zero-noise','IID 오류 없음 · 선택 Z는 별도','No IID noise · selected Z is separate','ZII','0'),
 extra('maximum-family','16연산자 가족 · 모든256쌍','Sixteen operators · all256 pairs','XII','1/10',Array.from({length:4},()=>family).flat()),
]);
export const errorCorrectionStateDraft=(state:InformationState)=>state.flatMap(z=>[...z.real,...z.imaginary]);
export function readErrorCorrectionDraft(input:readonly string[],selectedError:string,physicalXProbability:string,errors:string):ErrorCorrectionInput|null{
 if(input.length!==8)return null;const words=errors.trim()?errors.trim().split(/[\s,]+/):[];
 const value=readQuantumInformationInput({kind:'error-correction',input:[0,4].map(i=>({real:[input[i],input[i+1]],imaginary:[input[i+2],input[i+3]]})),selectedError,physicalXProbability,errors:words});return value?.kind==='error-correction'?value:null;
}
