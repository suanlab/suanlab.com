/** Reader-safe examples and draft ownership for ordered relative entropy. */
import {quantumInformationPresets} from './learning-quantum-information-presets';
import {readQuantumInformationInput,snapshotQuantumInformationInput,type QuantumInformationInput} from './learning-quantum-information-input';
export type RelativeInput=Extract<QuantumInformationInput,{kind:'relative-entropy'}>;
const extra=(id:string,ko:string,en:string,rho:readonly[string,string,string],sigma:readonly[string,string,string])=>Object.freeze({id,label:Object.freeze({ko,en}),input:snapshotQuantumInformationInput({kind:'relative-entropy',rho,sigma})as RelativeInput});
export const relativeExamples=Object.freeze([
 ...quantumInformationPresets['quantum-relative-entropy-data-processing'].map(p=>({...p,input:p.input as RelativeInput})),
 extra('reverse-diagonal','대각 예제의 순서 반전','Reverse the diagonal example',['0','0','0'],['0','0','3/5']),
 extra('opposite-pure','직교 순수 상태 · 무한대 유지','Orthogonal pure states · infinity remains',['0','0','1'],['0','0','-1']),
 extra('equal-pure','같은 순수 상태 · 정확한0','Equal pure states · exact zero',['0','0','1'],['0','0','1']),
 extra('mixed-pure','혼합 상태와 순수 참조','Mixed state and pure reference',['0','0','0'],['1','0','0']),
 extra('pure-mixed','순수 상태와 완전 혼합 참조','Pure state and maximally mixed reference',['1','0','0'],['0','0','0']),
 extra('erased-pair','서로 다른 상태가 같은 출력으로','Distinct states become the same output',['3/5','0','0'],['-3/5','0','0']),
 extra('complex-y','복소 Y 성분을 지우기','Erase complex Y components',['0','3/5','0'],['0','-3/5','0']),
 extra('near-pure','거의 순수하지만 전계수인 참조','Nearly pure but full-rank reference',['0','0','1'],['999999999999999999999999999999/1000000000000000000000000000000','0','0']),
 extra('tiny-finite','거의 같은 전계수 상태 · 작은 양수','Nearly identical full-rank states · tiny positive',['0','0','0'],['0','0','1/1000000000000000000000000000000']),
]);
export function readRelativeDraft(rho:readonly string[],sigma:readonly string[]):RelativeInput|null{
 const input=readQuantumInformationInput({kind:'relative-entropy',rho:Array.from(rho),sigma:Array.from(sigma)});return input?.kind==='relative-entropy'?input:null;
}
