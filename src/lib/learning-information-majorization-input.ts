/** Reader-safe Schmidt-probability and explicit Bell-dilution drafts. */
import {quantumInformationPresets} from './learning-quantum-information-presets';
import {readQuantumInformationInput,snapshotQuantumInformationInput,type QuantumInformationInput} from './learning-quantum-information-input';
export type MajorizationInput=Extract<QuantumInformationInput,{kind:'majorization'}>;
const compare=(id:string,ko:string,en:string,source:readonly string[],target:readonly string[])=>Object.freeze({id,label:Object.freeze({ko,en}),input:snapshotQuantumInformationInput({kind:'majorization',mode:'compare',source,target})as MajorizationInput});
const dilute=(id:string,ko:string,en:string,a:string,b:string)=>Object.freeze({id,label:Object.freeze({ko,en}),input:snapshotQuantumInformationInput({kind:'majorization',mode:'bell-dilution',a,b})as MajorizationInput});
export const majorizationExamples=Object.freeze([
 ...quantumInformationPresets['pure-state-entanglement-majorization-resources'].map(p=>({...p,input:p.input as MajorizationInput})),
 compare('permuted-equal','순서만 다른 같은 스펙트럼','Same spectrum in a different input order',['1/4','1/2','1/4'],['1/2','1/4','1/4']),
 compare('zero-padded','명시적0과 길이가 다른 벡터','Explicit zeros and different vector lengths',['1/2','1/2','0'],['1']),
 compare('uniform-sixteen','16차원 균일 상태에서 곱 상태로','Sixteen-dimensional uniform state to product',Array(16).fill('1/16'),['1']),
 compare('tiny-violation','반올림으로 가려진 prefix 위반','Prefix violation hidden by rounding',['500000000000000000000000000001/1000000000000000000000000000000','499999999999999999999999999999/1000000000000000000000000000000'],['1/2','1/2']),
 compare('tiny-success','거의 곱 상태에서 Bell로 · 작은 성공률','Nearly product to Bell · tiny success',['999999999999999999999999999999/1000000000000000000000000000000','1/1000000000000000000000000000000'],['1/2','1/2']),
 dilute('bell-endpoint-zero','Bell 희석 · a=0,b=1','Bell dilution · a=0,b=1','0','1'),
 dilute('bell-endpoint-one','Bell 희석 · a=1,b=0','Bell dilution · a=1,b=0','1','0'),
 dilute('swapped-amplitudes','Bell 희석 · a,b 맞바꾸기','Bell dilution · swapped amplitudes','3/5','4/5'),
 compare('bell-target-comparison','Bell 희석 목표의 확정 변환 조건','Deterministic criterion for the dilution target',['1/2','1/2'],['16/25','9/25']),
]);
export type MajorizationDraft={mode:'compare'|'bell-dilution';source:string;target:string;a:string;b:string};
export function majorizationDraft(input:MajorizationInput):MajorizationDraft{return input.mode==='compare'?{mode:input.mode,source:input.source.join(', '),target:input.target.join(', '),a:'4/5',b:'3/5'}:{mode:input.mode,source:'1/2, 1/2',target:'16/25, 9/25',a:input.a,b:input.b};}
export function readMajorizationDraft(draft:MajorizationDraft):MajorizationInput|null{
 const list=(s:string)=>s.trim()?s.trim().split(/[\s,]+/):[];
 const input=readQuantumInformationInput(draft.mode==='compare'?{kind:'majorization',mode:draft.mode,source:list(draft.source),target:list(draft.target)}:{kind:'majorization',mode:draft.mode,a:draft.a,b:draft.b});return input?.kind==='majorization'?input:null;
}
