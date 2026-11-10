/** Reader-safe exact inputs. Ranks remain decimal strings, including2^64. */
import {quantumInformationPresets} from './learning-quantum-information-presets';
import {readQuantumInformationInput,snapshotQuantumInformationInput,type QuantumInformationInput} from './learning-quantum-information-input';
export type InformationEntropyInput=Extract<QuantumInformationInput,{kind:'entropy'}>;
const example=(id:string,ko:string,en:string,values:Omit<InformationEntropyInput,'kind'>)=>Object.freeze({id,label:Object.freeze({ko,en}),input:snapshotQuantumInformationInput({kind:'entropy',...values})as InformationEntropyInput});
const base={correlationWeight:'1',coherence:'1',probabilityOne:'1/5',copies:2,retainedRank:'3'};
export const informationEntropyExamples=Object.freeze([
 ...quantumInformationPresets['quantum-entropy-compression-conditional-information'].map(p=>({...p,input:p.input as InformationEntropyInput})),
 example('separable-coherence','얽힘 없는 결맞음','Coherence without entanglement',{...base,correlationWeight:'1/5'}),
 example('rank-two','rank2 · 1큐비트와80% 가중치','Rank2 · one qubit and80% weight',{...base,retainedRank:'2'}),
 example('rank-one-pure','순수 정보원 · rank1은0큐비트','Pure source · rank1 needs zero qubits',{...base,probabilityOne:'0',copies:4,retainedRank:'1'}),
 example('uniform-half','균일 정보원 · 절반의 부분공간','Uniform source · half the space',{...base,probabilityOne:'1/2',copies:4,retainedRank:'8'}),
 example('full-block-64','64복사본 · 전체 보존','64copies · retain the full block',{...base,probabilityOne:'1/2',copies:64,retainedRank:'18446744073709551616'}),
 example('rank-before-rounding','rank2⁵³ · 메모리53큐비트','Rank2^53 ·53qubits',{...base,probabilityOne:'1/2',copies:54,retainedRank:'9007199254740992'}),
 example('rank-after-rounding','rank2⁵³+1 · 메모리54큐비트','Rank2^53+1 ·54qubits',{...base,probabilityOne:'1/2',copies:54,retainedRank:'9007199254740993'}),
]);
export function readInformationEntropyDraft(correlationWeight:string,coherence:string,probabilityOne:string,copies:string,retainedRank:string):InformationEntropyInput|null {
 if(!/^(?:[1-9]|[1-5]\d|6[0-4])$/.test(copies))return null;
 const input=readQuantumInformationInput({kind:'entropy',correlationWeight,coherence,probabilityOne,copies:Number(copies),retainedRank});return input?.kind==='entropy'?input:null;
}
