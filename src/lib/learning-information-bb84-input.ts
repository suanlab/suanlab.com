/** Reader-safe amplitudes and independent cloning/BB84 examples. */
import {quantumInformationPresets} from './learning-quantum-information-presets';
import {readQuantumInformationInput,snapshotQuantumInformationInput,type InformationState,type QuantumInformationInput} from './learning-quantum-information-input';
export type Bb84Input=Extract<QuantumInformationInput,{kind:'bb84'}>;
const s=(real:string,imaginary='0',root='0')=>({real:[real,root],imaginary:[imaginary,'0']}as const);
const zero:InformationState=[s('1'),s('0')],one:InformationState=[s('0'),s('1')],plus:InformationState=[s('1'),s('1')];
const extra=(id:string,ko:string,en:string,input:InformationState,comparison:InformationState,attackFraction='1',eveZProbability='1/2',sampleSize=20)=>Object.freeze({id,label:Object.freeze({ko,en}),input:snapshotQuantumInformationInput({kind:'bb84',input,comparison,attackFraction,eveZProbability,sampleSize})as Bb84Input});
export const bb84Examples=Object.freeze([
 ...quantumInformationPresets['no-cloning-bb84-information-disturbance'].map(p=>({...p,input:p.input as Bb84Input})),
 extra('basis-zero','기저0 · CNOT 후보 성공','Basis0 · CNOT candidate succeeds',zero,one),
 extra('basis-one','기저1 · CNOT 후보 성공','Basis1 · CNOT candidate succeeds',one,zero),
 extra('identical-rays','같은 ray · 복제 필요조건','Identical rays · cloning necessary condition',plus,[s('2'),s('2')]),
 extra('complex-input','복소 위상 · 원하는 복제와 비교','Complex phase · compare the desired clone',[s('3/5'),s('0','4/5')],zero),
 extra('raw-scale','비정규화 진폭 · 같은 상태','Unnormalized amplitudes · same state',[s('3'),s('0','4')],zero),
 extra('root-amplitudes','√2 진폭 · 정확 계수','Square-root amplitudes · exact coefficients',[s('1','0','1'),s('0','1')],zero),
 extra('eve-x','X만 고르는 Eve · 반대 기저 교란','Eve chooses only X · opposite-basis disturbance',plus,zero,'1','0'),
 extra('tiny-attack','아주 작은 공격 · 표본256개','Tiny attack ·256 test positions',plus,zero,'1/1000000000000000000000000000000','1/2',256),
]);
export const informationStateDraft=(state:InformationState)=>state.flatMap(z=>[...z.real,...z.imaginary]);
export function readBb84Draft(input:readonly string[],comparison:readonly string[],attackFraction:string,eveZProbability:string,sample:string):Bb84Input|null{
 if(input.length!==8||comparison.length!==8||!/^(?:0|[1-9]\d{0,2})$/.test(sample))return null;
 const state=(xs:readonly string[])=>[0,4].map(i=>({real:[xs[i],xs[i+1]],imaginary:[xs[i+2],xs[i+3]]}));
 const value=readQuantumInformationInput({kind:'bb84',input:state(input),comparison:state(comparison),attackFraction,eveZProbability,sampleSize:Number(sample)});return value?.kind==='bb84'?value:null;
}
