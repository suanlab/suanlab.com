/** Reader-safe ensemble examples; numerical evaluation belongs to the Workers. */
import {quantumInformationPresets} from './learning-quantum-information-presets';
import {readQuantumInformationInput,snapshotQuantumInformationInput,type QuantumInformationInput} from './learning-quantum-information-input';
export type EnsembleInput=Extract<QuantumInformationInput,{kind:'ensemble'}>;
const extra=(id:string,ko:string,en:string,bloch0:readonly[string,string,string],bloch1:readonly[string,string,string],prior='1/2',measurement:'z'|'helstrom'='z')=>Object.freeze({id,label:Object.freeze({ko,en}),input:snapshotQuantumInformationInput({kind:'ensemble',bloch0,bloch1,prior,measurement})as EnsembleInput});
export const ensembleExamples=Object.freeze([
 ...quantumInformationPresets['classical-quantum-ensembles-holevo-information'].map(p=>({...p,input:p.input as EnsembleInput})),
 extra('orthogonal-z','직교 Z 상태 · 한 bit','Orthogonal Z states · one bit',['0','0','1'],['0','0','-1']),
 extra('same-mixed','같은 혼합 상태 · 같은 평균','Identical mixed states · the same average',['0','0','0'],['0','0','0']),
 extra('same-pure','같은 순수 상태 · 정보 없음','Identical pure states · no information',['0','0','1'],['0','0','1']),
 extra('certain-label','확실한 레이블 · 불확실성 없음','Certain label · no uncertainty',['0','0','1'],['0','0','-1'],'1'),
 extra('y-z','직교 Y 상태 · Z로 측정','Orthogonal Y states · measure Z',['0','1','0'],['0','-1','0']),
 extra('y-helstrom','직교 Y 상태 · Helstrom 기록','Orthogonal Y states · Helstrom record',['0','1','0'],['0','-1','0'],'1/2','helstrom'),
 extra('mixed-z','혼합 Z 쌍 · 작은 정보','Mixed Z pair · limited information',['0','0','1/5'],['0','0','-1/5']),
 extra('tiny-z','거의 같은 상태 · 아주 작은 정보','Nearly identical states · tiny information',['0','0','0'],['0','0','1/1000000000000000000000000000000']),
]);
export function readEnsembleDraft(bloch0:readonly string[],bloch1:readonly string[],prior:string,measurement:string):EnsembleInput|null{
 const input=readQuantumInformationInput({kind:'ensemble',bloch0:Array.from(bloch0),bloch1:Array.from(bloch1),prior,measurement});return input?.kind==='ensemble'?input:null;
}
