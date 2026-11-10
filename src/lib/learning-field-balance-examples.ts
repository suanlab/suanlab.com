/** Reader-safe choices; local-flow mathematics is never imported here. */
import {readFieldBalanceInput} from './learning-field-balance-input';
import {scalarFieldExamples} from './learning-scalar-field-examples';
import type {ScalarFieldInput} from './learning-scalar-field-input';
const field=scalarFieldExamples[0].input;
function example(id:string,ko:string,en:string,change:Partial<ScalarFieldInput>={},share='1/2'){
 const input=readFieldBalanceInput({field:{...field,...change},share});if(!input)throw Error('Invalid field-balance preparation');return Object.freeze({id,label:Object.freeze({ko,en}),input});
}
export const fieldBalanceExamples=Object.freeze([
 example('symmetric-flow','대칭 분할 · 점별 에너지 흐름','Symmetric split · local energy flow'),
 example('left-share-zero','왼쪽 몫0 · 같은 전체 에너지','Zero left share · unchanged total energy',{},'0'),
 example('left-share-one','왼쪽 몫1 · 달라진 국소 흐름','Full left share · different local currents',{},'1'),
 example('left-share-third','왼쪽 몫1/3 · 정확한 유리수','One-third left share · exact fractions',{},'1/3'),
 example('fixed-instant-zero','고정 경계 · Q̇=0이어도 대칭 아님','Fixed boundary · zero Q̇ is not symmetry',{boundary:'dirichlet-zero'}),
 example('neumann-shift','무유속 경계 · 장 값의 균일 이동','Zero-flux boundary · uniform field-value shifts',{boundary:'neumann-zero'}),
 example('massive-shift-breaking','이차 퍼텐셜 · 서로 다른 두 생성자','Quadratic potential · two different generators',{quadratic:'1',position:['1','0','0']}),
 example('quartic-cyclic','사차 반례 · 순환 대칭과 Ṗh=3','Quartic counterexample · cyclic symmetry and Ṗh=3',{quartic:'1',position:['1','2','0'],velocity:['0','0','0']}),
 example('equilibrium-not-symmetry','평형 궤적 · 작용 전체의 대칭과 구분','Equilibrium trajectory · distinguish whole-action symmetry',{quadratic:'-1',quartic:'1',position:['1','1','1'],velocity:['0','0','0']}),
 example('uniform-velocity','변 에너지 변화0 · 흐름은 남는다','Zero edge-energy derivatives · currents remain',{velocity:['1','1','1']}),
 example('thirty-two-symmetric','32점 · 대칭 분할과 부분 구간','32 sites · symmetric allocation and subregions',{...scalarFieldExamples.at(-1)!.input}),
 example('thirty-two-asymmetric','32점 · 비대칭 분할의 같은 전역 수지','32 sites · same global balance under an asymmetric split',{...scalarFieldExamples.at(-1)!.input},'0'),
]);
