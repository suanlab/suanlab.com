/** Reader-safe preparations; calculations and independent validators stay separate. */
import {readScalarFieldInput,type ScalarFieldInput} from './learning-scalar-field-input';
const rest=['0','0','0'];
const base:ScalarFieldInput={boundary:'periodic',spacing:'1',speed:'1',quadratic:'0',quartic:'0',position:['1','0','-1'],velocity:['1','0','0']};
function example(id:string,ko:string,en:string,change:Partial<ScalarFieldInput>={}){const input=readScalarFieldInput({...base,...change});if(!input)throw new RangeError(`Invalid scalar-field preparation: ${id}`);return Object.freeze({id,label:Object.freeze({ko,en}),input});}
export const scalarFieldExamples=Object.freeze([
 example('three-site-power','세 점 · 에너지7/2와 상쇄하는 일률','Three sites · energy7/2 and cancelling powers'),
 example('same-shape-rest','같은 변위 · 다른 초기 속도','Same displacement · different initial velocity',{velocity:rest}),
 example('uniform-shift','균일 이동 · 자유장의 같은 힘','Uniform shift · unchanged free-field forces',{position:['6','5','4']}),
 example('half-spacing','격자 간격1/2 · 질량과 힘의 척도','Half spacing · mass and force scales',{spacing:'1/2'}),
 example('periodic-uniform','주기 경계 · 균일장의 영모드','Periodic boundary · uniform zero mode',{position:['1','1','1'],velocity:rest}),
 example('neumann-uniform','무유속 경계 · 셀 중심의 균일장','Zero-flux boundary · uniform cell-centered field',{boundary:'neumann-zero',position:['1','1','1'],velocity:rest}),
 example('fixed-uniform','고정값0 경계 · 끝점의 복원력','Fixed-zero boundary · endpoint restoring forces',{boundary:'dirichlet-zero',position:['1','1','1'],velocity:rest}),
 example('onsite-restoring','균일장에 작용하는 국소 힘','Local restoring force on a uniform field',{quadratic:'1',position:['1','1','1'],velocity:rest}),
 example('double-well-minimum','이중 우물 · 균일한 비영 평형','Double well · uniform nonzero equilibrium',{quadratic:'-1',quartic:'1',position:['1','1','1'],velocity:rest}),
 example('boundary-stabilized','국소 음의 곡률과 고정 경계의 안정화','Negative local curvature and fixed-boundary stabilization',{boundary:'dirichlet-zero',speed:'2',quadratic:'-1',position:['1','1','1'],velocity:rest}),
 example('tiny-quartic','작은 양의 사차 에너지 · 정확 분수','Tiny positive quartic energy · exact fractions',{quartic:'1',position:Array(3).fill('1/100000000000000000'),velocity:rest}),
 example('thirty-two-sites','32개 격자점 · 혼합 부호와 비선형 힘','32 sites · mixed signs and nonlinear forces',{spacing:'1/2',speed:'3/2',quadratic:'-2/3',quartic:'5/7',position:Array.from({length:32},(_,i)=>`${(i*7)%11-5}/6`).map(x=>x.startsWith('0/')?'0':x),velocity:Array.from({length:32},(_,i)=>`${(i*5)%7-3}/5`).map(x=>x.startsWith('0/')?'0':x)}),
]);
