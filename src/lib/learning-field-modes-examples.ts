/** Reader-only presets: no interval, Fourier or matrix engine imports. */
import {readFieldModesInput,type FieldModesInput} from './learning-field-modes-input';
const base={boundary:'periodic',length:'4',speed:'1',quadratic:'0',quartic:'0',background:'0',position:['1','0','-1','0'],velocity:['1','0','0','0'],selectedBasis:1}as const;
function example(id:string,ko:string,en:string,change:Partial<FieldModesInput>={}){const input=readFieldModesInput({...base,...change});if(!input)throw Error('Invalid mode preparation');return Object.freeze({id,label:Object.freeze({ko,en}),input});}
export const fieldModesExamples=Object.freeze([
 example('four-site-cosine','4점 주기 · 코사인 모드','Four periodic sites · cosine mode'),
 example('four-site-sine','같은 상태 · 사인 성분','Same state · sine component',{selectedBasis:2}),
 example('three-site-pair','3점 주기 · 중복 고유값','Three periodic sites · repeated eigenvalue',{length:'3',position:['1','0','-1'],velocity:['1','0','0']}),
 example('periodic-nyquist','Nyquist 모드 · 군속도0','Nyquist mode · zero group speed',{selectedBasis:3}),
 example('fixed-length-dirichlet','같은 길이 · 고정 경계','Same length · fixed boundaries',{boundary:'dirichlet-zero',selectedBasis:0}),
 example('cell-centered-neumann','셀 중심 무유속 · 코사인 기저','Cell-centered zero flux · cosine basis',{boundary:'neumann-zero'}),
 example('boundary-stabilized','음의 국소 곡률 · 짧은 고정 구간','Negative local curvature · short fixed interval',{boundary:'dirichlet-zero',length:'4',speed:'2',quadratic:'-1',position:['1','0','-1'],velocity:['1','0','0'],selectedBasis:0}),
 example('long-interval-instability','같은 계수 · 긴 구간의 불안정 모드','Same coefficients · unstable mode on a longer interval',{boundary:'dirichlet-zero',length:'8',speed:'2',quadratic:'-1',position:['1','0','-1'],velocity:['1','0','0'],selectedBasis:0}),
 example('zero-curvature-drift','영 고유값 · 일정 속도 이동','Zero eigenvalue · constant-velocity drift',{position:['1','1','1','1'],velocity:['1','1','1','1'],selectedBasis:0}),
 example('zero-middle-mode','고정 경계 · 정확한 중간 영 모드','Fixed boundaries · exact middle zero mode',{boundary:'dirichlet-zero',length:'4',quadratic:'-2',position:['1','0','-1'],velocity:['1','0','0'],selectedBasis:1}),
 example('double-well-reference','이중 우물 평형 · 비선형 나머지','Double-well equilibrium · nonlinear remainder',{quadratic:'-1',quartic:'1',background:'1',position:['2','1','0','1']}),
 example('nonstationary-reference','비평형 기준장 · 곡률과 동역학 구별','Nonstationary reference · curvature versus dynamics',{quartic:'1',background:'1',position:['2','1','0','1']}),
 example('fixed-reference-force','국소 평형이어도 고정 벽의 힘이 남는다','Local equilibrium still feels fixed walls',{boundary:'dirichlet-zero',length:'5',quadratic:'-1',quartic:'1',background:'1',position:['1','1','1','1'],velocity:['0','0','0','0'],selectedBasis:0}),
 example('thirty-two-components','32점 · 모드 분해와 재구성','32 sites · decomposition and reconstruction',{position:Array.from({length:32},(_,i)=>`${i%7-3}/${i+1}`),velocity:Array.from({length:32},(_,i)=>`${i%3-1}/${i+2}`),selectedBasis:15}),
 example('thirty-two-nyquist','32점 Nyquist · 고정 모드와 구별','32-site Nyquist · distinguish a fixed physical mode',{position:Array.from({length:32},(_,i)=>i%2?'-1':'1'),velocity:Array.from({length:32},()=> '0'),selectedBasis:31}),
]);
