/** Preparations contain no numerical engine and can be imported by a reader. */
import {readGraphGaugeInput,type GraphGaugeInput,type GraphGaugePhase,type GraphGaugeEdge} from './learning-graph-gauge-input';
const phase=(cayley='0',minusOne=false):GraphGaugePhase=>({cayley,minusOne});
const edge=(from:number,to:number,cayley='0',minusOne=false):GraphGaugeEdge=>({from,to,cayley,minusOne,weight:'1'});
const matter=(n:number)=>Array.from({length:n},()=>({real:'1',imaginary:'0'}));
const loop={start:0,steps:[0,1,2].map(edge=>({edge,forward:true}))};
const base:GraphGaugeInput={matter:matter(3),frames:[phase(),phase('1'),phase('0',true)],edges:[edge(0,1),edge(1,2),edge(2,0)],walks:[loop],quadratic:'0',quartic:'0',loopCoupling:'1'};
function preparation(id:string,titleKo:string,titleEn:string,input:GraphGaugeInput){const owned=readGraphGaugeInput(input);if(!owned)throw Error(`Invalid preparation ${id}`);return Object.freeze({id,titleKo,titleEn,input:owned});}
const flux=(parameter:string):GraphGaugeInput=>({...base,edges:[edge(0,1,parameter),...base.edges.slice(1)]});
export const graphGaugeExamples=Object.freeze([
 preparation('flat-triangle','좌표만 바뀌는 삼각형','A triangle with changing frames',base),
 preparation('positive-flux','복소 루프 곱 +i','Complex loop product +i',flux('1')),
 preparation('negative-flux','같은 실수부와 다른 루프 곱','Same real part, different loop product',flux('-1')),
 preparation('reverse-walk','경로를 거꾸로 돌기','Reverse the closed walk',{...flux('1'),loopCoupling:'1/2',walks:[loop,{start:0,steps:[2,1,0].map(edge=>({edge,forward:false}))}]}),
 preparation('hidden-flux','왕복 경로가 놓치는 플럭스','Flux missed by a backtracking walk',{...flux('1'),matter:matter(3).map(()=>({real:'0',imaginary:'0'})),walks:[{start:0,steps:[{edge:0,forward:true},{edge:0,forward:false}]}]}),
 preparation('open-chain','루프 없는 사슬과 물질 에너지','A loop-free chain with matter energy',{...base,edges:[edge(0,1,'1'),edge(1,2,'0',true)],walks:[]}),
 preparation('common-frame','공통 위상은 링크를 바꾸지 않는다','A common phase leaves the links unchanged',{...flux('1'),frames:[phase('1'),phase('1'),phase('1')]}),
 preparation('disconnected','연결 성분별 공통 위상','A separate common phase per component',{...base,matter:matter(6),frames:[...Array.from({length:3},()=>phase('1')),...Array.from({length:3},()=>phase('0',true))],edges:[...base.edges,edge(3,4),edge(4,5),edge(5,3)],walks:[loop,{start:3,steps:[3,4,5].map(edge=>({edge,forward:true}))}]}),
 preparation('self-loop','자기 루프의 공변 차분','The covariant difference on a self-loop',{...base,matter:[{real:'1',imaginary:'0'},{real:'0',imaginary:'0'},{real:'0',imaginary:'0'}],edges:[edge(0,0,'1')],walks:[{start:0,steps:[{edge:0,forward:true}]}]}),
 preparation('parallel-edges','평행 간선이 만드는 루프','A loop formed by parallel edges',{...base,edges:[edge(0,1),edge(0,1,'1')],walks:[{start:0,steps:[{edge:0,forward:true},{edge:1,forward:false}]}]}),
 preparation('no-edges','고립된 노드의 국소 퍼텐셜','Local potentials on isolated sites',{...base,edges:[],walks:[],quadratic:'1/3',quartic:'2/5'}),
 preparation('rational-frames','정확한 분수 위상과 복소 물질','Exact rational phases and complex matter',{...base,frames:[phase('1/2'),phase('2'),phase('-1/3')],matter:[{real:'2/3',imaginary:'1/5'},{real:'-1/2',imaginary:'3/7'},{real:'0',imaginary:'-2'}],edges:[{...edge(0,1,'1/3'),weight:'2/7'},{...edge(1,2,'-2/5'),weight:'3/2'},edge(2,0,'0',true)],quadratic:'-1/3',quartic:'2/5',loopCoupling:'3/7'}),
 preparation('twelve-site-ring','12개 노드 고리','A twelve-site ring',{...base,matter:matter(12).map((z,i)=>({...z,real:String(i-5),imaginary:`1/${i+1}`})),frames:Array.from({length:12},(_,i)=>phase(`${i-6}/7`)),edges:Array.from({length:12},(_,i)=>edge(i,(i+1)%12,`${i-4}/5`)),walks:[{start:0,steps:Array.from({length:12},(_,edge)=>({edge,forward:true}))}],quadratic:'-1/3',quartic:'2/5',loopCoupling:'3/7'}),
]);
