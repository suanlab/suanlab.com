/** Reader-safe exact inputs for orbit and anchor comparisons. */
import {readGaugeOrbitInput,type GaugeOrbitInput} from './learning-gauge-orbits-input';
const p=(cayley='0',minusOne=false)=>({cayley,minusOne}),z=(real='1',imaginary='0')=>({real,imaginary}),ones=(n=3)=>Array.from({length:n},()=>z()),zeros=(n=3)=>Array.from({length:n},()=>z('0'));
const triangle=[{from:0,to:1},{from:1,to:2},{from:2,to:0}],unitLinks=Array.from({length:3},()=>p());
const base:GaugeOrbitInput={sites:3,edges:triangle,leftLinks:unitLinks,rightLinks:[p('-1'),p('-1'),p('0',true)],leftMatter:ones(),rightMatter:[z(),z('0','1'),z('-1')],forestEdges:[0,1],roots:[0],anchors:[]};
function example(id:string,ko:string,en:string,changes:Partial<GaugeOrbitInput>={}){const input=readGaugeOrbitInput({...base,...changes});if(!input)throw Error(`Invalid orbit example ${id}`);return Object.freeze({id,label:Object.freeze({ko,en}),input});}
export const gaugeOrbitExamples=Object.freeze([
 example('same-orbit-triangle','동일 궤도의 두 삼각형','Two triangles on the same orbit'),
 example('one-compatible-anchor','노드0을 고정한 같은 상태','The same state with site0 anchored',{anchors:[0]}),
 example('two-incompatible-anchors','두 경계 기준이 드러내는 차이','Two anchors reveal a difference',{anchors:[0,1]}),
 example('anchor-away-from-root','선택 루트와 다른 위치의 기준','An anchor away from the chosen root',{anchors:[1]}),
 example('global-phase-becomes-observable','공통 물질 위상과 외부 기준','Common matter phase and an external reference',{rightLinks:unitLinks,rightMatter:ones().map(()=>z('0','1')),anchors:[0]}),
 example('opposite-flux','같은 실수부·다른 순환 곱','Equal real parts,different cycle products',{leftLinks:[p('1'),p(),p()],rightLinks:[p('-1'),p(),p()],leftMatter:zeros(),rightMatter:zeros()}),
 example('different-amplitudes','링크는 같지만 물질 크기가 다르다','Equal links,different matter magnitudes',{rightLinks:unitLinks,rightMatter:[z('2'),z(),z()]}),
 example('different-relative-phases','밀도는 같지만 상대 위상이 다르다','Equal densities,different relative phases',{rightLinks:unitLinks,rightMatter:[z(),z('0','1'),z()]}),
 example('zero-matter-stabilizer','영 물질과 남는 안정자','Zero matter and a surviving stabilizer',{rightLinks:unitLinks,leftMatter:zeros(),rightMatter:zeros()}),
 example('anchored-open-chain','순환이 없는 사슬의 경계 정보','Boundary information on a loop-free chain',{edges:triangle.slice(0,2),leftLinks:[p(),p()],rightLinks:[p('1'),p()],leftMatter:zeros(),rightMatter:zeros(),anchors:[0,2]}),
 example('compatible-anchor-path','양끝 기준과 양립하는 트리 고정','Tree fixing compatible with endpoint anchors',{edges:triangle.slice(0,2),leftLinks:[p('1'),p('-1')],rightLinks:[p(),p()],leftMatter:zeros(),rightMatter:zeros(),anchors:[0,2]}),
 example('disconnected-residuals','연결 성분별 잔여 위상','A residual phase per connected component',{sites:6,edges:[...triangle,...triangle.map(e=>({from:e.from+3,to:e.to+3}))],leftLinks:Array.from({length:6},()=>p()),rightLinks:Array.from({length:6},()=>p()),leftMatter:ones(6),rightMatter:[z('0','1'),z('0','1'),z('0','1'),z('-1'),z('-1'),z('-1')],forestEdges:[0,1,3,4],roots:[0,3],anchors:[0]}),
 example('parallel-and-self-loops','평행 간선·자기 루프·고립 노드','Parallel edges,self-loops and an isolated site',{edges:[{from:0,to:1},{from:0,to:1},{from:1,to:1}],leftLinks:[p(),p('1'),p('0',true)],rightLinks:[p(),p('1'),p('0',true)],leftMatter:[z(),z('0'),z('0')],rightMatter:[z(),z('0'),z('0')],forestEdges:[0],roots:[0,2],anchors:[2]}),
 example('isolated-sites','고립 노드의 독립 위상','Independent phases at isolated sites',{edges:[],leftLinks:[],rightLinks:[],leftMatter:[z(),z('0'),z('0','1')],rightMatter:[z('0','1'),z('0'),z('-1')],forestEdges:[],roots:[2,0,1],anchors:[1]}),
 example('alternative-tree-root','다른 트리와 루트의 같은 궤도','The same orbit with another tree and root',{forestEdges:[1,2],roots:[1]}),
 example('every-site-anchored','모든 노드의 기준이 고정된 경우','Every site anchored',{rightLinks:unitLinks,rightMatter:ones(),anchors:[0,1,2]}),
 example('rational-data','분수 위상과 복소 물질','Rational phases and complex matter',{leftLinks:[p('1/2'),p('-2/3'),p('0',true)],rightLinks:[p('1/2'),p('-2/3'),p('0',true)],leftMatter:[z('2/3','1/5'),z('-1/2','3/7'),z('0','-2')],rightMatter:[z('2/3','1/5'),z('-1/2','3/7'),z('0','-2')],anchors:[1]}),
]);
