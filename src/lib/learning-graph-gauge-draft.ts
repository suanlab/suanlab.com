/** Human-readable graph editor. Exact calculations never run in this module. */
import {readGraphGaugeInput,type GraphGaugeInput} from './learning-graph-gauge-input';
export type GraphGaugeDraft={matter:string;frames:string;edges:string;walks:string;quadratic:string;quartic:string;loopCoupling:string};
export function graphGaugeDraft(p:GraphGaugeInput):GraphGaugeDraft{return{matter:p.matter.map(z=>`${z.real} ${z.imaginary}`).join('\n'),frames:p.frames.map(g=>`${g.cayley} ${g.minusOne}`).join('\n'),edges:p.edges.map(e=>`${e.from} ${e.to} ${e.cayley} ${e.minusOne} ${e.weight}`).join('\n'),walks:p.walks.map(w=>`${w.start}: ${w.steps.map(s=>`${s.forward?'+':'-'}${s.edge}`).join(' ')}`).join('\n'),quadratic:p.quadratic,quartic:p.quartic,loopCoupling:p.loopCoupling};}
export function readGraphGaugeDraft(d:GraphGaugeDraft):GraphGaugeInput|null{
 try{
  if(Object.values(d).some(s=>typeof s!=='string'||s.length>16000))return null;
  const rows=(s:string,width:number)=>s.split(/\r?\n/).filter(s=>s.trim()).map(s=>{const cells=s.trim().split(/[\s,]+/);if(cells.length!==width)throw Error('Wrong column count');return cells;});
  const integer=(s:string)=>{if(!/^(?:0|[1-9]\d?)$/.test(s))throw Error('Integer index required');return Number(s);};
  const boolean=(s:string)=>{if(s!=='true'&&s!=='false')throw Error('Use true or false');return s==='true';};
  return readGraphGaugeInput({matter:rows(d.matter,2).map(([real,imaginary])=>({real,imaginary})),frames:rows(d.frames,2).map(([cayley,minusOne])=>({cayley,minusOne:boolean(minusOne)})),edges:rows(d.edges,5).map(([from,to,cayley,minusOne,weight])=>({from:integer(from),to:integer(to),cayley,minusOne:boolean(minusOne),weight})),walks:d.walks.split(/\r?\n/).filter(s=>s.trim()).map(s=>{const pair=s.trim().split(':');if(pair.length!==2)throw Error('Walk start: steps');const start=integer(pair[0].trim()),steps=pair[1].trim().split(/\s+/).map(s=>{const m=/^([+-])((?:0|[1-9]\d?))$/.exec(s);if(!m)throw Error('Signed edge required');return{edge:integer(m[2]),forward:m[1]==='+'};});return{start,steps};}),quadratic:d.quadratic,quartic:d.quartic,loopCoupling:d.loopCoupling});
 }catch{return null;}
}
