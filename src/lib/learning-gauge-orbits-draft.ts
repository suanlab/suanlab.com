/** Reader-safe,explicit graph-pair editor; no exact arithmetic engine. */
import {readGaugeOrbitInput,type GaugeOrbitInput} from './learning-gauge-orbits-input';
export type GaugeOrbitDraft={sites:string;edges:string;leftLinks:string;rightLinks:string;leftMatter:string;rightMatter:string;forestEdges:string;roots:string;anchors:string};
export function gaugeOrbitDraft(p:GaugeOrbitInput):GaugeOrbitDraft{return{sites:String(p.sites),edges:p.edges.map(e=>`${e.from} ${e.to}`).join('\n'),leftLinks:p.leftLinks.map(g=>`${g.cayley} ${g.minusOne}`).join('\n'),rightLinks:p.rightLinks.map(g=>`${g.cayley} ${g.minusOne}`).join('\n'),leftMatter:p.leftMatter.map(z=>`${z.real} ${z.imaginary}`).join('\n'),rightMatter:p.rightMatter.map(z=>`${z.real} ${z.imaginary}`).join('\n'),forestEdges:p.forestEdges.join(', '),roots:p.roots.join(', '),anchors:p.anchors.join(', ')};}
export function readGaugeOrbitDraft(d:GaugeOrbitDraft):GaugeOrbitInput|null{
 try{
  if(Object.values(d).some(s=>typeof s!=='string'||s.length>16000))return null;
  const integer=(s:string)=>{if(!/^(?:0|[1-9]\d?)$/.test(s))throw Error('Integer index required');return Number(s);};
  const boolean=(s:string)=>{if(s!=='true'&&s!=='false')throw Error('Use true or false');return s==='true';};
  const rows=(s:string)=>s.split(/\r?\n/).filter(row=>row.trim()).map(row=>{const parts=row.trim().split(/[\s,]+/);if(parts.length!==2)throw Error('Two columns required');return parts;});
  const indices=(s:string)=>s.trim()?s.trim().split(/[\s,]+/).map(integer):[];
  const phases=(s:string)=>rows(s).map(([cayley,minusOne])=>({cayley,minusOne:boolean(minusOne)})),matter=(s:string)=>rows(s).map(([real,imaginary])=>({real,imaginary}));
  return readGaugeOrbitInput({sites:integer(d.sites.trim()),edges:rows(d.edges).map(([from,to])=>({from:integer(from),to:integer(to)})),leftLinks:phases(d.leftLinks),rightLinks:phases(d.rightLinks),leftMatter:matter(d.leftMatter),rightMatter:matter(d.rightMatter),forestEdges:indices(d.forestEdges),roots:indices(d.roots),anchors:indices(d.anchors)});
 }catch{return null;}
}
