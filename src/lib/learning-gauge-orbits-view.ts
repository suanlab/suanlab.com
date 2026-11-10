/** Drawing only: exact wire tables and verified decisions remain authoritative. */
import type {GaugeOrbitResult} from './learning-gauge-orbits';
import {scalarFieldProjection} from './learning-scalar-field-view';
export type GaugeOrbitSide='left'|'right';
export type GaugeOrbitScheme='full'|'anchored';
export type GaugeOrbitRepresentation='original'|'fixed';
export function gaugeOrbitView(r:GaugeOrbitResult,side:GaugeOrbitSide,scheme:GaugeOrbitScheme,representation:GaugeOrbitRepresentation,coordinate:number){
 const project=(z:{real:string;imaginary:string})=>{const re=scalarFieldProjection(z.real),im=scalarFieldProjection(z.imaginary);return re===null||im===null?null:[re,im]as const;};
 const all=[...r.left.originalMatter,...r.right.originalMatter].map(project),values=(representation==='original'?r[side].originalMatter:r[side][scheme].matter).map(project);
 if(all.some(z=>z===null)||values.some(z=>z===null))return null;
 const maximum=Math.max(...all.map(z=>Math.hypot(...z!)));if(!Number.isFinite(maximum))return null;
 const forest=scheme==='full'?r.fullForest:r.anchoredForest,highlight=new Set(r[side][scheme].coordinates[coordinate]?.steps.map(s=>s.edge)??[]);
 const nodes=values.map((z,site)=>{const re=maximum===0?0:z![0]/maximum,im=maximum===0?0:z![1]/maximum;return{site,x:2.8*Math.cos(2*Math.PI*site/r.sites),z:2.8*Math.sin(2*Math.PI*site/r.sites),height:1.2*(re*re+im*im),re,im,anchor:r.input.anchors.includes(site),root:forest.roots.includes(site)};});
 const edges=r.input.edges.map((e,index)=>{
  const from=nodes[e.from],to=nodes[e.to],copies=r.input.edges.map((x,i)=>({x,i})).filter(({x})=>Math.min(x.from,x.to)===Math.min(e.from,e.to)&&Math.max(x.from,x.to)===Math.max(e.from,e.to)),position=copies.findIndex(x=>x.i===index),offset=(position-(copies.length-1)/2)*Math.min(.35,3/Math.max(1,copies.length-1));
  const points=Array.from({length:25},(_,j)=>{const t=j/24;if(e.from===e.to){const angle=t*2*Math.PI,rad=.2+.45*position/Math.max(1,copies.length-1);return{x:from.x+rad*(1-Math.cos(angle)),y:from.height,z:from.z+rad*Math.sin(angle)};}const dx=to.x-from.x,dz=to.z-from.z,length=Math.hypot(dx,dz),bend=Math.sin(Math.PI*t)*offset*(e.from<e.to?1:-1);return{x:from.x+t*dx-bend*dz/length,y:from.height+t*(to.height-from.height),z:from.z+t*dz+bend*dx/length};});
  const link=project((representation==='original'?r[side].originalLinks:r[side][scheme].links)[index]);return{index,from:e.from,to:e.to,forest:forest.edges.includes(index),highlight:highlight.has(index),points,re:link?.[0]??null,im:link?.[1]??null};
 });if(edges.some(e=>e.re===null||e.im===null))return null;
 const extent=Math.max(...nodes.flatMap(p=>[Math.abs(p.x),Math.abs(p.z)]),...edges.flatMap(e=>e.points.flatMap(p=>[Math.abs(p.x),Math.abs(p.z)])))+.65;
 return{nodes,edges,maximum,extent};
}

export function gaugeOrbitCoordinateView(r:GaugeOrbitResult,scheme:GaugeOrbitScheme,index:number){
 const values=([r.left[scheme].coordinates[index],r.right[scheme].coordinates[index]]).map(c=>c?[scalarFieldProjection(c.product.real),scalarFieldProjection(c.product.imaginary)]:null);
 return values.some(z=>z===null||z.some(v=>v===null))?null:values as [number,number][];
}
