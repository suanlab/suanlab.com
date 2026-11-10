/** Approximate drawings only; exact verified tables decide constraints. */
import type {GraphGaussResult} from './learning-graph-gauss';
import {scalarFieldProjection} from './learning-scalar-field-view';
export type GraphGaussRepresentation='original'|'fixed';
export type GraphGaussElectricMode='input'|'lift';
export function graphGaussView(r:GraphGaussResult,representation:GraphGaussRepresentation,electricMode:GraphGaussElectricMode,coordinate:number){
 const project=(z:{real:string;imaginary:string})=>{const re=scalarFieldProjection(z.real),im=scalarFieldProjection(z.imaginary);return re===null||im===null?null:[re,im]as const;};
 const input=r.input.electric.map(scalarFieldProjection),lift=r.canonical.liftFromCurrentNonForestMomenta?.map(scalarFieldProjection),values=electricMode==='input'?input:lift;
 if(!values||input.some(v=>v===null)||lift?.some(v=>v===null))return null;
 const maximum=Math.max(0,...[...input,...lift??[]].map(v=>Math.abs(v!))),highlight=new Set(r.coordinates[coordinate]?.steps.map(s=>s.edge)??[]);
 const nodes=Array.from({length:r.sites},(_,site)=>({site,x:2.8*Math.cos(2*Math.PI*site/r.sites),z:2.8*Math.sin(2*Math.PI*site/r.sites),height:0,re:0,im:0,anchor:r.input.anchors.includes(site),root:r.fixing.roots.includes(site)}));
 const edges=r.input.edges.map((e,index)=>{
  const from=nodes[e.from],to=nodes[e.to],copies=r.input.edges.map((x,i)=>({x,i})).filter(({x})=>Math.min(x.from,x.to)===Math.min(e.from,e.to)&&Math.max(x.from,x.to)===Math.max(e.from,e.to)),position=copies.findIndex(x=>x.i===index),offset=(position-(copies.length-1)/2)*Math.min(.35,3/Math.max(1,copies.length-1));
  const points=Array.from({length:25},(_,j)=>{const t=j/24;if(e.from===e.to){const angle=t*2*Math.PI,rad=.2+.45*position/Math.max(1,copies.length-1);return{x:from.x+rad*(1-Math.cos(angle)),y:from.height,z:from.z+rad*Math.sin(angle)};}const dx=to.x-from.x,dz=to.z-from.z,length=Math.hypot(dx,dz),bend=Math.sin(Math.PI*t)*offset*(e.from<e.to?1:-1);return{x:from.x+t*dx-bend*dz/length,y:from.height+t*(to.height-from.height),z:from.z+t*dz+bend*dx/length};});
  const phase=r.input.links[index],t=scalarFieldProjection(phase.cayley);if(t===null)return{index,from:e.from,to:e.to,forest:r.fixing.forestEdges.includes(index),highlight:highlight.has(index),points,re:null,im:null,electric:0};const link=representation==='fixed'?project(r.fixing.links[index]):phase.minusOne?[-1,0]:[(1-t*t)/(1+t*t),2*t/(1+t*t)];return{index,from:e.from,to:e.to,forest:r.fixing.forestEdges.includes(index),highlight:highlight.has(index),points,re:link?.[0]??null,im:link?.[1]??null,electric:maximum===0?0:values[index]!/maximum};
 });if(edges.some(e=>e.re===null||e.im===null))return null;
 const extent=Math.max(...nodes.flatMap(p=>[Math.abs(p.x),Math.abs(p.z)]),...edges.flatMap(e=>e.points.flatMap(p=>[Math.abs(p.x),Math.abs(p.z)])))+.65;
 return{nodes,edges,maximum,extent};
}
