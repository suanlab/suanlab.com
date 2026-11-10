/** Approximate drawing projections only. Fractions and exact residuals are authoritative. */
import type {GraphGaugeResult} from './learning-graph-gauge';
import {scalarFieldProjection} from './learning-scalar-field-view';
export type GraphGaugeMode='original'|'coordinated'|'matter-only';
export function graphGaugeView(r:GraphGaugeResult,mode:GraphGaugeMode,selectedWalk:number){
 const amplitudes=r.nodes.map(z=>scalarFieldProjection(z.density));
 const values=r.nodes.map(z=>mode==='original'?z.matter:z.transformedMatter),projected=values.map(z=>[scalarFieldProjection(z.real),scalarFieldProjection(z.imaginary)]);
 if(amplitudes.some(x=>x===null)||projected.some(pair=>pair.some(x=>x===null)))return null;
 const maximum=Math.sqrt(Math.max(...amplitudes as number[])),densityMaximum=Math.max(...amplitudes as number[]),highlight=new Set(r.input.walks[selectedWalk]?.steps.map(s=>s.edge)??[]);
 const nodes=projected.map((z,i)=>({site:i,x:2.8*Math.cos(2*Math.PI*i/r.sites),z:2.8*Math.sin(2*Math.PI*i/r.sites),height:densityMaximum===0?0:1.2*amplitudes[i]!/densityMaximum,re:maximum===0?0:z[0]!/maximum,im:maximum===0?0:z[1]!/maximum}));
 const edges=r.edges.map((e,index)=>{const from=nodes[e.from],to=nodes[e.to],copies=r.edges.map((x,i)=>({x,i})).filter(({x})=>Math.min(x.from,x.to)===Math.min(e.from,e.to)&&Math.max(x.from,x.to)===Math.max(e.from,e.to)),offset=(copies.findIndex(x=>x.i===index)-(copies.length-1)/2)*Math.min(.35,3/Math.max(1,copies.length-1));
  const points=Array.from({length:25},(_,j)=>{const t=j/24;if(e.from===e.to){const angle=t*2*Math.PI,rad=.2+.45*copies.findIndex(x=>x.i===index)/Math.max(1,copies.length-1);return{x:from.x+rad*(1-Math.cos(angle)),y:from.height,z:from.z+rad*Math.sin(angle)};}const dx=to.x-from.x,dz=to.z-from.z,length=Math.hypot(dx,dz);return{x:from.x+t*dx+Math.sin(Math.PI*t)*offset*(e.from<e.to?1:-1)*(-dz/length),y:from.height+t*(to.height-from.height),z:from.z+t*dz+Math.sin(Math.PI*t)*offset*(e.from<e.to?1:-1)*(dx/length)};});
  const link=mode==='coordinated'?e.transformedLink:e.link,re=scalarFieldProjection(link.real),im=scalarFieldProjection(link.imaginary);return{index,from:e.from,to:e.to,highlight:highlight.has(index),points,re,im};
 });if(edges.some(e=>e.re===null||e.im===null))return null;const extent=Math.max(...nodes.flatMap(p=>[Math.abs(p.x),Math.abs(p.z)]),...edges.flatMap(e=>e.points.flatMap(p=>[Math.abs(p.x),Math.abs(p.z)])))+.65;return{nodes,edges,maximum,densityMaximum,mode,extent};
}
export function graphGaugeEnergyView(r:GraphGaugeResult){const values=[r.energy.total,r.energy.transformedTotal,r.energy.fieldOnlyTotal].map(scalarFieldProjection);if(values.some(x=>x===null))return null;const maximum=Math.max(...(values as number[]).map(Math.abs));return{values:values as number[],maximum,normalized:(values as number[]).map(x=>maximum===0?0:x/maximum)};}
export function graphGaugeLoopView(r:GraphGaugeResult,index:number){const w=r.walks[index];if(!w)return null;const values=[w.holonomy,w.transformedHolonomy,w.reverseHolonomy].map(z=>[scalarFieldProjection(z.real),scalarFieldProjection(z.imaginary)]);return values.some(z=>z.some(x=>x===null))?null:values as [number,number][];}
