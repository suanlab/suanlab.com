/** Reader-only projections of verified samples; no BVP or exponential engine. */
import type {BoundaryLayerResult} from './learning-boundary-layer-protocol';
import {scalarFieldProjection as project} from './learning-scalar-field-view';
export type BoundaryLayerCoordinate='physical'|'inner';
export type BoundaryLayerScale='linear'|'signed-log';
export function boundaryLayerView(result:BoundaryLayerResult,coordinate:BoundaryLayerCoordinate,scale:BoundaryLayerScale,showInner:boolean){
 const actualCoordinate=coordinate==='inner'&&result.layer?'inner':'physical',source=actualCoordinate==='inner'?result.layerSamples:result.physicalSamples;
 const midpoint=(r:{lower:string;upper:string})=>(project(r.lower)!+project(r.upper)!)/2;
 const transform=(y:number)=>scale==='signed-log'?Math.sign(y)*Math.log1p(Math.abs(y))/Math.LN10:y;
 const points=source.map(p=>({coordinate:project(actualCoordinate==='inner'?p.stretchedCoordinate!:p.x)!,physicalX:project(p.x)!,exact:midpoint(p.exact),outer:p.outer===null?null:project(p.outer)!,inner:p.inner===null?null:midpoint(p.inner),composite:p.composite===null?null:midpoint(p.composite),source:p}));
 const kinds=(result.layer?['exact','outer','composite',...(showInner?['inner']:[])]:['exact'])as('exact'|'outer'|'composite'|'inner')[];
 const values=points.flatMap(p=>kinds.map(k=>transform(p[k]!))),minimum=Math.min(...values),maximum=Math.max(...values),pad=maximum===minimum?Math.max(1,Math.abs(maximum)*.05):(maximum-minimum)*.07;
 const yMin=minimum-pad,yMax=maximum+pad,xMax=Math.max(1e-100,...points.map(p=>p.coordinate));
 return{coordinate:actualCoordinate,scale,showInner,points,kinds,transform,yMin,yMax,xMax,zeroDrift:result.layer===null};
}
export type BoundaryLayerView=ReturnType<typeof boundaryLayerView>;
