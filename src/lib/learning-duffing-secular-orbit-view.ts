/** Exact-reference local (position, velocity, time) projection, with independent axis scales. */
import type {DuffingSecularResult} from './learning-duffing-secular-protocol';
import {scalarFieldProjection as project} from './learning-scalar-field-view';
export function duffingSecularOrbitView(result:DuffingSecularResult,selected:number){
 const midpoint=(v:{lower:string;upper:string})=>(project(v.lower)!+project(v.upper)!)/2;
 const samples=result.localSamples.map(s=>({time:project(s.time)!,position:midpoint(s.exactPosition),velocity:midpoint(s.exactVelocity)}));
 const positionScale=Math.max(...samples.map(p=>Math.abs(p.position)))||1,velocityScale=Math.max(...samples.map(p=>Math.abs(p.velocity)))||1,start=samples[0].time,end=samples[64].time;
 return{positionScale,velocityScale,start,end,selected:Math.max(0,Math.min(64,selected)),equilibrium:result.mode==='equilibrium',points:samples.map(p=>({x:2*p.position/positionScale,y:2*p.velocity/velocityScale,z:end===start?0:6*(p.time-start)/(end-start)-3}))};
}
export type DuffingOrbitView=ReturnType<typeof duffingSecularOrbitView>;
