import {reversibleReactionAt,reversibleReactionWindow,fisherFrontAt,fisherFrontWindow,fisherLevelPosition}from './learning-continuum-reaction-transport';
export const continuumReactionKinds=['reversible-reaction-advection-diffusion-budget','fisher-kpp-reaction-diffusion-front']as const;
export type ContinuumReactionKind=typeof continuumReactionKinds[number];
export const isContinuumReactionKind=(id:string):id is ContinuumReactionKind=>(continuumReactionKinds as readonly string[]).includes(id);
export const reactionControlBounds=[[0,4],[.25,4],[.125,2],[-2,2],[-2,2],[0,1],[0,128],[0,128],[0,1],[0,4],[-16,16],[-16,16],[-16,16]];
export const fisherControlBounds=[[.0001,4],[.0001,4],[.25,4],[-2,2],[-2,2],[.25,4],[0,4],[-16,16],[-16,16],[-16,16],[.01,.99]];
function validate(values:number[],bounds:number[][]){if(values.length!==bounds.length)throw new RangeError('Wrong reaction control count');for(let i=0;i<bounds.length;i++)if(!Number.isFinite(values[i])||values[i]<bounds[i][0]||values[i]>bounds[i][1])throw new RangeError(`Invalid reaction control ${i}`);}
function positions(extra:number[],minimum=-16,maximum=16){return [...new Set([...Array.from({length:241},(_,i)=>minimum+(maximum-minimum)*i/240),minimum,maximum,...extra].filter(x=>Number.isFinite(x)&&x>=minimum&&x<=maximum))].sort((a,b)=>a-b);}
export function reversibleTransportLab(values:number[]){
 validate(values,reactionControlBounds);
 const parameters={mass:values[0],area:values[1],initialSigma:values[2],initialMean:values[3],velocity:values[4],diffusivity:values[5],forwardRate:values[6],reverseRate:values[7],initialFractionA:values[8]},time=values[9],x=values[10],left=values[11],right=values[12];
 const query=reversibleReactionAt(parameters,x,time),s=query.total.state;
 const extra=[x,left,right,s.mean,parameters.initialMean,...Array.from({length:121},(_,i)=>s.mean+s.sigma*(-5+i/12)),...Array.from({length:121},(_,i)=>parameters.initialMean+parameters.initialSigma*(-5+i/12))];
 const minimum=Math.max(-16,Math.min(s.mean-4*s.sigma,parameters.initialMean-4*parameters.initialSigma,x,left,right));
 const maximum=Math.min(16,Math.max(s.mean+4*s.sigma,parameters.initialMean+4*parameters.initialSigma,x,left,right));
 const spatial=positions(extra,minimum,maximum).map(x=>({x,value:reversibleReactionAt(parameters,x,time)}));
 const temporal=Array.from({length:81},(_,i)=>{const time=i/20;return{time,value:reversibleReactionAt(parameters,x,time)};});
 const window=left<right?reversibleReactionWindow(parameters,left,right,time):null;
 return{parameters,time,x,left,right,query,window,spatial,temporal,minimum,maximum,windowStatus:window?'finite-window' as const:'invalid-boundary-order' as const};
}
export function fisherTransportLab(values:number[]){
 validate(values,fisherControlBounds);
 const parameters={diffusivity:values[0],growthRate:values[1],carryingConcentration:values[2],bulkVelocity:values[3],initialFront:values[4],area:values[5]},time=values[6],x=values[7],left=values[8],right=values[9],level=values[10];
 const query=fisherFrontAt(parameters,x,time),levelPosition=fisherLevelPosition(parameters,level,time),halfPosition=fisherLevelPosition(parameters,.5,time);
 const extra=[x,left,right,query.center,levelPosition,halfPosition,...Array.from({length:161},(_,i)=>query.center+(-20+i/4)/query.lambda)];
 const spatial=positions(extra).map(x=>({x,value:fisherFrontAt(parameters,x,time)}));
 const temporal=Array.from({length:81},(_,i)=>{const time=i/20;return{time,value:fisherFrontAt(parameters,x,time),levelPosition:fisherLevelPosition(parameters,level,time)};});
 const window=left<right?fisherFrontWindow(parameters,left,right,time):null;
 return{parameters,time,x,left,right,level,query,levelPosition,halfPosition,window,spatial,temporal,windowStatus:window?'finite-window' as const:'invalid-boundary-order' as const};
}
