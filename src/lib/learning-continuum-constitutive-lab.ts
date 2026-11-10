import{viscoelasticAt,viscoelasticWindow,viscoelasticCycle,viscoelasticModuli,neoHookeanAt,neoHookeanWork}from './learning-continuum-constitutive';
export const continuumConstitutiveKinds=['kelvin-voigt-maxwell-memory-dissipation','neo-hookean-uniaxial-constitutive-objectivity']as const;
export type ContinuumConstitutiveKind=typeof continuumConstitutiveKinds[number];
export const isContinuumConstitutiveKind=(id:string):id is ContinuumConstitutiveKind=>(continuumConstitutiveKinds as readonly string[]).includes(id);
export const viscoelasticControlBounds=[[.25,4],[.0625,4],[-.005,.005],[0,4],[-.005,.005],[0,16],[0,16],[0,16]];
export const neoHookeanControlBounds=[[.25,4],[-Math.log(4),Math.log(4)],[-Math.PI,Math.PI],[-Math.PI,Math.PI],[.25,4],[.25,4],[-Math.log(4),Math.log(4)],[-Math.log(4),Math.log(4)]];
function validate(v:number[],bounds:number[][]){if(v.length!==bounds.length)throw new RangeError('Wrong constitutive control count');for(let i=0;i<bounds.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1])throw new RangeError(`Invalid constitutive control ${i}`);}
const unique=(v:number[],lo:number,hi:number)=>[...new Set(v.filter(x=>Number.isFinite(x)&&x>=lo&&x<=hi))].sort((a,b)=>a-b);
export function viscoelasticLab(v:number[]){
 validate(v,viscoelasticControlBounds);
 const parameters={elasticModulus:v[0],viscosity:v[1],strainAmplitude:v[2],angularFrequency:v[3],initialElasticStrain:v[4]},time=v[5],start=v[6],end=v[7];
 const query=viscoelasticAt(parameters,time),cycle=viscoelasticCycle(parameters),window=start<=end?viscoelasticWindow(parameters,start,end):null;
 const horizon=Math.min(16,Math.max(time,start,end,cycle.period??4,4*query.relaxationTime,1));
 const times=unique([0,time,start,end,horizon,...Array.from({length:241},(_,i)=>horizon*i/240),...Array.from({length:81},(_,i)=>6*query.relaxationTime*i/80)],0,horizon);
 const temporal=times.map(time=>({time,value:viscoelasticAt(parameters,time)}));
 const spectrum=Array.from({length:61},(_,i)=>{const logFrequency=-4+(4+Math.log10(4))*i/60;return{logFrequency,value:viscoelasticModuli({...parameters,angularFrequency:i===60?4:10**logFrequency})};});
 const minimumLogTime=Math.min(-24,time>0?Math.log10(time):-24),smallTimes=unique([...Array.from({length:81},(_,i)=>10**(minimumLogTime+(-2-minimumLogTime)*i/80)),time>0&&time<=.01?time:NaN],Number.MIN_VALUE,.01);
 const smallChanges=smallTimes.map(time=>({time,logTime:Math.log10(time),value:viscoelasticAt(parameters,time)}));
 return{parameters,time,start,end,query,cycle,window,horizon,temporal,spectrum,smallChanges,windowStatus:window?'finite-time-window' as const:'invalid-time-order' as const};
}
export function neoHookeanLab(v:number[]){
 validate(v,neoHookeanControlBounds);
 const parameters={shearModulus:v[0],logStretch:v[1],rotation:v[2],normalAngle:v[3],referenceArea:v[4],referenceLength:v[5]},start=v[6],end=v[7],query=neoHookeanAt(parameters),work=neoHookeanWork(parameters,start,end),reverse=neoHookeanWork(parameters,end,start);
 const limit=Math.log(4),stretches=unique([-limit,0,limit,parameters.logStretch,start,end,...Array.from({length:161},(_,i)=>-limit+2*limit*i/160)],-limit,limit);
 const response=stretches.map(logStretch=>({logStretch,value:neoHookeanAt({...parameters,logStretch})}));
 const normals=Array.from({length:121},(_,i)=>{const normalAngle=i===0?-Math.PI:i===120?Math.PI:-Math.PI+2*Math.PI*i/120;return{normalAngle,value:neoHookeanAt({...parameters,normalAngle})};});
 const minimumLog=Math.min(-24,parameters.logStretch!==0?Math.log10(Math.abs(parameters.logStretch)):-24);
 const small=Array.from({length:81},(_,i)=>{const logMagnitude=minimumLog+(-2-minimumLog)*i/80,logStretch=10**logMagnitude;return{logMagnitude,logStretch,value:neoHookeanAt({...parameters,logStretch}),compression:neoHookeanAt({...parameters,logStretch:-logStretch})};}).filter(s=>s.logStretch>0);
 return{parameters,start,end,query,work,reverse,response,normals,small,closedPathResidual:work.workDensity===null||reverse.workDensity===null?null:work.workDensity+reverse.workDensity};
}
