import{thermoelasticAt,thermoelasticProfile,thermoelasticWindow,thermoelectricAt,type ThermoelasticParameters,type ThermoelectricParameters}from './learning-continuum-multiphysics';
export const continuumMultiphysicsKinds=['thermoelastic-mode-reciprocal-energy-exchange','thermoelectric-bar-seebeck-peltier-joule-balance']as const;
export type ContinuumMultiphysicsKind=typeof continuumMultiphysicsKinds[number];
export const isContinuumMultiphysicsKind=(id:string):id is ContinuumMultiphysicsKind=>(continuumMultiphysicsKinds as readonly string[]).includes(id);
export const thermoelasticBounds=[[.5,4],[0,.05],[0,1],[.25,2],[.25,4],[-.001,.001],[-.001,.001],[-.02,.02],[0,4],[0,2*Math.PI],[0,4],[0,4]];
export const thermoelectricBounds=[[.25,4],[.25,4],[.01,4],[.1,4],[-.05,.05],[250,350],[250,350],[-10,10],[-4,4],[0,1]];
function validate(v:number[],bounds:number[][]){if(v.length!==bounds.length)throw new RangeError('Wrong multiphysics control count');for(let i=0;i<bounds.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1])throw new RangeError(`Invalid multiphysics control ${i}`);}
const unique=(v:number[],max:number)=>[...new Set(v.filter(x=>Number.isFinite(x)&&x>=0&&x<=max))].sort((a,b)=>a-b);
export function thermoelasticLab(v:number[]){
 validate(v,thermoelasticBounds);
 const parameters:ThermoelasticParameters={elasticModulus:v[0],coupling:v[1],conductivity:v[2],wavenumber:v[3],area:v[4],initialDisplacement:v[5],initialVelocity:v[6],initialTemperatureDeviation:v[7],density:1,heatCapacity:1,referenceTemperature:300};
 const time=v[8],phase=v[9],start=v[10],end=v[11],query=thermoelasticAt(parameters,time,phase),oneWay=thermoelasticAt(parameters,time,phase,'one-way');
 const times=unique([0,4,time,start,end,...Array.from({length:121},(_,i)=>i/30)],4);
 const temporal=times.map(time=>({time,value:thermoelasticAt(parameters,time,phase),oneWay:thermoelasticAt(parameters,time,phase,'one-way')}));
 const phases=unique([0,2*Math.PI,phase,...Array.from({length:81},(_,i)=>i===80?2*Math.PI:2*Math.PI*i/80)],2*Math.PI);
 const profile=thermoelasticProfile(parameters,time,phases),other=thermoelasticProfile(parameters,time,phases,'one-way');
 const spatial=phases.map((phase,i)=>({phase,x:phase/parameters.wavenumber,value:profile[i],oneWay:other[i]}));
 const minimumLog=Math.min(-24,time>0?Math.log10(time):-24);
 const smallTimes=unique([...Array.from({length:31},(_,i)=>10**(minimumLog+(-2-minimumLog)*i/30)),time],.01).filter(t=>t>0);
 const small=smallTimes.map(time=>({time,logTime:Math.log10(time),value:thermoelasticAt(parameters,time,phase),window:thermoelasticWindow(parameters,0,time)}));
 const window=start<=end?thermoelasticWindow(parameters,start,end):null,oneWayWindow=start<=end?thermoelasticWindow(parameters,start,end,'one-way'):null;
 const startState=thermoelasticAt(parameters,start,phase),endState=thermoelasticAt(parameters,end,phase);
 const roundedWindowChange=startState.totalAvailability.value===null||endState.totalAvailability.value===null?null:endState.totalAvailability.value-startState.totalAvailability.value;
 return{parameters,time,phase,start,end,query,oneWay,temporal,spatial,small,window,oneWayWindow,roundedWindowChange,windowStatus:window?'finite-window' as const:'reversed-time-order' as const};
}
export function thermoelectricLab(v:number[]){
 validate(v,thermoelectricBounds);
 const parameters:ThermoelectricParameters={length:v[0],area:v[1],resistivity:v[2],conductivity:v[3],seebeck:v[4],leftTemperature:v[5],rightTemperature:v[6],currentDensity:v[7],referencePotential:v[8]},xi=v[9];
 const query=thermoelectricAt(parameters,xi),left=thermoelectricAt(parameters,0),right=thermoelectricAt(parameters,1);
 const spatial=unique([0,1,xi,...Array.from({length:121},(_,i)=>i/120)],1).map(xi=>({xi,value:thermoelectricAt(parameters,xi)}));
 const currents=unique([0,Math.abs(parameters.currentDensity),...Array.from({length:81},(_,i)=>10*i/80)],10);
 const response=currents.flatMap(j=>j===0?[{current:0,value:thermoelectricAt({...parameters,currentDensity:0},xi)}]:[-1,1].map(sign=>({current:sign*j,value:thermoelectricAt({...parameters,currentDensity:sign*j},xi)}))).sort((a,b)=>a.current-b.current);
 const minimumLog=Math.min(-24,parameters.currentDensity!==0?Math.log10(Math.abs(parameters.currentDensity)):-24);
 const tiny=unique([...Array.from({length:41},(_,i)=>10**(minimumLog+(-2-minimumLog)*i/40)),Math.abs(parameters.currentDensity)],.01).filter(j=>j>0).map(j=>({logCurrent:Math.log10(j),value:thermoelectricAt({...parameters,currentDensity:parameters.currentDensity<0?-j:j},xi)}));
 return{parameters,xi,query,left,right,spatial,response,tiny};
}
