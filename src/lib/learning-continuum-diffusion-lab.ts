import {heatModeAt,heatModeCell,heatModeWindow,gaussianDiffusionAt,gaussianDiffusionState,gaussianDiffusionWindow}from './learning-continuum-diffusion';
export const continuumDiffusionKinds=['fourier-heat-diffusion-mode-budget','gaussian-fick-diffusion-spreading']as const;
export type ContinuumDiffusionKind=typeof continuumDiffusionKinds[number];
export const isContinuumDiffusionKind=(id:string):id is ContinuumDiffusionKind=>(continuumDiffusionKinds as readonly string[]).includes(id);
export const heatControlBounds=[[.25,4],[0,4],[250,400],[-.2,.2],[.25,4],[-Math.PI,Math.PI],[.25,4],[0,4],[-16,16],[-4,4],[0,2]];
export const gaussianControlBounds=[[0,4],[.25,4],[.125,2],[-2,2],[-2,2],[0,1],[0,4],[-16,16],[-16,16],[-16,16]];
function validate(v:number[],bounds:number[][]){if(v.length!==bounds.length)throw new RangeError('Wrong diffusion control count');for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1])throw new RangeError(`Invalid diffusion control ${i}`);}
export function heatDiffusionLab(v:number[]){
 validate(v,heatControlBounds);
 const parameters={heatCapacity:v[0],conductivity:v[1],meanTemperature:v[2],contrast:v[3],waveNumber:v[4],phase:v[5],area:v[6]},time=v[7],x=v[8],center=v[9],halfWidth=v[10];
 const query=heatModeAt(parameters,x,time),cell=heatModeCell(parameters,time),window=heatModeWindow(parameters,center,halfWidth,time);
 const radius=Math.max(Math.PI/parameters.waveNumber,Math.abs(x),Math.abs(window.left),Math.abs(window.right),1),count=Math.max(240,Math.ceil(16*parameters.waveNumber*radius/Math.PI));
 const spatial=Array.from({length:count+1},(_,i)=>{const x=i===0?-radius:i===count?radius:-radius+2*radius*i/count;return{x,value:heatModeAt(parameters,x,time),initial:heatModeAt(parameters,x,0)};});
 const temporal=Array.from({length:81},(_,i)=>{const time=i/20;return{time,value:heatModeAt(parameters,x,time),cell:heatModeCell(parameters,time),window:heatModeWindow(parameters,center,halfWidth,time)};});
 const waveNumbers=Array.from({length:61},(_,i)=>{const waveNumber=.25+3.75*i/60;return{waveNumber,rate:heatModeAt({...parameters,waveNumber},x,time).decayRate};});
 const smallConductivities=Array.from({length:121},(_,i)=>{const logConductivity=-30+i/4;return{logConductivity,value:heatModeAt({...parameters,conductivity:10**logConductivity},x,time)};});
 return{parameters,time,x,center,halfWidth,query,cell,window,radius,spatial,temporal,waveNumbers,smallConductivities};
}
/** Rounded CDF used only as a subtraction control, not as the interval integrator. */
export function roundedStandardNormalCdf(z:number):number{
 if(!Number.isFinite(z))throw new RangeError('Finite standardized coordinate required');
 if(z===0)return .5;
 const x=Math.abs(z);
 if(x>=1){let tail=0;for(let j=512;j>=1;j--)tail=j/(x+tail);const probability=Math.exp(-x*x/2-.5*Math.log(2*Math.PI)-Math.log(x+tail));return z>0?1-probability:probability;}
 let term=z,total=z;for(let j=1;j<30;j++){term*=(-z*z/2)/j;total+=term/(2*j+1);}return .5+total/Math.sqrt(2*Math.PI);
}
export function gaussianDiffusionLab(v:number[]){
 validate(v,gaussianControlBounds);
 const parameters={mass:v[0],area:v[1],initialSigma:v[2],initialMean:v[3],velocity:v[4],diffusivity:v[5]},time=v[6],x=v[7],left=v[8],right=v[9];
 const query=gaussianDiffusionAt(parameters,x,time),state=query.state,window=left<right?gaussianDiffusionWindow(parameters,left,right,time):null;
 const minimum=Math.max(-16,Math.min(state.mean-4*state.sigma,parameters.initialMean-4*parameters.initialSigma,x,left,right));
 const maximum=Math.min(16,Math.max(state.mean+4*state.sigma,parameters.initialMean+4*parameters.initialSigma,x,left,right));
 const candidates=[...Array.from({length:241},(_,i)=>minimum+(maximum-minimum)*i/240),...Array.from({length:121},(_,i)=>state.mean+state.sigma*(-5+i/12)),...Array.from({length:121},(_,i)=>parameters.initialMean+parameters.initialSigma*(-5+i/12)),minimum,maximum,state.mean,parameters.initialMean,x,left,right];
 const positions=[...new Set(candidates.filter(x=>x>=minimum&&x<=maximum))].sort((a,b)=>a-b);
 const spatial=positions.map(x=>({x,value:gaussianDiffusionAt(parameters,x,time),initial:gaussianDiffusionAt(parameters,x,0)}));
 const temporal=Array.from({length:81},(_,i)=>{const time=i/20,state=gaussianDiffusionState(parameters,time);return{time,state,peak:gaussianDiffusionAt(parameters,state.mean,time).concentration};});
 const peak=gaussianDiffusionAt(parameters,state.mean,time).concentration,initialPeak=gaussianDiffusionAt(parameters,parameters.initialMean,0).concentration;
 const leftCdf=roundedStandardNormalCdf((left-state.mean)/state.sigma),rightCdf=roundedStandardNormalCdf((right-state.mean)/state.sigma);
 return{parameters,time,x,left,right,query,state,peak,initialPeak,window,spatial,temporal,minimum,maximum,leftCdf,rightCdf,cdfDifference:window?rightCdf-leftCdf:null,
  windowStatus:window?'finite-window' as const:'invalid-boundary-order' as const};
}
