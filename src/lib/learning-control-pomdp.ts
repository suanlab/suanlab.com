/** Finite sensing/stop POMDP. Hidden binary state is fixed until a terminal guess. */
const range=(v:number,lo:number,hi:number)=>{if(!Number.isFinite(v)||v<lo||v>hi)throw new Error('Use the stated finite sensing POMDP domain.');};
export function binaryObservation(b:number,accuracy:number,y:number){
 const likelihood1=y===1?accuracy:1-accuracy,likelihood0=y===0?accuracy:1-accuracy,scale=Math.max(likelihood1,likelihood0),a=b*(likelihood1/scale),z=(1-b)*(likelihood0/scale),normalizer=a+z;
 return {mass:scale*normalizer,posterior:normalizer===0?null:a/normalizer,likelihood1,likelihood0};
}
export interface SensingPlan {value:number;action:0|1|2;listenCost:number|null;errorRisk:number;expectedListens:number;}
/** Exact observation-tree recursion; horizon counts remaining possible listens, not all actions. */
export function sensingPlan(b:number,accuracy:number,cost:number,H:number):SensingPlan{
 const guess:0|1=b<=1-b?0:1,stop=guess===0?b:1-b,base:SensingPlan={value:stop,action:guess,listenCost:null,errorRisk:stop,expectedListens:0};if(H===0)return base;
 const branches:{mass:number;next:SensingPlan}[]=[];for(const y of [0,1]){const event=binaryObservation(b,accuracy,y);if(event.posterior===null)continue;branches.push({mass:event.mass,next:sensingPlan(event.posterior,accuracy,cost,H-1)});}
 // Constant-preserving expectation avoids underflow of two half-weighted tiny equal risks.
 const expectation=(field:'value'|'errorRisk'|'expectedListens')=>{const floor=Math.min(...branches.map(v=>v.next[field]));return floor+branches.reduce((sum,v)=>sum+v.mass*(v.next[field]-floor),0);},listen=cost+expectation('value');
 return listen<stop?{value:listen,action:2,listenCost:listen,errorRisk:expectation('errorRisk'),expectedListens:1+expectation('expectedListens')}:{...base,listenCost:listen};
}
export function finiteSensingPomdp(b:number,accuracy:number,cost:number,H:number,y:number){
 range(b,0,1);range(accuracy,.5,1);range(cost,0,.3);range(H,0,3);range(y,0,1);if(!Number.isInteger(H)||!Number.isInteger(y))throw new Error('Use integer sensing horizon and observation choices.');
 const selected=sensingPlan(b,accuracy,cost,H),event=binaryObservation(b,accuracy,y),one=sensingPlan(b,accuracy,cost,1),history=Array.from({length:H+1},(_,h)=>({h,...sensingPlan(b,accuracy,cost,h)})),curve=Array.from({length:201},(_,i)=>{const belief=i/200;return [belief,sensingPlan(belief,accuracy,cost,H).value] as [number,number];}),immediate=curve.map(([belief])=>[belief,Math.min(belief,1-belief)] as [number,number]);
 return {selected,event,one,history,curve,immediate,immediateRisk:Math.min(b,1-b),riskReduction:Math.min(b,1-b)-selected.value,conditionalError:event.posterior===null?null:Math.min(event.posterior,1-event.posterior)};
}
