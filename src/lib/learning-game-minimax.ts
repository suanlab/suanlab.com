export type MinimaxFraction=[number,number];
export type MinimaxInterval=[MinimaxFraction,MinimaxFraction];
const gcd=(a:number,b:number):number=>{a=Math.abs(a);b=Math.abs(b);while(b){const r=a%b;a=b;b=r;}return a;};
const fraction=(n:number,d=1):MinimaxFraction=>{if(d<0){n=-n;d=-d;}const g=gcd(n,d);return [n===0?0:n/g,d/g];};
const compare=(a:MinimaxFraction,b:MinimaxFraction)=>a[0]*b[1]-b[0]*a[1],copy=(v:MinimaxFraction):MinimaxFraction=>[...v];
export const minimaxNumber=(v:MinimaxFraction)=>v[0]/v[1];
export const minimaxLabel=(v:MinimaxFraction)=>v[1]===1?String(v[0]):`${v[0]}/${v[1]}`;
const line=(p:MinimaxFraction,low:number,high:number)=>fraction(p[0]*high+(p[1]-p[0])*low,p[1]);
export function minimaxSolution(A:readonly number[]){
 if(A.length!==4||A.some(v=>!Number.isInteger(v)||Math.abs(v)>4))throw new Error('Use four integer zero-sum payoffs from-4 through4.');
 const [a,b,c,d]=A,delta=a-c-b+d,candidates=(n:number)=>{const xs=[fraction(0),fraction(1)];if(delta){const r=fraction(n,delta);if(compare(r,fraction(0))>0&&compare(r,fraction(1))<0)xs.push(r);}return xs.sort(compare);},lower=(p:MinimaxFraction)=>{const vals=[line(p,c,a),line(p,d,b)];return compare(vals[0],vals[1])<=0?vals[0]:vals[1];},upper=(q:MinimaxFraction)=>{const vals=[line(q,b,a),line(q,d,c)];return compare(vals[0],vals[1])>=0?vals[0]:vals[1];},pc=candidates(d-c),qc=candidates(d-b),pValues=pc.map(lower),qValues=qc.map(upper),primalValue=pValues.reduce((s,v)=>compare(s,v)>=0?s:v),dualValue=qValues.reduce((s,v)=>compare(s,v)<=0?s:v),pArg=pc.filter((_,i)=>compare(pValues[i],primalValue)===0),qArg=qc.filter((_,i)=>compare(qValues[i],dualValue)===0),pOptimal:MinimaxInterval=[copy(pArg[0]),copy(pArg.at(-1)!)],qOptimal:MinimaxInterval=[copy(qArg[0]),copy(qArg.at(-1)!)];
 return {A:A.slice(),primalValue:copy(primalValue),dualValue:copy(dualValue),pOptimal,qOptimal,pcandidates:pc.map((p,i)=>({probability:copy(p),bound:copy(pValues[i])})),qcandidates:qc.map((q,i)=>({probability:copy(q),bound:copy(qValues[i])})),pureLower:Math.max(Math.min(a,b),Math.min(c,d)),pureUpper:Math.min(Math.max(a,c),Math.max(b,d)),strongDuality:compare(primalValue,dualValue)===0};
}
export function minimaxLab(a:number,b:number,c:number,d:number,pNum:number,qNum:number){
 if(!Number.isInteger(pNum)||pNum<0||pNum>12||!Number.isInteger(qNum)||qNum<0||qNum>12)throw new Error('Use probability numerators from0 through12.');
 const solution=minimaxSolution([a,b,c,d]),columnNumerators=[pNum*a+(12-pNum)*c,pNum*b+(12-pNum)*d],rowNumerators=[qNum*a+(12-qNum)*b,qNum*c+(12-qNum)*d],lowerNumerator=Math.min(...columnNumerators),upperNumerator=Math.max(...rowNumerators),payoffNumerator=pNum*qNum*a+pNum*(12-qNum)*b+(12-pNum)*qNum*c+(12-pNum)*(12-qNum)*d,gapNumerator=12*(upperNumerator-lowerNumerator),rowGainNumerator=12*upperNumerator-payoffNumerator,columnGainNumerator=payoffNumerator-12*lowerNumerator,inside=(n:number,v:MinimaxInterval)=>compare([n,12],v[0])>=0&&compare([n,12],v[1])<=0,pOptimal=inside(pNum,solution.pOptimal),qOptimal=inside(qNum,solution.qOptimal);
 return {...solution,pNum,qNum,columnNumerators,rowNumerators,lowerNumerator,upperNumerator,payoffNumerator,gapNumerator,rowGainNumerator,columnGainNumerator,pQueryOptimal:pOptimal,qQueryOptimal:qOptimal,isSaddle:gapNumerator===0,queryInOptimalProduct:pOptimal&&qOptimal,dimension:Number(compare(solution.pOptimal[0],solution.pOptimal[1])!==0)+Number(compare(solution.qOptimal[0],solution.qOptimal[1])!==0)};
}
