export const logicKinds=['propositions-implications','quantifiers-counterexamples','sets-functions','relations-orders','proof-strategies','mathematical-induction'] as const;
export type LogicKind=typeof logicKinds[number];
export function isLogicKind(kind:string):kind is LogicKind{return (logicKinds as readonly string[]).includes(kind);}
export type LogicCell=boolean|number|string;
export interface LogicResult {
 headers:string[];rows:LogicCell[][];stats:Record<string,LogicCell|number[]|null>;
 propertyFailures?:Record<string,number[][]>;matrix?:boolean[][];sources?:number[];targets?:number[];mapping?:number[];counterexamples:number[][];checked:number;
}
export const implies=(p:boolean,q:boolean)=>!p||q;
export function isPrime(n:number){if(!Number.isInteger(n)||n<2)return false;for(let d=2;d*d<=n;d++)if(n%d===0)return false;return true;}
export function smallestDivisor(n:number){if(!Number.isInteger(n)||n<2)return null;for(let d=2;d*d<=n;d++)if(n%d===0)return d;return n;}
export function relationProperties(domain:number[],relation:(x:number,y:number)=>boolean){
 const failures:{reflexive:number[][];symmetric:number[][];antisymmetric:number[][];transitive:number[][]}={reflexive:[],symmetric:[],antisymmetric:[],transitive:[]};
 for(const x of domain){if(!relation(x,x))failures.reflexive.push([x]);for(const y of domain){if(relation(x,y)&&!relation(y,x))failures.symmetric.push([x,y]);if(x!==y&&relation(x,y)&&relation(y,x))failures.antisymmetric.push([x,y]);for(const z of domain)if(relation(x,y)&&relation(y,z)&&!relation(x,z))failures.transitive.push([x,y,z]);}}
 const reflexive=!failures.reflexive.length,symmetric=!failures.symmetric.length,antisymmetric=!failures.antisymmetric.length,transitive=!failures.transitive.length;
 return {reflexive,symmetric,antisymmetric,transitive,equivalence:reflexive&&symmetric&&transitive,partialOrder:reflexive&&antisymmetric&&transitive,totalOrder:reflexive&&antisymmetric&&transitive&&domain.every(x=>domain.every(y=>relation(x,y)||relation(y,x))),failures};
}
export function evaluateLogic(kind:LogicKind,p:number[]):LogicResult {
 const counts:Record<LogicKind,number>={'propositions-implications':2,'quantifiers-counterexamples':2,'sets-functions':5,'relations-orders':3,'proof-strategies':2,'mathematical-induction':2};
 if(p.length!==counts[kind]||p.some(x=>!Number.isInteger(x)))throw new Error('Use integer parameters of the expected shape.');
 const [a,b,c,d,e]=p;const between=(n:number,min:number,max:number)=>n>=min&&n<=max;
 if(kind==='propositions-implications'){
  if(!p.every(x=>between(x,0,1)))throw new Error('Truth values are 0 or 1.');
  const rows=[false,true].flatMap(x=>[false,true].map(y=>[x,y,x&&y,x||y,implies(x,y),implies(!y,!x),implies(y,x)]));
  return {headers:['P','Q','P∧Q','P∨Q','P⇒Q','¬Q⇒¬P','Q⇒P'],rows,stats:{'P⇒Q':implies(!!a,!!b),'¬Q⇒¬P':implies(!b,!a),'Q⇒P':implies(!!b,!!a),'P⇔Q':!!a===!!b},matrix:[[true,true],[false,true]],counterexamples:implies(!!a,!!b)?[]:[[a,b]],checked:4};
 }
 if(kind==='quantifiers-counterexamples'){
  if(!between(a,0,6)||!between(b,0,2))throw new Error('Use domain size 0–6 and predicate 0–2.');
  const domain=Array.from({length:a},(_,i)=>i);const relation=(x:number,y:number)=>b===0?y===x:b===1?y>x:y>=x;
  const matrix=domain.map(x=>domain.map(y=>relation(x,y)));const failed=domain.filter(x=>!domain.some(y=>relation(x,y)));const universalWitnesses=domain.filter(y=>domain.every(x=>relation(x,y)));
  return {headers:['x','∃y R(x,y)','min witness y'],rows:domain.map(x=>[x,matrix[x].some(Boolean),domain.find(y=>relation(x,y))??'—']),stats:{'∀x∃y R(x,y)':!failed.length,'∃y∀x R(x,y)':universalWitnesses.length>0,'∀x∀y R(x,y)':matrix.every(row=>row.every(Boolean)),'∃x∃y R(x,y)':matrix.some(row=>row.some(Boolean)),'global witness y':universalWitnesses.length?universalWitnesses:null},matrix,counterexamples:failed.map(x=>[x]),checked:a*a};
 }
 if(kind==='sets-functions'){
  if(!between(a,1,5)||!between(b,1,5)||!between(c,0,4)||!between(d,0,4)||!between(e,0,5))throw new Error('Invalid finite mapping parameters.');
  const sources=Array.from({length:a},(_,i)=>i),targets=Array.from({length:b},(_,i)=>i),mapping=sources.map(x=>(c*x+d)%b);const A=sources.filter(x=>x%2===0),B=sources.filter(x=>x>=e);const image=[...new Set(mapping)].sort((x,y)=>x-y);
  const collisions=sources.flatMap(x=>sources.filter(y=>y>x&&mapping[x]===mapping[y]).map(y=>[x,y]));
  return {headers:['x','x∈A','x∈B','x∈A∩B','x∈A∪B','x∈A\\B','f(x)'],rows:sources.map(x=>[x,A.includes(x),B.includes(x),A.includes(x)&&B.includes(x),A.includes(x)||B.includes(x),A.includes(x)&&!B.includes(x),mapping[x]]),stats:{A,B,'A∩B':A.filter(x=>B.includes(x)),'A∪B':sources.filter(x=>A.includes(x)||B.includes(x)),'f(D)':image,'f⁻¹(even)':sources.filter(x=>mapping[x]%2===0),'injective':!collisions.length,'surjective':image.length===b,'bijective':!collisions.length&&image.length===b},sources,targets,mapping,counterexamples:collisions,checked:a};
 }
 if(kind==='relations-orders'){
  if(!between(a,1,5)||!between(b,0,3)||!between(c,2,3))throw new Error('Invalid relation parameters.');const domain=Array.from({length:a},(_,i)=>i+1);const relation=(x:number,y:number)=>b===0?x%c===y%c:b===1?y%x===0:b===2?x<y:x<=y;const props=relationProperties(domain,relation);const matrix=domain.map(x=>domain.map(y=>relation(x,y)));
  return {headers:['x',...domain.map(String)],rows:domain.map((x,i)=>[x,...matrix[i]]),stats:{reflexive:props.reflexive,symmetric:props.symmetric,antisymmetric:props.antisymmetric,transitive:props.transitive,equivalence:props.equivalence,'partial order':props.partialOrder,'total order':props.totalOrder},matrix,propertyFailures:props.failures,counterexamples:Object.values(props.failures).flat(),checked:a*a};
 }
 if(kind==='proof-strategies'){
  if(!between(a,0,2)||!between(b,1,8))throw new Error('Invalid claim or integer bound.');const domain=Array.from({length:2*b+1},(_,i)=>i-b);const antecedent=(n:number)=>a===2?n*n%2!==0:n*n%2===0;const consequent=(n:number)=>a===1?n%4===0:n%2===0;const rows=domain.map(n=>[n,antecedent(n),consequent(n),implies(antecedent(n),consequent(n))]);const failed=domain.filter(n=>!implies(antecedent(n),consequent(n)));
  return {headers:['n','P(n)','Q(n)','P(n)⇒Q(n)'],rows,stats:{'finite check':!failed.length,'first counterexample':failed[0]??null,'global proof available':a===0},counterexamples:failed.map(n=>[n]),checked:domain.length};
 }
 if(!between(a,0,2)||!between(b,0,45))throw new Error('Invalid induction claim or limit.');const start=a===1?1:0;const domain=Array.from({length:Math.max(0,b-start+1)},(_,i)=>i+start);let sum=0;const rows:LogicCell[][]=domain.map(n=>{if(a===0){sum+=n;return [n,sum,n*(n+1)/2,sum===n*(n+1)/2];}if(a===1)return [n,2**n,n*n,2**n>n*n];const value=n*n+n+41;return [n,value,smallestDivisor(value)??'—',isPrime(value)];});const failed=rows.filter(row=>!row[3]).map(row=>[row[0] as number]);
 return {headers:a===0?['n','Σ₁ⁿ k','n(n+1)/2','=']:a===1?['n','2ⁿ','n²','>']:['n','n²+n+41','min divisor','prime?'],rows,stats:{'checked cases':domain.length,'finite check':!failed.length,'first counterexample':failed[0]?.[0]??null,'global proof available':a===0},counterexamples:failed,checked:domain.length};
}
