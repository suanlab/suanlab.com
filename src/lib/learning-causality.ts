import { causalLogBit, normalizeCausalLogs } from './learning-causal-probability';
import { isCausalResearchKind, evaluateCausalResearch } from './learning-causal-research';
import { doCalculus, doGraphs, type CausalNetwork } from './learning-causal-graphs';
import type { MeasureResult } from './learning-measure';
import type { Vec2 } from './learning-simulations';

export const causalKinds = ['binary-structural-causal-model', 'confounding-selection-reversal', 'observation-intervention-policy', 'do-calculus-graph-conditions', 'abduction-action-counterfactual', 'observational-equivalence-causal-bounds','instrumental-moment-assumptions','natural-controlled-mediation','gaussian-markov-discovery','invariant-mechanism-transport','hidden-confounding-off-policy'] as const;
export type CausalKind = typeof causalKinds[number];
export const isCausalKind = (kind: string): kind is CausalKind => (causalKinds as readonly string[]).includes(kind);
export interface CausalResult extends MeasureResult {
  graph: 'chain' | 'fork' | 'triangle' | 'confounding-selection' | 'criterion' | 'twin' | 'identified-set' | 'instrument' | 'mediation' | 'discovery' | 'transport' | 'policy';
  network?: CausalNetwork;
  intervention: boolean;
}
const probability = (p: number) => {
  if (!Number.isFinite(p) || p < 0 || p > 1) throw new Error('Use probabilities in [0, 1].');
};
const choice = (p: number, lo: number, hi: number) => {
  if (!Number.isInteger(p) || p < lo || p > hi) throw new Error('Use the stated integer causal choice.');
};
const bernoulli = (p: number, bit: number) => bit ? p : 1 - p;
const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
const expectation = (joint: number[]) => joint[1] + joint[3];
const condition = (joint: number[], x: number) => {
  const mass = joint[2 * x] + joint[2 * x + 1];
  return mass > 0 ? joint[2 * x + 1] / mass : null;
};
const indexed = (values: number[]): Vec2[] => values.map((p, i) => [i, p]);

/** Enumerate independent exogenous bits; an intervention replaces only X's assignment. */
export function binarySCM(pi: number, noiseX: number, noiseY: number, mechanism: number, force: number) {
  [pi, noiseX, noiseY].forEach(probability);
  choice(mechanism, 0, 2); choice(force, -1, 1);
  const observational = [0, 0, 0, 0], manipulated = [0, 0, 0, 0];
  const atoms = [];
  for (const z of [0, 1]) for (const ex of [0, 1]) for (const ey of [0, 1]) {
    const mass = bernoulli(pi, z) * bernoulli(noiseX, ex) * bernoulli(noiseY, ey);
    const naturalX = z ^ ex, x = force < 0 ? naturalX : force;
    const outcome = (value: number) => (mechanism === 0 ? value : mechanism === 1 ? z : value ^ z) ^ ey;
    const naturalY = outcome(naturalX), y = outcome(x);
    observational[2 * naturalX + naturalY] += mass;
    manipulated[2 * x + y] += mass;
    const logMass=causalLogBit(pi,z)+causalLogBit(noiseX,ex)+causalLogBit(noiseY,ey);
    atoms.push({ z, ex, ey, mass, logMass, naturalX, naturalY, x, y });
  }
  const conditional=normalizeCausalLogs(atoms.map(a=>a.naturalX===1?a.logMass:-Infinity));
  return { atoms, observational, manipulated, observationalMean: expectation(observational), manipulatedMean: expectation(manipulated), conditionalOne: conditional.probabilities?sum(atoms.map((a,i)=>conditional.probabilities![i]*a.naturalY)):null };
}

/** Binary backdoor example with an explicit, known outcome mechanism. */
export function binaryConfounding(pi: number, e0: number, e1: number, effect: number) {
  [pi, e0, e1].forEach(probability);
  if (!Number.isFinite(effect) || effect < -.2 || effect > .2) throw new Error('Use effect in [-0.2, 0.2].');
  const population = [1 - pi, pi], propensities = [e0, e1];
  const outcome = (z: number, x: number) => [.2, .7][z] + effect * x;
  const joint = [0, 0, 0, 0], cells: { z: number; x: number; y: number; mass: number; logMass:number }[] = [];
  for (const z of [0, 1]) for (const x of [0, 1]) for (const y of [0, 1]) {
    const mass = population[z] * bernoulli(propensities[z], x) * bernoulli(outcome(z, x), y);
    joint[2 * x + y] += mass; cells.push({ z, x, y, mass, logMass:causalLogBit(pi,z)+causalLogBit(propensities[z],x)+causalLogBit(outcome(z,x),y) });
  }
  const observed = [0,1].map(x=>{const norm=normalizeCausalLogs(cells.map(c=>c.x===x?c.logMass:-Infinity));return norm.probabilities?sum(cells.map((c,i)=>norm.probabilities![i]*c.y)):null;});
  const interventional = [0, 1].map(x => sum(population.map((p, z) => p * outcome(z, x))));
  const adjusted = [0, 1].map(x => {
    let value = 0;
    for (const z of [0, 1]) {
      if (population[z] === 0) continue;
      const group=cells.filter(c=>c.z===z&&c.x===x),norm=normalizeCausalLogs(group.map(c=>c.logMass));
      if(!norm.probabilities)return null;
      value+=population[z]*sum(group.map((c,i)=>norm.probabilities![i]*c.y));
    }
    return value;
  });
  return { population, propensities, outcome, joint, cells, observed, interventional, adjusted,
    crudeDifference: observed.every(v => v !== null) ? observed[1]! - observed[0]! : null,
    adjustedDifference: adjusted.every(v => v !== null) ? adjusted[1]! - adjusted[0]! : null,
    effect: interventional[1] - interventional[0] };
}

/** A separate collider world: independent fair X,Y; selection depends on their OR. */
export function colliderSelection(strength: number) {
  probability(strength);
  const masses = [1 - strength, 1, 1, 1].map(p => p / 4), selectedMass = sum(masses);
  const joint = masses.map(p => p / selectedMass), observed = [condition(joint, 0)!, condition(joint, 1)!];
  return { joint, selectedMass, observed, difference: observed[1] - observed[0], causalEffect: 0 };
}

/** Observation filters the population; an external Bernoulli policy replaces its assignment. */
export function observationPolicy(pi: number, e0: number, e1: number, query: number, policy: number) {
  [pi, e0, e1, policy].forEach(probability); choice(query, 0, 1);
  const population = [1 - pi, pi], propensity = [e0, e1];
  const outcome = (z: number, x: number) => .1 + .6 * z + .2 * x;
  const queryMass = sum(population.map((p, z) => p * bernoulli(propensity[z], query)));
  const normalized=normalizeCausalLogs([0,1].map(z=>causalLogBit(pi,z)+causalLogBit(propensity[z],query))),posterior=normalized.probabilities;
  const observed = posterior ? sum(posterior.map((p, z) => p * outcome(z, query))) : null;
  const doMeans = [0, 1].map(x => sum(population.map((p, z) => p * outcome(z, x))));
  const supported = population.every((p, z) => p === 0 || bernoulli(propensity[z], query) > 0);
  return { population, propensity, outcome, queryMass, logQueryMass:normalized.logMass, posterior, observed, doMeans,
    intervention: doMeans[query], policyMean: (1 - policy) * doMeans[0] + policy * doMeans[1],
    supported, adjusted: supported ? doMeans[query] : null };
}

/** Abduction reweights the original exogenous atoms; action retains the same noise. */
export function binaryCounterfactual(pi:number,ex:number,ey:number,mechanism:number,factualX:number,factualY:number,action:number){
  choice(factualX,0,1);choice(factualY,0,1);choice(action,0,1);
  const factual=binarySCM(pi,ex,ey,mechanism,-1),worlds=[binarySCM(pi,ex,ey,mechanism,0),binarySCM(pi,ex,ey,mechanism,1)];
  const evidence=factual.atoms.reduce((s,a)=>s+(a.naturalX===factualX&&a.naturalY===factualY?a.mass:0),0);
  const normalized=normalizeCausalLogs(factual.atoms.map(a=>a.naturalX===factualX&&a.naturalY===factualY?a.logMass:-Infinity)),posterior=normalized.probabilities;
  const means=worlds.map(world=>posterior?sum(world.atoms.map((a,i)=>posterior[i]*a.y)):null);
  return {factual,worlds,evidence,logEvidence:normalized.logMass,posterior,means,populationMeans:worlds.map(world=>world.manipulatedMean),selected:means[action],action};
}
/** Sharp binary ATE bounds under consistency alone, with explicit compatible completions. */
export function observationalBounds(pi:number,r0:number,r1:number,completion:number,mask:number){
  [pi,r0,r1,completion].forEach(probability);choice(mask,0,15);
  const joint=[(1-pi)*(1-r0),(1-pi)*r0,pi*(1-r1),pi*r1];
  const lower=joint[3]-joint[1]-pi,upper=joint[3]-joint[1]+1-pi;
  const atoms=[];
  for(const u of [0,1])for(const y0 of [0,1])for(const y1 of [0,1]){
    const q0=u===0?r0:1-completion,q1=u===0?completion:r1;
    atoms.push({u,y0,y1,y:u===0?y0:y1,mass:bernoulli(pi,u)*bernoulli(q0,y0)*bernoulli(q1,y1)});
  }
  const doMeans=[sum(atoms.map(a=>a.mass*a.y0)),sum(atoms.map(a=>a.mass*a.y1))];
  const vertices=Array.from({length:16},(_,bits)=>sum(joint.map((p,i)=>{const x=Math.floor(i/2),y=i%2,missing=(bits>>i)&1;return p*(x===0?missing-y:y-missing);}))); 
  return {joint,lower,upper,atoms,doMeans,ate:doMeans[1]-doMeans[0],vertices,selectedVertex:vertices[mask],crude:pi>0&&pi<1?r1-r0:null};
}

export function evaluateCausal(kind: CausalKind, p: number[]): CausalResult {
  if(isCausalResearchKind(kind))return evaluateCausalResearch(kind,p);
  const dimensions = { 'binary-structural-causal-model': 5, 'confounding-selection-reversal': 5, 'observation-intervention-policy': 5, 'do-calculus-graph-conditions': 7, 'abduction-action-counterfactual': 7, 'observational-equivalence-causal-bounds': 5 };
  if (p.length !== dimensions[kind]) throw new Error('Use the complete causal parameter vector.');
  const [a, b, c, d, e] = p;
  if (kind === 'binary-structural-causal-model') {
    const r = binarySCM(a, b, c, d, e);
    return { plot: 'discrete', graph: (['chain', 'fork', 'triangle'] as const)[d], intervention: e >= 0,
      statement: `Z~Bern(${a}); E_X~Bern(${b}); E_Y~Bern(${c}); independent exogenous bits; X=Z XOR E_X; Y=${['X', 'Z', '(X XOR Z)'][d]} XOR E_Y; ${e < 0 ? 'observational assignments' : `do(X=${e}) replaces only X`}`,
      stats: { 'total exogenous mass': sum(r.atoms.map(v => v.mass)), 'observational E[Y]': r.observationalMean, 'manipulated E[Y]': r.manipulatedMean, 'observational P(Y=1|X=1)': r.conditionalOne, '1{assignment replaced}': Number(e >= 0) },
      curves: [indexed(r.observational), indexed(r.manipulated)], curveColors: ['blue', 'orange'], bounds: [0, 3, 0, 1.05],
      table: { headers: ['Z', 'E_X', 'E_Y', 'atom probability', 'natural X', 'natural Y', 'current X', 'current Y'], rows: r.atoms.map(v => [v.z, v.ex, v.ey, v.mass, v.naturalX, v.naturalY, v.x, v.y]) } };
  }
  if (kind === 'confounding-selection-reversal') {
    const r = binaryConfounding(a, b, c, d), s = colliderSelection(e);
    const observed = r.observed.flatMap((v, x) => v === null ? [] : [[x, v] as Vec2]);
    return { plot: 'discrete', graph: 'confounding-selection', intervention: false,
      statement: `World A: Z~Bern(${a}); P(X=1|Z=0,1)=(${b},${c}); P(Y=1|Z,X)=(0.2,0.7)[Z]+${d}X. World B: independent fair X,Y; P(S=1|X,Y)=1 except P(S=1|0,0)=1-${e}; condition on S=1.`,
      stats: { 'crude observed difference': r.crudeDifference, 'oracle causal difference': r.effect, 'backdoor adjusted difference': r.adjustedDifference, '1{both interventions supported}': Number(r.adjusted.every(v => v !== null)), 'collider selected difference': s.difference, 'collider population causal difference': s.causalEffect, 'collider selection probability': s.selectedMass },
      curves: [observed, indexed(r.interventional)], curveColors: ['blue', 'orange'], bounds: [0, 1, 0, 1.05], dots: observed, dotColors: observed.map(() => 'blue'),
      table: { headers: ['world', 'X', 'Y', 'joint probability'], rows: [...r.joint.map((v, i) => ['A population', Math.floor(i / 2), i % 2, v]), ...s.joint.map((v, i) => ['B given S=1', Math.floor(i / 2), i % 2, v])] } };
  }
  if(kind==='observation-intervention-policy'){
  const r = observationPolicy(a, b, c, d, e);
  const curve: Vec2[] = Array.from({ length: 41 }, (_, i) => [i / 40, (1 - i / 40) * r.doMeans[0] + i / 40 * r.doMeans[1]]);
  return { plot: 'continuous', graph: 'triangle', intervention: true,
    statement: `Z~Bern(${a}); observational P(X=1|Z=0,1)=(${b},${c}); Y=1{U_Y<0.1+0.6Z+0.2X}, U_Y independent uniform; query X=${d}; external policy X~Bern(r), independent of Z,U_Y. Oracle SCM means are distinct from identification from observed support.`,
    stats: { 'observational P(X=query)': r.queryMass, 'observational E[Y|X=query]': r.observed, 'oracle E[Y|do(X=query)]': r.intervention, 'external policy E[Y]': r.policyMean, 'oracle average treatment effect': r.doMeans[1] - r.doMeans[0], 'backdoor identified query mean': r.adjusted, '1{query has stratum support}': Number(r.supported) },
    curves: [curve, ...(r.observed === null ? [] : [[[0, r.observed], [1, r.observed]] as Vec2[]])], curveColors: ['blue', 'orange'], bounds: [0, 1, 0, 1.05], dots: [[e, r.policyMean]], dotColors: ['blue'],
    table: { headers: ['Z', 'population P(Z)', 'observed P(Z|X=query)', 'outcome mean at query'], rows: [0, 1].map(z => [z, r.population[z], r.posterior ? r.posterior[z] : '—', r.outcome(z, d)]) } };
  }
  if(kind==='do-calculus-graph-conditions'){
    const r=doCalculus(a,b,c,d,e,p[5],p[6]),names=['X','Z','W','Y'];
    const points=Array.from({length:21},(_,i)=>{const t=.6*i/20,v=doCalculus(a,b,c,d,e,t,p[6]);return {t,left:v.left.mean!,right:v.right.mean!};});
    return {plot:'continuous',graph:'criterion',intervention:true,
      network:{nodes:[{name:'X',x:65,y:50,intervened:true},{name:'Z',x:285,y:50},{name:'W',x:65,y:175,conditioned:Boolean(p[6])},{name:'Y',x:285,y:175}],edges:r.original.map(([from,to])=>({from,to,removed:!r.criterion.some(([u,v])=>u===from&&v===to)}))},
      statement:`Rule ${b}; ${doGraphs[a].name}. Criterion graph, not a second intervention distribution: remove inputs to X; ${b===2?'also remove outputs from Z':b===3&&r.zInZW?'also remove inputs to Z':'no additional edge deletion'}. Condition on X${p[6]?',W':''}. Independent uniform-noise assignments q=0.2+strength*mean(parents). Numerical coincidence does not certify the graphical rule.`,
      stats:{'1{graph criterion licenses rule}':Number(r.separation.separated),'left conditional mean':r.left.mean,'right conditional mean':r.right.mean,'left minus right':r.left.mean!-r.right.mean!,'Z(W) node count':Number(r.zInZW),'active path count':r.separation.active.length,'left conditioning probability':r.left.mass,'right conditioning probability':r.right.mass},
      curves:[points.map(v=>[v.t,v.left] as Vec2),points.map(v=>[v.t,v.right] as Vec2)],curveColors:['blue','orange'],bounds:[0,.6,0,1.05],dots:[[p[5],r.left.mean!],[p[5],r.right.mean!]],dotColors:['blue','orange'],
      table:{headers:['kind','detail','value'],rows:[...r.original.map(([u,v])=>['original edge',`${names[u]} -> ${names[v]}`,r.criterion.some(([f,t])=>f===u&&t===v)?'retained':'removed']),...r.separation.paths.map(path=>['criterion path',path.map(i=>names[i]).join(' -- '),r.separation.active.some(q=>q.join(',')===path.join(','))?'active':'blocked'])]}};
  }
  if(kind==='abduction-action-counterfactual'){
    const r=binaryCounterfactual(a,b,c,d,e,p[5],p[6]),conditional=r.means.flatMap((v,i)=>v===null?[]:[[i,v] as Vec2]);
    return {plot:'discrete',graph:'twin',intervention:true,
      network:{nodes:[{name:'U',x:175,y:35,hidden:true},{name:'X_f',x:65,y:100,conditioned:true},{name:'Y_f',x:285,y:100,conditioned:true},{name:'X_a',x:65,y:195,intervened:true},{name:'Y_a',x:285,y:195}],edges:[{from:0,to:1},{from:0,to:2},{from:0,to:4},...(d===1?[]:[{from:1,to:2},{from:3,to:4}])]},
      statement:`Same binary SCM as the structural chapter, mechanism ${d}; independent U=(Z,E_X,E_Y). Evidence X_f=${e},Y_f=${p[5]}; action do(X_a=${p[6]}). Abduction filters original atoms; action reuses their same Z,E_X,E_Y. New noise would answer a different question. Zero evidence gives no posterior.`,
      stats:{'factual evidence probability':r.evidence,'counterfactual E[Y_action|evidence]':r.selected,'population E[Y|do(action)]':r.populationMeans[p[6]],'posterior mass sum':r.posterior?sum(r.posterior):null,'1{posterior defined}':Number(r.posterior!==null)},
      curves:[conditional,indexed(r.populationMeans)],curveColors:['blue','orange'],dots:conditional,dotColors:conditional.map(()=>'blue'),bounds:[0,1,0,1.05],
      table:{headers:['Z','E_X','E_Y','prior mass','posterior mass','factual X','factual Y','action Y'],rows:r.factual.atoms.map((v,i)=>[v.z,v.ex,v.ey,v.mass,r.posterior?r.posterior[i]:'—',v.naturalX,v.naturalY,r.worlds[p[6]].atoms[i].y])}};
  }
  const r=observationalBounds(a,b,c,d,e),points:Vec2[]=Array.from({length:21},(_,i)=>[i/20,r.lower+i/20]);
  return {plot:'continuous',graph:'identified-set',intervention:false,
    network:{nodes:[{name:'U',x:175,y:45,hidden:true},{name:'X',x:75,y:170},{name:'Y',x:275,y:170}],edges:[{from:0,to:1},{from:0,to:2},{from:1,to:2}]},
    statement:`Hidden U~Bern(${a}); natural X=U. Within U=0: E[Y0]=${b}, E[Y1]=lambda; within U=1: E[Y0]=1-lambda, E[Y1]=${c}; independent potential-outcome noises conditional on U. Observed P(X,Y) stays fixed as lambda changes. Bounds use consistency alone, without exchangeability or monotonicity. Mask chooses missing potential outcomes for cells 00,01,10,11.`,
    stats:{'completion oracle ATE':r.ate,'sharp consistency lower bound':r.lower,'sharp consistency upper bound':r.upper,'identified interval width':r.upper-r.lower,'selected extreme completion ATE':r.selectedVertex,'observed crude difference':r.crude,'oracle E[Y0]':r.doMeans[0],'oracle E[Y1]':r.doMeans[1],'observed P00':r.joint[0],'observed P01':r.joint[1],'observed P10':r.joint[2],'observed P11':r.joint[3]},
    curves:[points,[[0,r.lower],[1,r.lower]],[[0,r.upper],[1,r.upper]]],curveColors:['blue','orange','orange'],dots:[[d,r.ate]],dotColors:['blue'],bounds:[0,1,-1.05,1.05],
    table:{headers:['U=X','Y0','Y1','observed Y','completion mass'],rows:r.atoms.map(v=>[v.u,v.y0,v.y1,v.y,v.mass])}};
}
