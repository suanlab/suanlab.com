/** Independent all-edge constraint solving and exhaustive anchored-forest checking.
 * Exact arithmetic is shared only with another independent verifier,not producers.
 */
import {readGaugeOrbitInput,gaugeOrbitInputKey,type GaugeOrbitInput} from './learning-gauge-orbits-input';
import {gaugeOrbitContract} from './learning-gauge-orbits-contract';
import {snapshotQubitWire} from './learning-qubit-wire';
import type {GaugeOrbitResult} from './learning-gauge-orbits';
import {Rational as R,O,CO,complex,minus,times,conjugate,norm,wire,equal,unitPhase,type Complex} from './learning-graph-gauge-validation-math';
type Step={edge:number;forward:boolean};
const divide=(z:Complex,w:Complex):Complex=>{const value=times(z,conjugate(w)),d=norm(w);return complex(value[0].div(d),value[1].div(d));};
const isZero=(z:Complex)=>z[0].n===BigInt(0)&&z[1].n===BigInt(0);
function same(actual:unknown,expected:unknown):boolean{
 if(expected===null||typeof expected!=='object')return actual===expected;
 if(!actual||typeof actual!=='object'||Array.isArray(actual)!==Array.isArray(expected))return false;
 if(Array.isArray(expected))return Array.isArray(actual)&&actual.length===expected.length&&expected.every((v,i)=>same(actual[i],v));
 const a=actual as Record<string,unknown>,e=expected as Record<string,unknown>,keys=Object.keys(e);return Object.keys(a).length===keys.length&&keys.every(k=>Object.hasOwn(a,k)&&same(a[k],e[k]));
}
export function validateGaugeOrbitResult(source:GaugeOrbitInput,value:unknown):GaugeOrbitResult|null{
 try{
  const parsed=readGaugeOrbitInput(source);if(!parsed)return null;const input:GaugeOrbitInput=parsed,owned=snapshotQubitWire(value);
  if(!owned||typeof owned!=='object')return null;const attributed=readGaugeOrbitInput((owned as Record<string,unknown>).input);if(!attributed||gaugeOrbitInputKey(attributed)!==gaugeOrbitInputKey(input))return null;
  const n=input.sites,m=input.edges.length,allEdges=Array.from({length:m},(_,i)=>i);
  const adjacent=(site:number,selected:readonly number[])=>selected.flatMap(edge=>{const e=input.edges[edge];return e.from===site?[{site:e.to,edge,forward:true}]:e.to===site?[{site:e.from,edge,forward:false}]:[];});
  const groupsOf=(selected:readonly number[])=>{const unseen=new Set(Array.from({length:n},(_,i)=>i)),groups:number[][]=[];while(unseen.size){const start=unseen.values().next().value as number,stack=[start],group:number[]=[];unseen.delete(start);while(stack.length){const site=stack.pop()!;group.push(site);for(const step of adjacent(site,selected))if(unseen.delete(step.site))stack.push(step.site);}groups.push(group.sort((a,b)=>a-b));}return groups;};
  const groups=groupsOf(allEdges),C=groups.length,A=input.anchors.length,info=groups.map((sites,index)=>({index,sites,root:input.roots.find(r=>sites.includes(r))!,anchors:input.anchors.filter(a=>sites.includes(a))})),componentOf=Array.from({length:n},(_,i)=>groups.findIndex(g=>g.includes(i))),unanchored=info.filter(g=>!g.anchors.length).length;
  // Enumerate maximum-cardinality subsets in lexicographic index order.
  // This verifies the declared earliest-edge basis without using union-find.
  const target=n-A-unanchored;
  function enumerate(start:number,selected:number[]):number[]|null{
   if(selected.length===target)return groupsOf(selected).every(g=>input.anchors.filter(a=>g.includes(a)).length<=1)?selected:null;
   const needed=target-selected.length;for(let i=start;i<=input.forestEdges.length-needed;i++){const result=enumerate(i+1,[...selected,input.forestEdges[i]]);if(result)return result;}return null;
  }
  const anchoredEdges=enumerate(0,[]);if(!anchoredEdges)return null;
  const anchoredRoots=input.anchors.length?groupsOf(anchoredEdges).map(g=>input.anchors.find(a=>g.includes(a))??input.roots.find(r=>g.includes(r))!).sort((a,b)=>a-b):[...input.roots];
  function path(start:number,end:number,selected:readonly number[],seen:Set<number>):Step[]|null{
   if(start===end)return[];for(const step of adjacent(start,selected)){if(seen.has(step.site))continue;const tail=path(step.site,end,selected,new Set([...seen,step.site]));if(tail)return[{edge:step.edge,forward:step.forward},...tail];}return null;
  }
  const describe=(selected:readonly number[],roots:readonly number[])=>Array.from({length:n},(_,site)=>{
   for(const root of roots){const steps=path(root,site,selected,new Set([root]));if(!steps)continue;const last=steps.at(-1),edge=last?input.edges[last.edge]:null;return{site,root,parent:last&&edge?(last.forward?edge.from:edge.to):null,viaEdge:last?.edge??null,forward:last?.forward??null,steps};}throw Error('Forest root missing');
  });
  const fullPaths=describe(input.forestEdges,input.roots),anchoredPaths=describe(anchoredEdges,anchoredRoots);
  const product=(links:Complex[],steps:Step[])=>steps.reduce((z,s)=>times(z,s.forward?links[s.edge]:conjugate(links[s.edge])),CO);
  const normal=(links:Complex[],matter:Complex[],forest:readonly number[],nodes:ReturnType<typeof describe>)=>{
   const phases=nodes.map(node=>product(links,node.steps)),fixedLinks=input.edges.map((e,k)=>times(phases[e.from],times(links[k],conjugate(phases[e.to])))),fixedMatter=matter.map((z,i)=>times(z,phases[i]));
   const coordinates=input.edges.flatMap((edge,k)=>{
    if(forest.includes(k))return[];const left=nodes[edge.from],right=nodes[edge.to],reverse=[...right.steps].reverse().map(s=>({edge:s.edge,forward:!s.forward})),steps=[...left.steps,{edge:k,forward:true},...reverse],value=product(links,steps);
    return[{edge:k,start:left.root,end:right.root,kind:left.root===right.root?'closed-cycle':'anchored-path',steps,product:wire(value),canonicalLink:wire(fixedLinks[k]),productResidual:wire(minus(value,fixedLinks[k]))}];
   });
   return{phases,links:fixedLinks,matter:fixedMatter,wire:{phases:phases.map(wire),links:fixedLinks.map(wire),matter:fixedMatter.map(wire),nodeTree:nodes,coordinates,treeResiduals:forest.map(edge=>({edge,value:wire(minus(fixedLinks[edge],CO))})),anchorResiduals:input.anchors.map(site=>({site,value:wire(minus(phases[site],CO))}))}};
  };
  const side=(rawLinks:GaugeOrbitInput['leftLinks'],rawMatter:GaugeOrbitInput['leftMatter'])=>{
   const links=rawLinks.map(unitPhase),matter=rawMatter.map(z=>complex(R.read(z.real),R.read(z.imaginary))),full=normal(links,matter,input.forestEdges,fullPaths),anchored=input.anchors.length?normal(links,matter,anchoredEdges,anchoredPaths):full;
   const treeGram=full.matter.map((z,i)=>full.matter.map((w,j)=>componentOf[i]===componentOf[j]?wire(times(conjugate(z),w)):null)),zeroGroups=info.filter(g=>g.sites.every(i=>isZero(matter[i]))),compatibility=info.map(g=>{const phases=g.anchors.map(a=>full.phases[a]),compatible=phases.every(z=>equal(z,phases[0]));return{component:g.index,anchorCount:phases.length,compatible,rootPhaseFree:!phases.length,requiredRootPhase:phases.length&&compatible?wire(conjugate(phases[0])):null};});
   return{links,matter,full,anchored,zeroGroups,wire:{originalLinks:links.map(wire),originalMatter:matter.map(wire),full:full.wire,anchored:anchored.wire,treeGram,fullRootNormalizedFixAllowedAtAnchors:input.anchors.every(a=>equal(full.phases[a],CO)),fullTreeFixPossibleWithAnchors:compatibility.every(g=>g.compatible),anchorCompatibility:compatibility}};
  };
  const left=side(input.leftLinks,input.leftMatter),right=side(input.rightLinks,input.rightMatter);
  // Equivalence decisions use every original edge,not chord normal forms.
  const solved=info.map(group=>{
   const potentials=new Map<number,Complex>([[group.sites[0],CO]]),queue=[group.sites[0]];let consistent=true;
   while(queue.length){const i=queue.shift()!;for(const step of adjacent(i,allEdges)){
    const ratio=step.forward?times(left.links[step.edge],conjugate(right.links[step.edge])):times(conjugate(left.links[step.edge]),right.links[step.edge]),required=times(potentials.get(i)!,ratio),prior=potentials.get(step.site);
    if(prior){if(!equal(prior,required))consistent=false;}else{potentials.set(step.site,required);queue.push(step.site);}
   }}
   const anchorConstraints=group.anchors.map(a=>conjugate(potentials.get(a)!)),matterConstraints:Complex[]=[];let support=true;
   for(const i of group.sites){if(!isZero(left.matter[i]))matterConstraints.push(divide(right.matter[i],times(potentials.get(i)!,left.matter[i])));else if(!isZero(right.matter[i]))support=false;}
   const compatible=(values:Complex[])=>values.every(z=>norm(z).text()==='1')&&values.every(z=>equal(z,values[0]));
   return{connections:consistent,joint:consistent&&support&&compatible(matterConstraints),anchoredConnections:consistent&&compatible(anchorConstraints),anchoredJoint:consistent&&support&&compatible([...anchorConstraints,...matterConstraints])};
  });
  const witness=(phases:Complex[])=>({phases:phases.map(wire),unitNormResiduals:phases.map(g=>norm(g).sub(O).text()),linkResiduals:input.edges.map((e,k)=>wire(minus(times(phases[e.from],times(left.links[k],conjugate(phases[e.to]))),right.links[k]))),matterResiduals:phases.map((g,i)=>wire(minus(times(g,left.matter[i]),right.matter[i]))),anchorResiduals:input.anchors.map(site=>({site,value:wire(minus(phases[site],CO))}))});
  const compare=(pinned:boolean)=>{
   const l=pinned?left.anchored:left.full,r=pinned?right.anchored:right.full,connectionsEquivalent=solved.every(g=>pinned?g.anchoredConnections:g.connections),jointEquivalent=solved.every(g=>pinned?g.anchoredJoint:g.joint),linkMismatchEdges=l.links.flatMap((z,k)=>equal(z,r.links[k])?[]:[k]);
   const rows=info.map(group=>{
    const fixed=pinned&&group.anchors.length>0,pivot=group.sites.find(i=>!isZero(l.matter[i])),rightZero=group.sites.every(i=>isZero(r.matter[i])),lambda=!fixed&&pivot!==undefined?divide(r.matter[pivot],l.matter[pivot]):CO,unit=norm(lambda).text()==='1',mismatchSites=group.sites.filter(i=>!equal(times(lambda,l.matter[i]),r.matter[i])),matches=unit&&!mismatchSites.length;
    return{phase:lambda,wire:{component:group.index,pinned:fixed,pivot:pivot??null,leftZero:pivot===undefined,rightZero,candidatePhaseUnit:unit,densityMismatchSites:group.sites.filter(i=>norm(l.matter[i]).text()!==norm(r.matter[i]).text()),matches,residualPhase:matches?wire(lambda):null,phaseFree:!fixed&&pivot===undefined&&rightZero,mismatchSites}};
   });
   const gramMismatchPairs:{i:number;j:number}[]=[];
   for(let i=0;i<n;i++)for(let j=i;j<n;j++)if(componentOf[i]===componentOf[j]&&!(pinned&&info[componentOf[i]].anchors.length)&&!equal(times(conjugate(l.matter[i]),l.matter[j]),times(conjugate(r.matter[i]),r.matter[j])))gramMismatchPairs.push({i,j});
   const anchoredMatterMismatches=pinned?Array.from({length:n},(_,i)=>i).filter(i=>info[componentOf[i]].anchors.length&&!equal(l.matter[i],r.matter[i])):[],criterion=!linkMismatchEdges.length&&!gramMismatchPairs.length&&!anchoredMatterMismatches.length,base=l.phases.map((z,i)=>times(z,conjugate(r.phases[i]))),joint=base.map((z,i)=>times(z,rows[componentOf[i]].phase));
   return{connectionsEquivalent,jointEquivalent,linkMismatchEdges,gramMismatchPairs,anchoredMatterMismatches,matterComponents:rows.map(g=>g.wire),connectionWitness:connectionsEquivalent?witness(base):null,jointWitness:jointEquivalent?witness(joint):null,completeJointInvariantCriterion:criterion,criterionAgreesWithWitness:criterion===jointEquivalent};
  };
  const lz=left.zeroGroups.length,rz=right.zeroGroups.length,lf=left.zeroGroups.filter(g=>!g.anchors.length).length,rf=right.zeroGroups.filter(g=>!g.anchors.length).length;
  const expected={input,sites:n,edgeCount:m,componentCount:C,cycleRank:m-n+C,components:info,fullForest:{edges:input.forestEdges,roots:input.roots},anchoredForest:{edges:anchoredEdges,roots:anchoredRoots,removedFromFullForest:input.forestEdges.filter(k=>!anchoredEdges.includes(k))},left:left.wire,right:right.wire,comparison:{full:compare(false),anchored:compare(true)},counts:{fullGaugeGroupDimension:n,anchoredGaugeGroupDimension:n-A,connectionStabilizerDimension:C,anchoredConnectionStabilizerDimension:unanchored,leftJointStabilizerDimension:lz,rightJointStabilizerDimension:rz,leftAnchoredJointStabilizerDimension:lf,rightAnchoredJointStabilizerDimension:rf,fullConnectionOrbitDimension:n-C,anchoredConnectionOrbitDimension:n-A-unanchored,leftJointOrbitDimension:n-lz,rightJointOrbitDimension:n-rz,leftAnchoredJointOrbitDimension:n-A-lf,rightAnchoredJointOrbitDimension:n-A-rf,freeConnectionInvariantCount:m-n+C,anchoredConnectionInvariantCount:m-n+A+unanchored},contract:gaugeOrbitContract};
  return same(owned,expected)?owned as GaugeOrbitResult:null;
 }catch{return null;}
}
