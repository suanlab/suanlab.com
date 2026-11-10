/** Owned preparations for a fixed finite U(1) graph and two configurations. */
import type {GraphGaugePhase,GraphGaugeComplex} from './learning-graph-gauge-input';
export type GaugeOrbitInput=Readonly<{sites:number;edges:readonly Readonly<{from:number;to:number}>[];leftLinks:readonly GraphGaugePhase[];rightLinks:readonly GraphGaugePhase[];leftMatter:readonly GraphGaugeComplex[];rightMatter:readonly GraphGaugeComplex[];forestEdges:readonly number[];roots:readonly number[];anchors:readonly number[]}>;
function record(v:unknown,keys:string[]):Record<string,unknown>{if(!v||typeof v!=='object'||Object.getPrototypeOf(v)!==Object.prototype||Reflect.ownKeys(v).length!==keys.length)throw Error('Exact plain record required');const out:Record<string,unknown>={};for(const key of keys){const d=Object.getOwnPropertyDescriptor(v,key);if(!d||!Object.hasOwn(d,'value'))throw Error('Owned data required');out[key]=d.value;}return out;}
function list(v:unknown,min:number,max:number):unknown[]{if(!Array.isArray(v)||Object.getPrototypeOf(v)!==Array.prototype)throw Error('Plain array required');const n=Object.getOwnPropertyDescriptor(v,'length')?.value;if(!Number.isSafeInteger(n)||n<min||n>max||Reflect.ownKeys(v).length!==n+1)throw Error('Dense bounded array required');return Array.from({length:n},(_,i)=>{const d=Object.getOwnPropertyDescriptor(v,String(i));if(!d||!Object.hasOwn(d,'value'))throw Error('Array accessor');return d.value;});}
function integer(v:unknown,max:number){if(typeof v!=='number'||!Number.isSafeInteger(v)||v<0||v>=max)throw Error('Index outside range');return v;}
function fraction(v:unknown){if(typeof v!=='string'||v.length>40||!/^(?:0|-?[1-9]\d{0,17})(?:\/[1-9]\d{0,17})?$/.test(v))throw Error('Bounded rational required');return v;}
function phase(v:unknown){const d=record(v,['cayley','minusOne']),cayley=fraction(d.cayley);if(typeof d.minusOne!=='boolean'||(d.minusOne&&cayley.split('/')[0]!=='0'))throw Error('Invalid circle chart');return Object.freeze({cayley,minusOne:d.minusOne});}
function components(n:number){const parent=Array.from({length:n},(_,i)=>i);const find=(i:number):number=>parent[i]===i?i:parent[i]=find(parent[i]);return{find,join(i:number,j:number){const a=find(i),b=find(j);if(a===b)return false;parent[a]=b;return true;}};}
export function readGaugeOrbitInput(value:unknown):GaugeOrbitInput|null{
 try{
  const d=record(value,['sites','edges','leftLinks','rightLinks','leftMatter','rightMatter','forestEdges','roots','anchors']),n=integer(d.sites,13);if(n<3)return null;
  const edges=Object.freeze(list(d.edges,0,32).map(v=>{const e=record(v,['from','to']);return Object.freeze({from:integer(e.from,n),to:integer(e.to,n)});})),m=edges.length;
  const leftLinks=Object.freeze(list(d.leftLinks,m,m).map(phase)),rightLinks=Object.freeze(list(d.rightLinks,m,m).map(phase));
  const matter=(v:unknown)=>Object.freeze(list(v,n,n).map(v=>{const z=record(v,['real','imaginary']);return Object.freeze({real:fraction(z.real),imaginary:fraction(z.imaginary)});}));
  const leftMatter=matter(d.leftMatter),rightMatter=matter(d.rightMatter),forestEdges=Object.freeze(list(d.forestEdges,0,n-1).map(v=>integer(v,m))),roots=Object.freeze(list(d.roots,1,n).map(v=>integer(v,n))),anchors=Object.freeze(list(d.anchors,0,n).map(v=>integer(v,n)));
  if(new Set(forestEdges).size!==forestEdges.length||new Set(roots).size!==roots.length||new Set(anchors).size!==anchors.length)return null;
  const full=components(n),forest=components(n);for(const e of edges)full.join(e.from,e.to);for(const k of forestEdges){const e=edges[k];if(!forest.join(e.from,e.to))return null;}
  if(edges.some(e=>forest.find(e.from)!==forest.find(e.to)))return null;
  const count=new Set(Array.from({length:n},(_,i)=>full.find(i))).size;if(roots.length!==count||forestEdges.length!==n-count||new Set(roots.map(forest.find)).size!==count)return null;
  return Object.freeze({sites:n,edges,leftLinks,rightLinks,leftMatter,rightMatter,forestEdges,roots,anchors});
 }catch{return null;}
}
export const gaugeOrbitInputKey=(p:GaugeOrbitInput)=>JSON.stringify([p.sites,p.edges.map(e=>[e.from,e.to]),p.leftLinks.map(g=>[g.cayley,g.minusOne]),p.rightLinks.map(g=>[g.cayley,g.minusOne]),p.leftMatter.map(z=>[z.real,z.imaginary]),p.rightMatter.map(z=>[z.real,z.imaginary]),p.forestEdges,p.roots,p.anchors]);
