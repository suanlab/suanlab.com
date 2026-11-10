import test from 'node:test';
import assert from 'node:assert/strict';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {parseFragment} from 'parse5';
import Curve from '../src/components/learning/real-time-phase-curve';
interface Node {nodeName:string;attrs?:{name:string;value:string}[];childNodes?:Node[];}
const walk=(n:Node):Node[]=>[n,...(n.childNodes??[]).flatMap(walk)];
const attributes=(n:Node)=>Object.fromEntries((n.attrs??[]).map(a=>[a.name,a.value]));
function drawing(values:(number|null)[],lines=true){
  const html=renderToStaticMarkup(createElement(Curve,{title:'Signed values',description:'Decimal drawing; original intervals remain authoritative.',xLabel:'x',yLabel:'value',lines,curves:[{label:'values',color:'blue',points:values.map((y,x)=>({x,y}))}]}));return {html,nodes:walk(parseFragment(html))};
}
test('zero reference divides signed heights according to their actual value ratio',()=>{
  const {nodes}=drawing([-2,3]),dots=nodes.filter(n=>n.nodeName==='circle').map(attributes),guide=attributes(nodes.find(n=>attributes(n)['data-real-time-zero-guide'])!);
  assert(Math.abs((Number(dots[0].cy)-Number(guide.y1))/(Number(dots[0].cy)-Number(dots[1].cy))-.4)<1e-12);
});
test('positive and negative plots both identify zero at the correct boundary',()=>{
  for(const values of [[0,2],[-2,0]]){const {nodes}=drawing(values),dots=nodes.filter(n=>n.nodeName==='circle').map(attributes),guide=attributes(nodes.find(n=>attributes(n)['data-real-time-zero-guide'])!);assert.equal(Number(guide.y1),Number(dots[values.indexOf(0)].cy));}
});
test('unavailable samples create gaps and discrete mode never invents connecting lines',()=>{
  for(const [values,lines]of [[[1,null,2],true],[[1,2],false]] as const){const {nodes}=drawing([...values],lines);assert.equal(nodes.filter(n=>n.nodeName==='circle').length,2);assert.equal(nodes.filter(n=>n.nodeName==='line'&&!attributes(n)['data-real-time-zero-guide']).length,0);}
});
test('empty projections retain the description without invalid SVG coordinates',()=>{
  const {html}=drawing([null,null]);assert(!html.includes('<svg'));assert(!/NaN|Infinity/.test(html));assert(html.includes('original intervals'));
});
