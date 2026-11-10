import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { f, one, zero, sub, mul, div, le, neg, type F } from '../src/lib/learning-tensor-wire-math';
const runtime='.runtime/learning/many-body-schmidt-spectral';
function parse(s:string):F{if(s.includes('/')){const[n,d]=s.split('/');return f(BigInt(n),BigInt(d));}const m=/^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s.replace('decimal:',''));assert(m,s);const n=BigInt(m[1]+m[2]+(m[3]??'')),e=Number(m[4]??0)-(m[3]??'').length;return e>=0?f(n*BigInt(10)**BigInt(e)):f(n,BigInt(10)**BigInt(-e));}
const abs=(v:F)=>v.n<BigInt(0)?neg(v):v;
function compare(r:{lower:{n:string;d:string};upper:{n:string;d:string}},s:string,digits:number){const a=f(BigInt(r.lower.n),BigInt(r.lower.d)),b=f(BigInt(r.upper.n),BigInt(r.upper.d)),v=parse(s),unit=le(abs(v),one)?one:abs(v),eps=div(unit,f(BigInt(10)**BigInt(digits)));assert(le(a,b)&&le(a,{n:v.n*eps.d+eps.n*v.d,d:v.d*eps.d})&&le({n:v.n*eps.d-eps.n*v.d,d:v.d*eps.d},b),'reference overlap');return{a,b};}
async function main(){
 const refs=JSON.parse(readFileSync('scripts/fixtures/schmidt-reference.json','utf8')) as {cases:{label:string;expected:{cuts:{characteristicPolynomial:string[];spectrum:{weight:string;multiplicity:number}[]}[]}}[]};
 const cuts=refs.cases.flatMap(r=>r.expected.cuts.map(c=>({...c,label:r.label}))),logs=JSON.parse(readFileSync('scripts/fixtures/schmidt-wire-log-reference.json','utf8')).cases as {numerator:string;denominator:string;logarithm:string}[];
 const code=await build({entryPoints:['src/lib/learning-many-body-schmidt-spectral-wire.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'iife',globalName:'schmidtSpectralWire',target:'es2020'});
 writeFileSync(`${runtime}/native-spectral-bundle.js`,code.outputFiles[0].text);
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/__many_body_schmidt_spectral__/',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><html lang="en"><title>Schmidt independent spectral certificates</title><body>Verification</body></html>'}));
  await page.goto('http://localhost:4175/__many_body_schmidt_spectral__/');await page.addScriptTag({content:code.outputFiles[0].text});
  const input=cuts.map(c=>c.characteristicPolynomial.map(s=>{const v=parse(s);return{n:String(v.n),d:String(v.d)};}));
  const result=await page.evaluate(({input,logs})=>{
   type F={n:bigint;d:bigint};type R={lower:F;upper:F};
   const api=(globalThis as unknown as {schmidtSpectralWire:{schmidtSpectrumCertificate:(p:F[])=>{roots:(R&{multiplicity:number})[]};schmidtLogCertificate:(x:F)=>R}}).schmidtSpectralWire;
   const roots=input.map(p=>{const start=performance.now(),cert=api.schmidtSpectrumCertificate(p.map(v=>({n:BigInt(v.n),d:BigInt(v.d)}))),milliseconds=performance.now()-start;if(!Object.isFrozen(cert)||!Object.isFrozen(cert.roots))throw Error('Certificate not frozen');return{roots:cert.roots.map(r=>({lower:{n:String(r.lower.n),d:String(r.lower.d)},upper:{n:String(r.upper.n),d:String(r.upper.d)},multiplicity:r.multiplicity})),milliseconds};});
   const logValues=logs.map(v=>{const r=api.schmidtLogCertificate({n:BigInt(v.numerator),d:BigInt(v.denominator)});return{lower:{n:String(r.lower.n),d:String(r.lower.d)},upper:{n:String(r.upper.n),d:String(r.upper.d)}};});return{roots,logValues};
  },{input,logs});
  let roots=0;for(const[i,cut]of cuts.entries()){const actual=result.roots[i].roots;assert.equal(actual.length,cut.spectrum.length);actual.forEach((r,j)=>{assert.equal(r.multiplicity,cut.spectrum[j].multiplicity);const{a,b}=compare(r,cut.spectrum[j].weight,240);assert(le(zero,a)&&le(b,one)&&le(mul(sub(b,a),f(BigInt(1)<<BigInt(4096))),one));roots++;});}
  result.logValues.forEach((r,i)=>{const{a,b}=compare(r,logs[i].logarithm,450);assert(le(sub(b,a),f(BigInt(1),BigInt(10)**BigInt(450))));});assert.deepEqual(errors,[]);
  const report={nativeBrowserPreparations:refs.cases.length,nativeBrowserCuts:cuts.length,nativeBrowserRootCertificates:roots,nativeBrowserLogarithms:logs.length,maximumBrowserSpectralMilliseconds:Math.max(...result.roots.map(r=>r.milliseconds)),spectralMethod:'square-free-rational-factors-interval-Descartes-sign-variation',rootIsolationBits:4096,logarithmMethod:'direct-atanh-series600-positive-tail-bound',logarithmWorkBits:1536,fullSchmidtResponseVerified:false,actualReaderVerified:false,productionNextWorkerAssetVerified:false,errors,cases:cuts.map((c,i)=>({label:c.label,milliseconds:result.roots[i].milliseconds}))};
  writeFileSync(`${runtime}/native-verification.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({...report,cases:undefined}));
 }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
