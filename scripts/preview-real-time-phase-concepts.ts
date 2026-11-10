/** Local concept review only; the unfinished unit is not registered as a reader. */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,existsSync,mkdirSync} from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {build} from 'esbuild';
const root=path.resolve('.runtime/learning/real-time-phase-concepts'),out=path.resolve('out');
async function main(){
  mkdirSync(root,{recursive:true});
  const html=readFileSync(path.join(out,'book/interactive/index.html'),'utf8');
  const styles=[...html.matchAll(/<link[^>]*>/g)].map(x=>x[0]).filter(x=>x.includes('rel="stylesheet"')).join('');
  const css=readFileSync('src/styles/qubit-lab.css','utf8');
  const result=await build({stdin:{contents:`
import React,{useState}from'react';import{createRoot}from'react-dom/client';
import{LanguageProvider,useLanguage}from'./src/components/language-provider';
import Concept from'./src/components/learning/real-time-phase-concept';
import Cover from'./src/components/learning/real-time-phase-cover';
function App(){const{language,setLanguage}=useLanguage(),[step,setStep]=useState(0);const controlled=new URLSearchParams(location.search).has('controlled');
return <><nav aria-label={language==='ko'?'미리보기 설정':'Review controls'} className="qubit-actions"><button onClick={()=>setLanguage('ko')}>한국어</button><button onClick={()=>setLanguage('en')}>English</button><button onClick={()=>document.documentElement.classList.remove('dark')}>Light</button><button onClick={()=>document.documentElement.classList.add('dark')}>Dark</button></nav>
<main className="ai-book"><h1>{language==='ko'?'실시간 위상·고전극한 — 개념도 검토':'Real-time phase and classical limits — concept review'}</h1>
<div className="ai-book-entry" data-real-time-cover style={{maxWidth:360}}><Cover/></div>
<Concept language={language} {...(controlled?{selectedStep:step,onStep:setStep}:{})}/>
<section id="experiment"><h2>{language==='ko'?'다음 제작 단계':'Next authoring stage'}</h2><p>{language==='ko'?'계산 실험 화면은 제작 중입니다. 위 그림은 현재 입력의 결과가 아닌 고정 기준입니다.':'The calculation interface is still being authored. The diagrams above are fixed references, not results for a current input.'}</p></section></main></>;}
createRoot(document.getElementById('root')).render(<LanguageProvider><App/></LanguageProvider>);`,resolveDir:process.cwd(),loader:'tsx'},
    bundle:true,write:false,platform:'browser',format:'esm',jsx:'automatic',metafile:true,define:{'process.env.NODE_ENV':'"production"'}});
  const graph=Object.keys(result.metafile!.inputs);
  assert(!graph.some(p=>/learning-real-time-phase(?:\.ts|-exact|-validation|-controller|-evaluator)|learning-quantum-hilbert-core|three\//.test(p)));
  writeFileSync(path.join(root,'review-app.js'),result.outputFiles[0].text);writeFileSync(path.join(root,'review-graph.json'),JSON.stringify(result.metafile,null,2)+'\n');
  const page=`<!doctype html><html lang="ko"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Real-time phase concept review</title>${styles}<style>${css}body{padding:clamp(.5rem,3vw,2rem)}main{max-width:1100px;margin:auto}h1{font-size:2rem;font-weight:700}nav button{border:1px solid hsl(var(--border));border-radius:.4rem;padding:.4rem}</style></head><body><div id="root"></div><script>globalThis.__name=fn=>fn;try{if(localStorage.getItem('theme')==='dark')document.documentElement.classList.add('dark');}catch{}</script><script type="module" src="/review-app.js"></script></body></html>`;
  writeFileSync(path.join(root,'review.html'),page);
  if(process.argv.includes('--build-only')){console.log('Concept review assets rebuilt.');return;}
  const server=http.createServer((req,res)=>{
    try{
      const url=decodeURIComponent(new URL(req.url??'/','http://localhost').pathname);
      if(url==='/'){res.setHeader('Content-Type','text/html; charset=utf-8');res.end(page);return;}
      const file=url==='/review-app.js'?path.join(root,'review-app.js'):url.startsWith('/_next/')||url.startsWith('/assets/')?path.resolve(out,'.'+url):null;
      if(!file||!file.startsWith((url==='/review-app.js'?root:out)+path.sep)||!existsSync(file)){res.writeHead(404);res.end('Not found');return;}
      const extension=path.extname(file);res.setHeader('Content-Type',extension==='.js'?'text/javascript':extension==='.css'?'text/css':extension==='.woff2'?'font/woff2':'application/octet-stream');res.setHeader('Cache-Control','no-store');res.end(readFileSync(file));
    }catch{res.writeHead(400);res.end('Bad request');}
  });
  server.listen(0,'127.0.0.1',()=>{const address=server.address();assert(address&&typeof address==='object');const info={url:`http://127.0.0.1:${address.port}/`,pid:process.pid,readerRegistered:false};writeFileSync(path.join(root,'review-server.json'),JSON.stringify(info,null,2)+'\n');console.log(JSON.stringify(info));});
}
main().catch(error=>{console.error(error);process.exitCode=1;});
