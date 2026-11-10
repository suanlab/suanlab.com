/** Isolated review using actual calculation and independent-validation Workers. */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,existsSync,mkdirSync} from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {build} from 'esbuild';
const root=path.resolve('.runtime/learning/real-time-phase-state'),out=path.resolve('out'),publicRoot=path.join(root,'public');
async function main(){
  mkdirSync(publicRoot,{recursive:true});
  for(const [entry,name]of [['src/lib/learning-real-time-phase.worker.ts','calculate'],['src/lib/learning-real-time-phase-validation.worker.ts','validate']]){
    const worker=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',target:'es2020',metafile:true});writeFileSync(path.join(publicRoot,name+'.js'),worker.outputFiles[0].text);
  }
  const app=await build({stdin:{contents:`import React from'react';import{createRoot}from'react-dom/client';import{useState}from'react';import{LanguageProvider,useLanguage}from'./src/components/language-provider';import Lab from'./src/components/learning/real-time-phase-lab';
function App(){const{language,setLanguage}=useLanguage(),[show,setShow]=useState(true);return <><nav className="qubit-actions" aria-label="Review controls"><button onClick={()=>setLanguage('ko')}>한국어</button><button onClick={()=>setLanguage('en')}>English</button><button onClick={()=>document.documentElement.classList.remove('dark')}>Light</button><button onClick={()=>document.documentElement.classList.add('dark')}>Dark</button><button onClick={()=>setShow(v=>!v)}>Toggle lab</button></nav><main className="ai-book"><h1>{language==='ko'?'실시간 위상 학습 화면 검토':'Real-time phase learning review'}</h1>{show&&<Lab language={language}/>}</main></>;}
createRoot(document.getElementById('root')).render(<LanguageProvider><App/></LanguageProvider>);`,resolveDir:process.cwd(),loader:'tsx'},bundle:true,write:true,splitting:true,outdir:publicRoot,entryNames:'review-app',chunkNames:'chunks/[name]-[hash]',platform:'browser',format:'esm',jsx:'automatic',metafile:true,define:{'process.env.NODE_ENV':'"production"'}});
  const inputs=Object.keys(app.metafile!.inputs);assert(!inputs.some(p=>/learning-real-time-phase(?:\.ts|-exact|-validation\.ts|-evaluator)|learning-quantum-hilbert-core/.test(p)));writeFileSync(path.join(root,'reader-graph.json'),JSON.stringify(app.metafile,null,2)+'\n');
  const html=readFileSync(path.join(out,'book/interactive/index.html'),'utf8'),styles=[...html.matchAll(/<link[^>]*>/g)].map(x=>x[0]).filter(x=>x.includes('rel="stylesheet"')).join('');
  const css=['src/styles/qubit-lab.css','src/styles/scalar-field-lab.css','src/styles/information-lab.css','src/styles/interactive-book.css'].map(p=>readFileSync(p,'utf8')).join('\n');
  const page=`<!doctype html><html lang="ko"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Real-time guided lessons review</title>${styles}<style>${css}body{padding:clamp(.5rem,3vw,2rem)}main{max-width:1000px;margin:auto}h1{font-size:2rem;font-weight:700}nav button,main>button{border:1px solid hsl(var(--border));padding:.5rem}select{max-width:100%;font-size:16px;background:hsl(var(--background));color:hsl(var(--foreground))}</style></head><body><div id="root"></div><script>globalThis.__name=fn=>fn;try{if(localStorage.getItem('theme')==='dark')document.documentElement.classList.add('dark');}catch{}</script><script type="module" src="/review-app.js"></script></body></html>`;
  const server=http.createServer((req,res)=>{try{
    const url=decodeURIComponent(new URL(req.url??'/','http://localhost').pathname);
    if(url==='/'){res.setHeader('Content-Type','text/html; charset=utf-8');res.end(page);return;}
    const prefix=url.startsWith('/_next/')||url.startsWith('/assets/')?out:publicRoot;
    const mapped=path.posix.basename(url)==='learning-real-time-phase.worker.ts'?'/calculate.js':path.posix.basename(url)==='learning-real-time-phase-validation.worker.ts'?'/validate.js':url;
    const file=path.resolve(prefix,'.'+mapped);if(!file.startsWith(prefix+path.sep)||!existsSync(file)){res.writeHead(404);res.end('Not found');return;}
    res.setHeader('Content-Type',/\.js$/.test(file)?'text/javascript':/\.css$/.test(file)?'text/css':'application/octet-stream');res.setHeader('Cache-Control','no-store');res.end(readFileSync(file));
  }catch{res.writeHead(400);res.end('Bad request');}});
  server.listen(0,'127.0.0.1',()=>{const address=server.address();assert(address&&typeof address==='object');const info={url:`http://127.0.0.1:${address.port}/`,pid:process.pid};writeFileSync(path.join(root,'review-server.json'),JSON.stringify(info,null,2)+'\n');console.log(JSON.stringify(info));});
}
main().catch(error=>{console.error(error);process.exitCode=1;});
