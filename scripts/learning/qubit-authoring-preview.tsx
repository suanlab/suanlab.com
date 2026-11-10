import {useState} from 'react';
import {createRoot} from 'react-dom/client';
import {LanguageProvider,useLanguage} from '../../src/components/language-provider';
import QubitConcept from '../../src/components/learning/qubit-concept';
import LearningQuiz from '../../src/components/learning/learning-quiz';
import {qubitTeaching} from '../../src/data/qubit-teaching';
import {isQubitTeachingId,qubitTeachingIds,type QubitTeachingId} from '../../src/lib/learning-qubit-teaching-input';
const params=new URL(location.href).searchParams,requested=params.get('unit')??'',initial=isQubitTeachingId(requested)?requested:qubitTeachingIds[0];
try{localStorage.setItem('suanlab-language',params.get('lang')==='en'?'en':'ko');}catch{/* Keep the provider's session fallback. */}
document.documentElement.classList.toggle('dark',params.get('theme')==='dark');
function Preview(){
 const {language}=useLanguage(),[kind,setKind]=useState<QubitTeachingId>(initial),item=qubitTeaching.find(x=>x.unit.id===kind)!,t=(ko:string,en:string)=>language==='ko'?ko:en;
 return <main className="ai-book" data-qubit-authoring={kind} data-language={language} style={{maxWidth:1100,margin:'auto',padding:20,minWidth:0}}>
 <label htmlFor="unit-select">{t('검토할 단원','Unit to review')}</label><select id="unit-select" value={kind} style={{width:'100%',minHeight:40,background:'hsl(var(--card))',color:'hsl(var(--foreground))',border:'1px solid hsl(var(--border))',margin:'8px 0 24px'}} onChange={e=>{if(isQubitTeachingId(e.target.value))setKind(e.target.value);}}>{qubitTeaching.map(x=><option key={x.unit.id} value={x.unit.id}>{x.unit.title[language]}</option>)}</select>
 <h1 style={{fontSize:28,lineHeight:1.35,fontWeight:700,wordBreak:'keep-all'}}>{item.unit.title[language]}</h1><p>{item.unit.subtitle[language]}</p>
 <QubitConcept kind={kind} language={language}/><LearningQuiz key={kind} unit={item.unit}/>
 </main>;
}
createRoot(document.getElementById('root')!).render(<LanguageProvider><Preview/></LanguageProvider>);
