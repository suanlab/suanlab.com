'use client';
import {useId,useState} from 'react';
import {Button} from '@/components/ui/button';
import {qubitTeaching} from '@/data/qubit-teaching';
import type {QubitTeachingId} from '@/lib/learning-qubit-teaching-input';
import QubitConceptDiagram from './qubit-concept-diagram';
type Props={kind:QubitTeachingId;language:'ko'|'en';experimentHref?:string};
export default function QubitConcept(props:Props){return <QubitConceptSteps key={props.kind} {...props}/>;}
function QubitConceptSteps({kind,language,experimentHref}:Props){
 const {guide}=qubitTeaching.find(item=>item.unit.id===kind)!,[step,setStep]=useState(0),[revealed,setRevealed]=useState(false),id=useId(),current=guide.steps[step],experiment=guide.experiments[step],t=(ko:string,en:string)=>language==='ko'?ko:en;
 const change=(i:number)=>{setStep(i);setRevealed(false);};
 return <section className="ai-intuition" id="visual-intuition" aria-labelledby={id} data-qubit-concept={kind} data-qubit-concept-step={step}>
 <div className="ai-section-heading"><div><span className="ai-eyebrow">BUILD YOUR INTUITION</span><h2 id={id}>{t('그림에서 예측하고, 수치로 확인하기','Predict from the diagram, then check the numbers')}</h2></div><p>{guide.intuition[language]}</p></div>
 <div className="ai-concept-steps" role="group" aria-label={t('개념 단계','Concept steps')}>{guide.steps.map((s,i)=><Button key={i} variant="outline" type="button" aria-pressed={step===i} style={step===i?{fontWeight:600}:undefined} onClick={()=>change(i)}>{i+1}. {s.title[language]}</Button>)}</div>
 <article className="ai-concept-explanation"><h3>{current.title[language]}</h3><p>{current.explanation[language]}</p><p className="ai-concept-equation" style={{overflowWrap:'anywhere'}}>{current.formula}</p><p>{current.diagram[language]}</p><QubitConceptDiagram kind={kind} step={step} language={language} heading={current.title[language]} description={current.diagram[language]}/></article>
 <article className="ai-concept-explanation"><h3>{t('먼저 예측','Predict first')} · {experiment.title[language]}</h3><p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><Button variant="outline" type="button" aria-expanded={revealed} aria-controls={`${id}-explanation`} style={{whiteSpace:'normal',height:'auto',minHeight:40,maxWidth:'100%'}} onClick={()=>setRevealed(v=>!v)}>{revealed?t('해설 접기','Hide explanation'):t('예측 후 해설 확인','Read explanation after predicting')}</Button><p id={`${id}-explanation`} hidden={!revealed} data-qubit-concept-explanation={revealed?'shown':undefined}>{experiment.explain[language]}</p></article>
 <div className="ai-concept-pagination"><Button variant="outline" type="button" disabled={step===0} onClick={()=>change(step-1)}>{t('이전 단계','Previous step')}</Button><Button variant="outline" type="button" disabled={step===guide.steps.length-1} onClick={()=>change(step+1)}>{t('다음 단계','Next step')}</Button>{experimentHref&&<a href={experimentHref}>{t('안내 실험에서 비교하기 →','Compare in the guided experiment →')}</a>}</div>
 </section>;
}
