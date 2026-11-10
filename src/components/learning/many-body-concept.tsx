'use client';
import {useId,useState} from 'react';
import {manyBodyTeaching} from '@/data/many-body-teaching';
import type {ManyBodyTeachingId} from '@/lib/learning-many-body-teaching';
import {MasterButton as Button} from './master-equation-view';
import ManyBodyCover from './many-body-cover';
export default function ManyBodyConcept({kind,language}:{kind:ManyBodyTeachingId;language:'ko'|'en'}){
 const{guide}=manyBodyTeaching.find(item=>item.unit.id===kind)!,[step,setStep]=useState(0),[revealed,setRevealed]=useState(false),id=useId(),s=guide.steps[step],g=guide.experiments[step],t=(ko:string,en:string)=>language==='ko'?ko:en;
 const change=(i:number)=>{setStep(i);setRevealed(false);};
 return <section className="ai-intuition" id="visual-intuition" aria-labelledby={id} data-many-body-concept={kind} data-many-body-concept-step={step}>
 <div className="ai-section-heading"><div><span className="ai-eyebrow">BUILD YOUR INTUITION</span><h2 id={id}>{t('그림·예측·실험으로 연결하기','Connect diagrams,predictions and experiments')}</h2></div><p>{guide.intuition[language]}</p></div>
 <div className="ai-concept-steps" role="group" aria-label={t('개념 단계','Concept steps')}>{guide.steps.map((s,i)=><Button key={i} variant="outline" aria-pressed={step===i} onClick={()=>change(i)}>{i+1}. {s.title[language]}</Button>)}</div>
 <div className="ai-concept-explanation"><h3>{s.title[language]}</h3><p>{s.explanation[language]}</p><p className="ai-concept-equation" style={{overflowWrap:'anywhere'}}>{s.formula}</p><p>{s.diagram[language]}</p><ManyBodyCover kind={kind} step={step}/><p className="ai-small">{t('그림은 개념 모식도입니다. 현재 입력의 수치는 아래 실습에서 확인하세요.','The diagram is schematic. Inspect current-input numbers in the experiment below.')}</p></div>
 <div className="ai-concept-explanation"><h3>{t('먼저 예측','Predict first')} · {g.title[language]}</h3><p>{g.predict[language]}</p><p>{g.action[language]}</p><Button variant="outline" aria-expanded={revealed} onClick={()=>setRevealed(v=>!v)}>{revealed?t('해설 접기','Hide explanation'):t('예측 후 해설 확인','Read explanation after predicting')}</Button>{revealed&&<p data-many-body-concept-explanation>{g.explain[language]}</p>}</div>
 <div className="ai-concept-pagination"><Button variant="outline" disabled={!step} onClick={()=>change(step-1)}>{t('이전 단계','Previous step')}</Button><Button variant="outline" disabled={step===guide.steps.length-1} onClick={()=>change(step+1)}>{t('다음 단계','Next step')}</Button><a href="#experiment">{t('실습에서 예제 비교하기 →','Compare examples in the experiment →')}</a></div>
 </section>;
}
