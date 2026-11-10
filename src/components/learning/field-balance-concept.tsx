'use client';
import {useId,useState} from 'react';
import {Button} from '@/components/ui/button';
import {fieldBalanceReader} from '@/data/field-balance-reader';
import FieldBalanceGraphic from './field-balance-graphic';
export default function FieldBalanceConcept({language}:{language:'ko'|'en'}){
 const{guide}=fieldBalanceReader,[step,setStep]=useState(0),id=useId(),current=guide.steps[step],t=(ko:string,en:string)=>language==='ko'?ko:en;
 return <section id="visual-intuition" className="ai-intuition" aria-labelledby={id} data-balance-concept={step}>
  <div className="ai-section-heading"><div><span className="ai-eyebrow">BUILD YOUR INTUITION</span><h2 id={id}>{t('그림으로 국소 흐름과 대칭 구별하기','Distinguish local currents and symmetry')}</h2></div><p>{guide.intuition[language]}</p></div>
  <div className="ai-concept-steps" role="group" aria-label={t('개념 단계','Concept steps')}>{guide.steps.map((s,i)=><Button key={i} variant="outline" aria-pressed={step===i} onClick={()=>setStep(i)}>{i+1}. {s.title[language]}</Button>)}</div>
  <article className="ai-concept-explanation"><h3>{current.title[language]}</h3><p>{current.explanation[language]}</p><p className="ai-concept-equation" style={{overflowWrap:'anywhere'}}>{current.formula}</p><figure><div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('기준 개념도 · 좌우 이동 가능','Reference diagram; scroll horizontally')}><FieldBalanceGraphic step={step} language={language}/></div><figcaption className="ai-small">{current.diagram[language]} {t('현재 실험 입력의 결과가 아닌 설명용 기준입니다.','This explanatory reference is separate from the current experiment input.')}</figcaption></figure></article>
  <div className="ai-concept-pagination"><Button variant="outline" disabled={step===0} onClick={()=>setStep(v=>v-1)}>{t('이전 단계','Previous step')}</Button><Button variant="outline" disabled={step===guide.steps.length-1} onClick={()=>setStep(v=>v+1)}>{t('다음 단계','Next step')}</Button><a href="#experiment">{t('직접 실험에서 비교하기 →','Compare in the experiment →')}</a></div>
 </section>;
}
