'use client';
import React, {useId} from 'react';
import {Button} from '@/components/ui/button';
import {realTimePhaseSteps} from '@/data/real-time-phase-teaching';
import RealTimePhaseGraphic from './real-time-phase-graphic';
import {useSavedState} from './use-saved-state';
const validStep=(value:unknown):value is number=>typeof value==='number'&&Number.isInteger(value)&&value>=0&&value<realTimePhaseSteps.length;
export default function RealTimePhaseConcept({language,selectedStep,onStep}:{language:'ko'|'en';selectedStep?:number;onStep?:(step:number)=>void}) {
  const id=useId(),[saved,setSaved,restored]=useSavedState('suanlab-learning:real-time-phase-and-classical-limit:concept',0,validStep);
  const candidate=selectedStep??saved,step=validStep(candidate)?candidate:0,current=realTimePhaseSteps[step];
  const ready=onStep?true:restored,setStep=onStep??setSaved,t=(ko:string,en:string)=>language==='ko'?ko:en;
  return <section id="visual-intuition" className="ai-intuition" aria-labelledby={id} data-real-time-concept={step} data-real-time-concept-ready={ready?'true':'false'}>
    <h2 id={id}>{t('그림으로 실시간 위상·간섭·고전극한 이해하기','Understand real-time phase, interference and classical limits visually')}</h2>
    <p>{t('복소수 화살표부터 초점·파동묶음·Airy까지 차례로 연결합니다. 모든 그림은 고정된 설명용 기준이며 현재 입력의 검산 결과와 독립적입니다.','Connect complex arrows to foci, wave packets and Airy behavior. Every diagram is a fixed teaching reference, independent of the verified result for your current input.')}</p>
    <div className="ai-concept-steps" role="group" aria-label={t('개념 단계','Concept steps')}>
      {realTimePhaseSteps.map((item,i)=><Button key={item.id} type="button" variant="outline" disabled={!ready} aria-pressed={step===i} onClick={()=>setStep(i)}>{i+1}. {item.title[language]}</Button>)}
    </div>
    <article className="ai-concept-explanation" aria-live="polite" aria-atomic="true">
      <h3>{current.title[language]}</h3><p>{current.body[language]}</p>
      <p className="ai-concept-equation" style={{overflowWrap:'anywhere'}}>{current.formula}</p>
      <figure><div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('고정 기준 개념도 · 좌우 이동 가능','Fixed reference diagram; scroll horizontally')}><RealTimePhaseGraphic step={step} language={language}/></div><figcaption>{current.diagram[language]}</figcaption></figure>
    </article>
    <div className="qubit-actions">
      <Button type="button" variant="outline" disabled={!ready||step===0} onClick={()=>setStep(step-1)}>{t('이전 단계','Previous step')}</Button>
      <Button type="button" variant="outline" disabled={!ready||step===realTimePhaseSteps.length-1} onClick={()=>setStep(step+1)}>{t('다음 단계','Next step')}</Button>
      <a href="#experiment">{t('실험에서 비교하기 →','Compare in the experiment →')}</a>
    </div>
  </section>;
}
