'use client';
import React,{useId,useState} from 'react';
import {Button} from '@/components/ui/button';
import {realTimePhaseGuides,realTimePhaseGuidePreparations,realTimePhaseQuestions} from '@/data/real-time-phase-teaching';
import {readRealTimePhaseInput,type RealTimePhaseInput} from '@/lib/learning-real-time-phase-input';
import {emptyRealTimeLesson,readRealTimeLesson,type RealTimeLessonState} from '@/lib/learning-real-time-phase-lesson-state';
export const realTimeLessonLimits={guides:realTimePhaseGuides.length,options:realTimePhaseQuestions.map(q=>q.options.length)};
/** Guide/answer changes are pure learning actions; only Apply requests a model. */
export default function RealTimePhaseLessons({language,onApply,state,onState,questionsVisible=true}:{language:'ko'|'en';onApply:(input:RealTimePhaseInput,id:string)=>void;state?:RealTimeLessonState;onState?:(value:RealTimeLessonState)=>void;questionsVisible?:boolean}){
  const id=useId(),t=(ko:string,en:string)=>language==='ko'?ko:en,[local,setLocal]=useState(()=>emptyRealTimeLesson(realTimeLessonLimits));
  const current=readRealTimeLesson(state??local,realTimeLessonLimits)??emptyRealTimeLesson(realTimeLessonLimits),set=onState??setLocal,guide=realTimePhaseGuides[current.guide];
  const choose=(name:string)=>{const prep=realTimePhaseGuidePreparations.find(p=>p.id===name),input=prep&&readRealTimePhaseInput(prep.input);if(input)onApply(input,name);};
  return <div className="information-lab" data-real-time-lessons>
    <section aria-labelledby={`${id}-guides`}><h2 id={`${id}-guides`}>{t('먼저 예측하고 비교하기','Predict first, then compare')}</h2>
      <p>{t('안내 선택·예측 작성·문제 풀이는 계산을 시작하지 않습니다. 예제 적용 버튼만 실제 입력을 바꾸며, 결과는 계산·독립 검산 후 나타납니다.','Choosing a guide, writing a prediction or answering a question does not start a calculation. Only Apply changes the model input; results appear after calculation and independent verification.')}</p>
      <label className="qubit-field">{t('안내 실험','Guided experiment')}<select aria-label={t('안내 실험','Guided experiment')} value={current.guide} onChange={e=>set({...current,guide:Number(e.target.value),explanationOpen:false})}>{realTimePhaseGuides.map((g,i)=><option key={g.id} value={i}>{i+1}. {g.title[language]}</option>)}</select></label>
      <p data-real-time-prediction-prompt>{guide.predict[language]}</p>
      <label className="qubit-field">{t('내 예측 · 선택 사항','My prediction · optional')}<textarea aria-label={t('내 예측','My prediction')} maxLength={2000} rows={3} value={current.predictions[current.guide]} onChange={e=>set({...current,predictions:current.predictions.map((v,i)=>i===current.guide?e.target.value:v)})}/></label>
      <div className="qubit-actions ai-guided-choices" role="group" aria-label={t('안내 예제 적용','Apply guided examples')}>{guide.preparations.map(name=>{const prep=realTimePhaseGuidePreparations.find(p=>p.id===name)!;return <Button key={name} type="button" variant="outline" onClick={()=>choose(name)} data-real-time-preparation={name}>{t('적용: ','Apply: ')}{prep.label[language]}</Button>;})}</div>
      <p data-real-time-observe>{guide.observe[language]}</p>
      <Button type="button" variant="outline" aria-expanded={current.explanationOpen} aria-controls={`${id}-explanation`} onClick={()=>set({...current,explanationOpen:!current.explanationOpen})}>{current.explanationOpen?t('해설 숨기기','Hide explanation'):t('해설 보기','Show explanation')}</Button>
      {current.explanationOpen&&<p id={`${id}-explanation`} data-real-time-guide-explanation>{guide.explanation[language]}</p>}
    </section>
    {questionsVisible&&<section aria-labelledby={`${id}-questions`}><h2 id={`${id}-questions`}>{t('이해 확인과 해설','Check your understanding with explanations')}</h2>
      {realTimePhaseQuestions.map((q,i)=><fieldset key={q.id} className="information-question" data-real-time-question={q.id}><legend>{i+1}. {q.prompt[language]}</legend>{q.options.map((option,j)=><label key={j}><input type="radio" name={`${id}-question-${i}`} value={j} checked={current.answers[i]===j} onChange={()=>set({...current,answers:current.answers.map((v,k)=>k===i?j:v)})}/>{option[language]}</label>)}{current.answers[i]>=0&&<p role="status" data-real-time-answer={current.answers[i]===q.correct?'correct':'incorrect'}><strong>{current.answers[i]===q.correct?t('맞습니다. ','Correct. '):t('다시 생각해보세요. ','Think again. ')}</strong>{q.explanation[language]}</p>}</fieldset>)}
      <Button type="button" variant="outline" onClick={()=>set({...current,answers:current.answers.map(()=>-1)})}>{t('문제 다시 풀기','Reset answers')}</Button>
    </section>}
  </div>;
}
