'use client';
import {useEffect,useId,useState} from 'react';
import {Button} from '@/components/ui/button';
import {relativeExamples,readRelativeDraft,type RelativeInput} from '@/lib/learning-information-relative-input';
import {quantumInformationInputKey} from '@/lib/learning-quantum-information-input';
import {relativeSteps,relativeExperiments,relativeQuestions} from '@/data/information-relative-teaching';
import {useSavedInformationInput} from './use-saved-information-input';
import {useInformationCalculation} from './use-information-calculation';
import InformationRelativeResult from './information-relative-result';
export default function InformationRelativeLab({language,embedded=false}:{language:'ko'|'en';embedded?:boolean}){
 const t=(ko:string,en:string)=>language==='ko'?ko:en,id=useId(),saved=useSavedInformationInput('relative-entropy',relativeExamples[0].input),[running,setRunning]=useState(true);
 const calculation=useInformationCalculation(saved.input,saved.ready&&running),[step,setStep]=useState(0),[experiment,setExperiment]=useState(0),[showExplanation,setShowExplanation]=useState(false);
 const[draft0,setDraft0]=useState<string[]>([...saved.input.rho]),[draft1,setDraft1]=useState<string[]>([...saved.input.sigma]),[invalid,setInvalid]=useState(false),[answers,setAnswers]=useState<Record<number,number>>({});
 useEffect(()=>{setDraft0([...saved.input.rho]);setDraft1([...saved.input.sigma]);setInvalid(false);},[saved.input]);
 const choose=(input:RelativeInput)=>{saved.setInput(input);setRunning(true);calculation.retry();};
 const currentKey=quantumInformationInputKey(saved.input),example=relativeExamples.find(p=>quantumInformationInputKey(p.input)===currentKey),guide=relativeExperiments[experiment];
 const error=calculation.error;
 return <section className="ai-qubit-lab information-lab" lang={language} aria-labelledby={`${id}-title`} data-information-lab="relative" data-information-phase={calculation.status}>
  <h2 id={`${id}-title`}>{t('실제 상태와 참조의 차이를 처리 전후로 비교하기','Compare actual and reference states before and after processing')}</h2>
  <p>{t('ρ와 σ의 순서를 정한 뒤 두 상태에 같은 Z 탈위상을 적용합니다. 유한/무한 분류를 먼저 확인하고 그림·스펙트럼·구간으로 이유를 설명하세요.','Order ρ and σ, then apply the same Z dephasing to both. Read finite/infinite classification first, then explain it through projections, spectra and intervals.')}</p>
  {!embedded&&<>  <nav className="qubit-actions" aria-label={t('개념 단계','Concept steps')}>{relativeSteps.map((s,i)=><Button key={i} variant="outline" aria-pressed={step===i} onClick={()=>setStep(i)}>{i+1}. {s.title[language]}</Button>)}</nav>
  <div className="information-step" key={step} aria-live="polite"><h3>{relativeSteps[step].title[language]}</h3><p>{relativeSteps[step].body[language]}</p><p className="information-equation">{relativeSteps[step].formula}</p></div></>}
  <h3>{t('예측하고 비교하기','Predict and compare')}</h3><label className="qubit-field" htmlFor={`${id}-guide`}>{t('안내 실험','Guided experiment')}<select id={`${id}-guide`} aria-label={t('안내 실험','Guided experiment')} value={experiment} onChange={e=>{setExperiment(Number(e.target.value));setShowExplanation(false);}}>{relativeExperiments.map((g,i)=><option key={i} value={i}>{g.title[language]}</option>)}</select></label>
  <p><strong>{t('먼저 예측','Predict first')}: </strong>{guide.predict[language]}</p>
  <div className="qubit-actions" role="group" aria-label={t('안내 예제','Guided examples')}>{guide.examples.map(key=>{const p=relativeExamples.find(x=>x.id===key)!;return <Button key={key} variant="outline" aria-pressed={example?.id===key} onClick={()=>choose(p.input)}>{p.label[language]}</Button>;})}</div>
  <Button variant="outline" aria-expanded={showExplanation} onClick={()=>setShowExplanation(v=>!v)}>{t('해설 보기','Show explanation')}</Button>{showExplanation&&<p data-information-guide-explanation>{guide.explanation[language]}</p>}
  <label className="qubit-field" htmlFor={`${id}-example`}>{t('전체 예제','All examples')}<select id={`${id}-example`} aria-label={t('전체 예제','All examples')} disabled={!saved.ready} value={example?.id??'custom'} onChange={e=>{const next=relativeExamples.find(p=>p.id===e.target.value);if(next)choose(next.input);}}><option value="custom" disabled>{t('직접 입력','Custom input')}</option>{relativeExamples.map(p=><option key={p.id} value={p.id}>{p.label[language]}</option>)}</select></label>
  {!saved.ready&&<p role="status">{t('저장된 입력을 확인하고 있습니다.','Checking the saved input.')}</p>}
  {saved.storage!=='available'&&<p role="status">{saved.storage==='corrupt'?t('저장된 입력 형식이 올바르지 않아 첫 예제로 복원했습니다.','The saved input format was invalid; the first example was restored.'):t('이 브라우저에서 저장할 수 없습니다. 현재 화면에서는 계속 실험할 수 있습니다.','Storage is unavailable. You can continue experimenting in this session.')}</p>}
  <form onSubmit={e=>{e.preventDefault();const next=readRelativeDraft(draft0,draft1);if(next){choose(next);setInvalid(false);}else setInvalid(true);}}>
   <h3>{t('실제 상태와 참조 직접 바꾸기','Edit actual and reference states')}</h3><p>{t('각 Bloch 좌표는0, −1/2, 3/5 같은 정수·분수로 적으세요. 두 벡터의 길이는 각각1이하이어야 합니다. 수정 후 적용하면 같은 채널을 두 상태에 자동으로 적용합니다.','Use Bloch coordinates such as0, −1/2 or3/5. Each vector must have length at most1. Applying edits automatically applies the same channel to both states.')}</p>
   <div className="information-inputs">{[draft0,draft1].map((values,state)=><fieldset key={state}><legend>{state===0?t('실제 상태 ρ','Actual state ρ'):t('참조 상태 σ','Reference state σ')}</legend>{values.map((value,axis)=><label className="qubit-field" key={axis} htmlFor={`${id}-r${state}-${axis}`}>{`${state===0?'ρ':'σ'} · ${['x','y','z'][axis]}`}<input id={`${id}-r${state}-${axis}`} value={value} maxLength={66} autoComplete="off" spellCheck={false} onChange={e=>{const update=state===0?setDraft0:setDraft1;update(v=>v.map((x,i)=>i===axis?e.target.value:x));setInvalid(false);}}/></label>)}</fieldset>)}</div>
   {invalid&&<p role="alert">{t('좌표 형식을 확인하세요. 소수 대신 분수를 사용하세요. 현재 결과는 이전 적용 입력의 결과입니다.','Check coordinate formats. Use fractions rather than decimals. The current result still belongs to the previously applied input.')}</p>}
   <Button type="button" variant="outline" onClick={()=>{setDraft0(draft1);setDraft1(draft0);setInvalid(false);}}>{t('실제·참조 초안 맞바꾸기','Swap actual and reference drafts')}</Button>
   <div className="qubit-actions"><Button type="submit" disabled={!saved.ready}>{t('입력 적용','Apply input')}</Button><Button type="button" variant="outline" onClick={()=>choose(relativeExamples[0].input)}>{t('첫 예제로 초기화','Reset to first example')}</Button></div>
  </form>
  <div className="qubit-actions">{calculation.status==='pending'&&<Button variant="outline" onClick={()=>{setRunning(false);calculation.cancel();}}>{t('계산 취소','Cancel calculation')}</Button>}{(error||!running)&&<Button onClick={()=>{setRunning(true);calculation.retry();}}>{t('다시 계산','Recalculate')}</Button>}</div>
  <div role={error?'alert':'status'} aria-live="polite" className="qubit-status">{error?error==='invalid-input'?t('입력의 수학 조건이 맞지 않습니다. 각 벡터 길이≤1을 확인하거나 예제로 돌아가세요.','The input does not satisfy the mathematical conditions. Check each vector length≤1, or return to an example.'):t('계산을 완료하지 못했습니다. 이전 결과를 새 입력의 결과로 표시하지 않습니다. 다시 계산하거나 다른 예제를 선택하세요.','The calculation could not complete. An earlier result is not shown for the new input. Recalculate or select another example.'):!running?t('계산을 취소했습니다.','Calculation cancelled.'):calculation.status==='ready'?t('현재 입력의 계산과 독립 검산을 마쳤습니다.','Calculation and independent checks for the current input are complete.'):calculation.phase==='validating'?t('결과를 독립적으로 검산하고 있습니다.','Independently checking the result.'):t('현재 입력을 계산하고 있습니다.','Calculating the current input.')}</div>
  {calculation.response?.kind==='relative-entropy'&&<InformationRelativeResult key={currentKey} response={calculation.response} language={language}/>}
  {!embedded&&<>  <h3>{t('이해 확인','Check your understanding')}</h3>{relativeQuestions.map((q,i)=><fieldset className="information-question" key={i}><legend>{i+1}. {q.prompt[language]}</legend>{q.options.map((o,j)=><label key={j}><input type="radio" name={`${id}-quiz${i}`} checked={answers[i]===j} onChange={()=>setAnswers(v=>({...v,[i]:j}))}/>{o[language]}</label>)}{answers[i]!==undefined&&<p role="status"><strong>{answers[i]===q.correct?t('맞습니다. ','Correct. '):t('다시 생각해 보세요. ','Think again. ')}</strong>{q.explanation[language]}</p>}</fieldset>)}</>}
 </section>;
}
