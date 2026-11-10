'use client';
import {useEffect,useId,useState} from 'react';
import {Button} from '@/components/ui/button';
import {capacityExamples,readCapacityDraft,type CapacityInput} from '@/lib/learning-information-capacity-input';
import {quantumInformationInputKey} from '@/lib/learning-quantum-information-input';
import {capacitySteps,capacityExperiments,capacityQuestions} from '@/data/information-capacity-teaching';
import {useSavedInformationInput} from './use-saved-information-input';
import {useInformationCalculation} from './use-information-calculation';
import InformationCapacityResult from './information-capacity-result';
export default function InformationCapacityLab({language,embedded=false}:{language:'ko'|'en';embedded?:boolean}){
 const t=(ko:string,en:string)=>language==='ko'?ko:en,id=useId(),saved=useSavedInformationInput('capacity',capacityExamples[0].input),[running,setRunning]=useState(true);
 const calculation=useInformationCalculation(saved.input,saved.ready&&running),[step,setStep]=useState(0),[experiment,setExperiment]=useState(0),[showExplanation,setShowExplanation]=useState(false);
 const[draft,setDraft]=useState<string[]>([...saved.input.input]),[probability,setProbability]=useState(saved.input.probability),[invalid,setInvalid]=useState(false),[answers,setAnswers]=useState<Record<number,number>>({});
 useEffect(()=>{setDraft([...saved.input.input]);setProbability(saved.input.probability);setInvalid(false);},[saved.input]);
 const choose=(input:CapacityInput)=>{saved.setInput(input);setRunning(true);calculation.retry();};
 const currentKey=quantumInformationInputKey(saved.input),example=capacityExamples.find(p=>quantumInformationInputKey(p.input)===currentKey),guide=capacityExperiments[experiment];
 const error=calculation.error;
 return <section className="ai-qubit-lab information-lab" lang={language} aria-labelledby={`${id}-title`} data-information-lab="capacity" data-information-phase={calculation.status}>
  <h2 id={`${id}-title`}>{t('채널 용량과 선택 입력의 정보 구분하기','Distinguish channel capacity from selected-input information')}</h2>
  <p>{t('알려진 memoryless 위상 반전 채널에서 C·CE·Q와 선택 입력의 Ic를 비교합니다. 통신 과제·자원·단위를 먼저 확인하고 입력/출력·환경을 살펴보세요.','Compare C, CE, Q and selected-input Ic for a known memoryless phase-flip channel. First identify the task, resource assumptions and units, then inspect input/output and environment.')}</p>
  {!embedded&&<>  <nav className="qubit-actions" aria-label={t('개념 단계','Concept steps')}>{capacitySteps.map((s,i)=><Button key={i} variant="outline" aria-pressed={step===i} onClick={()=>setStep(i)}>{i+1}. {s.title[language]}</Button>)}</nav>
  <div className="information-step" key={step} aria-live="polite"><h3>{capacitySteps[step].title[language]}</h3><p>{capacitySteps[step].body[language]}</p><p className="information-equation">{capacitySteps[step].formula}</p></div></>}
  <h3>{t('예측하고 비교하기','Predict and compare')}</h3><label className="qubit-field" htmlFor={`${id}-guide`}>{t('안내 실험','Guided experiment')}<select id={`${id}-guide`} aria-label={t('안내 실험','Guided experiment')} value={experiment} onChange={e=>{setExperiment(Number(e.target.value));setShowExplanation(false);}}>{capacityExperiments.map((g,i)=><option key={i} value={i}>{g.title[language]}</option>)}</select></label>
  <p><strong>{t('먼저 예측','Predict first')}: </strong>{guide.predict[language]}</p>
  <div className="qubit-actions" role="group" aria-label={t('안내 예제','Guided examples')}>{guide.examples.map(key=>{const p=capacityExamples.find(x=>x.id===key)!;return <Button key={key} variant="outline" aria-pressed={example?.id===key} onClick={()=>choose(p.input)}>{p.label[language]}</Button>;})}</div>
  <Button variant="outline" aria-expanded={showExplanation} onClick={()=>setShowExplanation(v=>!v)}>{t('해설 보기','Show explanation')}</Button>{showExplanation&&<p data-information-guide-explanation>{guide.explanation[language]}</p>}
  <label className="qubit-field" htmlFor={`${id}-example`}>{t('전체 예제','All examples')}<select id={`${id}-example`} aria-label={t('전체 예제','All examples')} disabled={!saved.ready} value={example?.id??'custom'} onChange={e=>{const next=capacityExamples.find(p=>p.id===e.target.value);if(next)choose(next.input);}}><option value="custom" disabled>{t('직접 입력','Custom input')}</option>{capacityExamples.map(p=><option key={p.id} value={p.id}>{p.label[language]}</option>)}</select></label>
  {!saved.ready&&<p role="status">{t('저장된 입력을 확인하고 있습니다.','Checking the saved input.')}</p>}
  {saved.storage!=='available'&&<p role="status">{saved.storage==='corrupt'?t('저장된 입력 형식이 올바르지 않아 첫 예제로 복원했습니다.','The saved input format was invalid; the first example was restored.'):t('이 브라우저에서 저장할 수 없습니다. 현재 화면에서는 계속 실험할 수 있습니다.','Storage is unavailable. You can continue experimenting in this session.')}</p>}
  <form onSubmit={e=>{e.preventDefault();const next=readCapacityDraft(draft,probability);if(next){choose(next);setInvalid(false);}else setInvalid(true);}}>
   <h3>{t('채널과 선택 입력 직접 바꾸기','Edit the channel and selected input')}</h3><p>{t('p는0–1의 정수·분수, Bloch 좌표는−1부터1의 정수·분수로 입력하세요. 벡터 길이는1이하이어야 합니다. 같은 p에서 입력만 바꾸면 채널 용량은 유지되며 입력별 정보가 달라집니다.','Enter integer/fraction p in0–1 and Bloch coordinates in−1–1. Vector length must not exceed one. At fixed p, changing only the input changes selected information while leaving channel capacities fixed.')}</p>
   <label className="qubit-field" htmlFor={`${id}-probability`}>{t('위상 반전 확률 p','Phase-flip probability p')}<input id={`${id}-probability`} value={probability} maxLength={66} autoComplete="off" onChange={e=>{setProbability(e.target.value);setInvalid(false);}}/></label>
   <fieldset><legend>{t('선택 입력의 Bloch 벡터','Selected-input Bloch vector')}</legend>{draft.map((value,axis)=><label className="qubit-field" key={axis} htmlFor={`${id}-r-${axis}`}>{`r · ${['x','y','z'][axis]}`}<input id={`${id}-r-${axis}`} value={value} maxLength={66} spellCheck={false} autoComplete="off" onChange={e=>{setDraft(xs=>xs.map((x,i)=>i===axis?e.target.value:x));setInvalid(false);}}/></label>)}</fieldset>
   {invalid&&<p role="alert">{t('확률·좌표 형식과 범위를 확인하세요. 소수 대신 분수를 사용하세요. 현재 결과는 이전 적용 입력의 결과입니다.','Check probability/coordinate formats and ranges. Use fractions rather than decimals. The current result still belongs to the previously applied input.')}</p>}

   <div className="qubit-actions"><Button type="submit" disabled={!saved.ready}>{t('입력 적용','Apply input')}</Button><Button type="button" variant="outline" onClick={()=>choose(capacityExamples[0].input)}>{t('첫 예제로 초기화','Reset to first example')}</Button></div>
  </form>
  <div className="qubit-actions">{calculation.status==='pending'&&<Button variant="outline" onClick={()=>{setRunning(false);calculation.cancel();}}>{t('계산 취소','Cancel calculation')}</Button>}{(error||!running)&&<Button onClick={()=>{setRunning(true);calculation.retry();}}>{t('다시 계산','Recalculate')}</Button>}</div>
  <div role={error?'alert':'status'} aria-live="polite" className="qubit-status">{error?error==='invalid-input'?t('Bloch 벡터 길이≤1인지 확인하거나 예제로 돌아가세요.','Check that the Bloch vector length≤1, or return to an example.'):t('계산을 완료하지 못했습니다. 이전 결과를 새 입력의 결과로 표시하지 않습니다. 다시 계산하거나 다른 예제를 선택하세요.','The calculation could not complete. An earlier result is not shown for the new input. Recalculate or select another example.'):!running?t('계산을 취소했습니다.','Calculation cancelled.'):calculation.status==='ready'?t('현재 입력의 계산과 독립 검산을 마쳤습니다.','Calculation and independent checks for the current input are complete.'):calculation.phase==='validating'?t('결과를 독립적으로 검산하고 있습니다.','Independently checking the result.'):t('현재 입력을 계산하고 있습니다.','Calculating the current input.')}</div>
  {calculation.response?.kind==='capacity'&&<InformationCapacityResult key={currentKey} response={calculation.response} language={language}/>}
  {!embedded&&<>  <h3>{t('이해 확인','Check your understanding')}</h3>{capacityQuestions.map((q,i)=><fieldset className="information-question" key={i}><legend>{i+1}. {q.prompt[language]}</legend>{q.options.map((o,j)=><label key={j}><input type="radio" name={`${id}-quiz${i}`} checked={answers[i]===j} onChange={()=>setAnswers(v=>({...v,[i]:j}))}/>{o[language]}</label>)}{answers[i]!==undefined&&<p role="status"><strong>{answers[i]===q.correct?t('맞습니다. ','Correct. '):t('다시 생각해 보세요. ','Think again. ')}</strong>{q.explanation[language]}</p>}</fieldset>)}</>}
 </section>;
}
