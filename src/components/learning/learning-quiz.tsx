'use client';
import { Button } from '@/components/ui/button';
import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import type { LearningUnit } from '@/data/interactive-book';
import { useSavedState } from './use-saved-state';
interface StudyState { complete: boolean; answers: Record<string, number>; }
const initial: StudyState = { complete: false, answers: {} };
function validStudy(value: unknown): value is StudyState {
  if (!value || typeof value !== 'object') return false;
  const s = value as StudyState;
  return typeof s.complete === 'boolean' && !!s.answers && typeof s.answers === 'object' && !Array.isArray(s.answers)
    && Object.values(s.answers).every(n => Number.isInteger(n) && n >= 0 && n < 3);
}
export default function LearningQuiz({ unit }: { unit: LearningUnit }) {
  const { language } = useLanguage(); const ko = language === 'ko';
  const [study, set, ready, storage] = useSavedState(`suanlab-learning:${unit.id}:study`, initial, validStudy);
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  return <section className="ai-quiz" id="check" aria-labelledby="ai-quiz-title"><span className="ai-eyebrow">CHECK YOUR UNDERSTANDING</span><h2 id="ai-quiz-title">{ko ? '설명할 수 있나요?' : 'Can you explain it?'}</h2>
    <p>{ko ? '실험 설정과 무관한 기준 예제를 묻습니다. 답을 고르고 설명을 확인해보세요.' : 'These questions use fixed reference examples, independent of your experiment settings. Choose an answer and check the explanation.'}</p>
    {unit.questions.map((q, index) => <fieldset key={q.id} disabled={!ready} className="ai-question"><legend>{index + 1}. {q.question[language]}</legend>
      {q.options.map((option, i) => <label key={i}><input type="radio" name={q.id} value={i} checked={study.answers[q.id] === i} onChange={() => { set(s => ({ ...s, answers: { ...s.answers, [q.id]: i } })); setChecked(c => ({ ...c, [q.id]: false })); }} />{option[language]}</label>)}
      <Button variant="outline" size="sm" type="button" disabled={study.answers[q.id] === undefined} onClick={() => setChecked(c => ({ ...c, [q.id]: true }))}>{ko ? '답과 해설 확인' : 'Check answer and explanation'}</Button>
      {checked[q.id] && <div className={`ai-answer ${study.answers[q.id] === q.answer ? 'ai-answer-correct' : ''}`} role="status"><strong>{study.answers[q.id] === q.answer ? (ko ? '맞았습니다.' : 'Correct.') : (ko ? `다시 생각해보세요. 정답: ${q.options[q.answer][language]}` : `Try again. Answer: ${q.options[q.answer][language]}`)}</strong><p>{q.explanation[language]}</p></div>}
    </fieldset>)}
    <label className="ai-completion"><input type="checkbox" disabled={!ready} checked={study.complete} onChange={e => set(s => ({ ...s, complete: e.target.checked }))} /><span>{ko ? '이 단원을 이해했고, 내 말로 설명할 수 있습니다.' : 'I understand this unit and can explain it in my own words.'}</span></label>
    <p className="ai-small">{storage ? (ko ? '선택한 답과 학습 표시는 이 브라우저에 저장됩니다. 다시 풀거나 표시를 해제할 수 있습니다.' : 'Answers and your learning marker are saved in this browser. You can revise or uncheck them.') : (ko ? '저장소를 사용할 수 없어 학습 표시는 현재 방문에만 유지됩니다.' : 'Storage is unavailable. Your learning marker lasts for this visit only.')}</p>
  </section>;
}
