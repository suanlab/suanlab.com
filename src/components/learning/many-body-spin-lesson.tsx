'use client';
import { useLanguage } from '@/components/language-provider';
import type { LearningContent } from '@/lib/learning-content';
import { manyBodySpinUnit as unit,manyBodySpinGuide as guide } from '@/data/many-body-spin-teaching';
import ManyBodySpinLab from './many-body-spin-lab';
import LearningQuiz from './learning-quiz';
/** Staged complete lesson; route registration follows the full reader checks. */
export default function ManyBodySpinLesson({content}:{content:LearningContent}){
 const{language}=useLanguage();return <article className="ai-book" lang={language} data-spin-lesson>
  <header><span className="ai-eyebrow">{unit.eyebrow}</span><h1>{unit.title[language]}</h1><p>{unit.subtitle[language]}</p><p>{unit.prerequisites[language]}</p></header>
  <section aria-label={language==='ko'?'먼저 잡을 세 가지 직관':'Three starting intuitions'}><p>{guide.intuition[language]}</p>{guide.steps.map((s,i)=><article key={i} data-spin-concept={i}><h2>{s.title[language]}</h2><p>{s.explanation[language]}</p><p className="ai-formula" style={{overflowWrap:'anywhere'}}>{s.formula}</p><p>{s.diagram[language]}</p></article>)}</section>
  <div className="ai-prose" data-spin-chapter dangerouslySetInnerHTML={{__html:content[language]}}/>
  <ManyBodySpinLab/><LearningQuiz unit={unit}/>
 </article>;
}
