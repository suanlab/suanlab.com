'use client';
import { useLanguage } from '@/components/language-provider';
import type { LearningContent } from '@/lib/learning-content';
import { manyBodyMpsUnit as unit,manyBodyMpsGuide as guide } from '@/data/many-body-mps-teaching';
import ManyBodyMpsLab from './many-body-mps-lab';
import LearningQuiz from './learning-quiz';
/** Staged complete lesson; route registration follows the full reader checks. */
export default function ManyBodyMpsLesson({content}:{content:LearningContent}){
 const{language}=useLanguage();return <article style={{minWidth:0,overflowWrap:'anywhere'}} className="ai-book" lang={language} data-mps-lesson>
  <header><span className="ai-eyebrow">{unit.eyebrow}</span><h1>{unit.title[language]}</h1><p>{unit.subtitle[language]}</p><p>{unit.prerequisites[language]}</p></header>
  <section aria-label={language==='ko'?'먼저 잡을 세 가지 직관':'Three starting intuitions'}><p>{guide.intuition[language]}</p>{guide.steps.map((s,i)=><article key={i} data-mps-concept={i}><h2>{s.title[language]}</h2><p>{s.explanation[language]}</p><p className="ai-formula" style={{overflowWrap:'anywhere'}}>{s.formula}</p><p>{s.diagram[language]}</p></article>)}</section>
  <div className="ai-prose" data-mps-chapter dangerouslySetInnerHTML={{__html:content[language]}}/>
  <ManyBodyMpsLab/><LearningQuiz unit={unit}/>
 </article>;
}
