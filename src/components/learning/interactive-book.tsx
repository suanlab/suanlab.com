'use client';
import { isProbabilityKind } from '@/lib/learning-probability';

import { isAbstractAlgebraKind } from '@/lib/learning-abstract-algebra';
import { isTensorKind } from '@/lib/learning-tensors';
import { isComplexKind } from '@/lib/learning-complex';
import { isAnalysisKind } from '@/lib/learning-analysis';
import { isMatrixStructureKind } from '@/lib/learning-matrix-structures';
import { isDiscreteKind } from '@/lib/learning-discrete';
import Link from 'next/link';
import PageHeader from '@/components/layout/PageHeader';
import { Button } from '@/components/ui/button';
import { ArrowDown, ArrowLeft, ArrowRight, BookOpen, MoveUpRight } from 'lucide-react';
import { useLanguage } from '@/components/language-provider';
import { bookPath, interactiveBook, learningUnits, type LearningUnit } from '@/data/interactive-book';
import BookArt from './book-art';
import CurriculumMap from './curriculum-map';
import InteractiveLab from './interactive-lab';
import AlgebraLab from './algebra-lab';
import LogicLab from './logic-lab';
import { isLogicKind } from '@/lib/learning-logic';
import FoundationsLab from './foundations-lab';
import { isElementaryKind } from '@/lib/learning-elementary';
import { isFoundationKind } from '@/lib/learning-foundations';
import CalculusLab from './calculus-lab';
import { isCalculusKind } from './calculus-model';
import { isAlgebraKind } from './algebra-model';
import LearningQuiz from './learning-quiz';
import ConceptWalkthrough from './concept-walkthrough';
import type { LearningContent } from '@/lib/learning-content';

export function InteractiveBookEntry() {
  const { language } = useLanguage(); const ko = language === 'ko';
  return <Link href={bookPath} className="ai-book-entry"><div><span className="ai-eyebrow">NEW · INTERACTIVE BOOK</span><h2>{interactiveBook.title[language]}</h2><p>{ko ? `${learningUnits.length}개 단원 · 읽고, 조작하고, 설명해보세요.` : `${learningUnits.length} units to explore · Read, experiment and explain.`}</p></div><span className="ai-entry-action">{ko ? '교재 열기' : 'Open book'} <ArrowRight size={18} aria-hidden="true" /></span></Link>;
}
export function InteractiveBookOverview() {
  const { language } = useLanguage(); const ko = language === 'ko';
  return <div className="ai-book ai-book-overview" lang={language}>
    <PageHeader title={interactiveBook.title[language]} subtitle={interactiveBook.description[language]} breadcrumbs={[{ label: 'Book', href: '/book/' }, { label: 'Online Book', href: '/book/online/' }, { label: interactiveBook.title[language] }]} />
    <div className="container ai-book-wrap">
      <header className="ai-book-hero"><div><span className="ai-eyebrow">SUANLAB OPEN BOOK · PREVIEW EDITION</span><h2>{ko ? '수식에서 직관으로' : 'From equations to intuition'}</h2><p className="ai-hero-description">{interactiveBook.description[language]}</p><p className="ai-hero-note">{ko ? 'AI의 개념과 동작을 이해하고 새로운 연구를 시작하기 위한, 끝을 정하지 않은 지식 지도. 논리·증명부터 벡터·회귀·PCA와 학습 원리까지 차근차근 확인해보세요.' : 'An open-ended knowledge map for understanding how AI works and beginning new research. Explore logic, proof, vectors, regression, PCA and the principles of learning step by step.'}</p><Button asChild size="lg"><Link href={`${bookPath}${learningUnits[0].id}/`}>{ko ? '첫 단원 읽기' : 'Read the first unit'}<ArrowRight size={17} aria-hidden="true" /></Link></Button></div><div className="ai-hero-art"><BookArt kind={learningUnits[0].id}/><span>01 / {learningUnits[0].eyebrow}</span></div></header>
      <section className="ai-book-intro" aria-labelledby="ai-how"><div><span className="ai-eyebrow">A DIFFERENT WAY TO LEARN</span><h2 id="ai-how">{ko ? '한 개념을, 세 가지 방법으로.' : 'One idea. Three ways to understand.'}</h2></div><ol>{(ko ? [['읽기', '짧은 설명과 수식으로 원리를 연결합니다.'], ['실험', '값을 바꾸며 그림과 수치의 변화를 비교합니다.'], ['설명', '질문에 답하고 자신의 말로 정리합니다.']] : [['Read', 'Connect the principle with concise explanations and equations.'], ['Experiment', 'Change a value and compare the geometry with the numbers.'], ['Explain', 'Answer questions and put the idea in your own words.']]).map(([title, body], i) => <li key={title}><span>{String(i + 1).padStart(2, '0')}</span><h3>{title}</h3><p>{body}</p></li>)}</ol></section>
      <CurriculumMap/>
      <section className="ai-unit-section" aria-labelledby="ai-units"><div className="ai-section-heading"><div><span className="ai-eyebrow">START EXPLORING</span><h2 id="ai-units">{ko ? '먼저 만나보는 단원' : 'Your first explorations'}</h2></div><p>{ko ? '정해진 주차 없이, 이해한 만큼 다음으로.' : 'No weekly schedule. Continue when the idea clicks.'}</p></div><div className="ai-unit-cards">{learningUnits.map((unit, index) => <Link key={unit.id} className={`ai-unit-card ai-accent-${unit.accent}`} href={`${bookPath}${unit.id}/`}><div className="ai-unit-card-art"><BookArt kind={unit.id} /><span className="ai-card-number">{String(index + 1).padStart(2, '0')}</span></div><div className="ai-unit-card-body"><span className="ai-eyebrow">{unit.eyebrow}</span><h3>{unit.title[language]}</h3><p>{unit.subtitle[language]}</p><div className="ai-unit-formula">{unit.formula}<MoveUpRight size={20} aria-hidden="true" /></div></div></Link>)}</div></section>
      <aside className="ai-roadmap-note"><BookOpen size={26} aria-hidden="true" /><div><h2>{ko ? '지식 지도는 계속 확장됩니다.' : 'The knowledge map keeps growing.'}</h2><p>{ko ? '전체 교재는 주차나 분량을 제한하지 않습니다. 선형대수, 해석학, 확률·통계, 최적화, 역학, 정보이론에서 AI 연구에 필요한 고급 주제로 확장합니다. 공개된 단원만 바로 열 수 있으며, 이후 내용은 순차적으로 집필합니다.' : 'The full textbook has no fixed duration or size. It will expand through linear algebra, analysis, probability, statistics, optimization, mechanics and information theory into advanced topics for AI research. Only completed preview units are available; further material will be developed progressively.'}</p></div></aside>
    </div>
  </div>;
}
export function InteractiveBookReader({ unit, html }: { unit: LearningUnit; html: LearningContent }) {
  const { language } = useLanguage(); const ko = language === 'ko'; const index = learningUnits.findIndex(item => item.id === unit.id);
  return <div className={`ai-book ai-reader ai-accent-${unit.accent}`} lang={language}>
    <PageHeader title={unit.title[language]} subtitle={unit.subtitle[language]} breadcrumbs={[{ label: 'Online Book', href: '/book/online/' }, { label: interactiveBook.title[language], href: bookPath }, { label: unit.title[language] }]} />
    <div className="container ai-book-wrap"><nav className="ai-reader-top" aria-label={ko ? '교재 메뉴' : 'Book navigation'}><Link href={bookPath}><ArrowLeft size={17} aria-hidden="true" /><span>{interactiveBook.title[language]}</span></Link><span className="ai-preview-label">PREVIEW EDITION</span></nav>
      <header className="ai-reader-header"><span className="ai-eyebrow">{String(index + 1).padStart(2, '0')} / {unit.eyebrow}</span><div className="ai-prerequisite">{ko ? '먼저 알아두면 좋아요' : 'Helpful before you start'} · {unit.prerequisites[language]}</div>{unit.prerequisiteIds.length > 0 && <div className="ai-prerequisite-links">{unit.prerequisiteIds.map(id => { const prior = learningUnits.find(item => item.id === id)!; return <Link key={id} href={`${bookPath}${id}/`}>{prior.title[language]} →</Link>; })}</div>}</header>
      <details className="ai-unit-selector"><summary>{ko ? `단원 목록 (${learningUnits.length})` : `Unit list (${learningUnits.length})`} · {unit.title[language]}</summary><nav className="ai-unit-nav" aria-label={ko ? '단원 선택' : 'Choose a unit'}>{learningUnits.map((item, i) => <Link href={`${bookPath}${item.id}/`} key={item.id} aria-current={unit.id === item.id ? 'page' : undefined}><span>{String(i + 1).padStart(2, '0')}</span>{item.title[language]}</Link>)}</nav></details>
      <div className="ai-reading-tools"><span>{ko ? '이해하는 속도로 읽으세요.' : 'Read at the pace of understanding.'}</span><a href="#visual-intuition">{ko ? '개념도' : 'Concept diagram'}</a><a href="#theory">{ko ? '이론' : 'Theory'}</a><a href="#experiment">{ko ? '실험' : 'Experiment'}<ArrowDown size={13} aria-hidden="true" /></a><a href="#check">{ko ? '확인 문제' : 'Check understanding'}</a></div>
      <ConceptWalkthrough key={unit.id} kind={unit.id} />
      <div className="ai-reader-columns"><article className="ai-theory" id="theory" aria-label={ko ? '단원 이론' : 'Unit theory'}><details className="ai-unit-outline"><summary>{ko ? '이 단원 목차' : 'In this unit'}</summary><nav aria-label={ko ? '본문 목차' : 'Theory contents'}><ol>{html.outline[language].map(heading => <li key={heading.id}><a href={`#${heading.id}`}>{heading.title}</a></li>)}</ol></nav></details><div className="ai-prose" dangerouslySetInnerHTML={{ __html: html[language] }} /><LearningQuiz key={unit.id} unit={unit} /></article><aside className="ai-experiment-column" aria-label={ko ? '인터랙티브 실험' : 'Interactive experiment'}>{isLogicKind(unit.id) ? <LogicLab key={unit.id} kind={unit.id}/> : (isElementaryKind(unit.id)||isDiscreteKind(unit.id)||isMatrixStructureKind(unit.id)||isAnalysisKind(unit.id)||isComplexKind(unit.id)||isProbabilityKind(unit.id)||isAbstractAlgebraKind(unit.id)||isTensorKind(unit.id)) ? <FoundationsLab key={unit.id} kind={unit.id}/> : isFoundationKind(unit.id) ? <FoundationsLab key={unit.id} kind={unit.id} /> : isCalculusKind(unit.id) ? <CalculusLab key={unit.id} kind={unit.id} /> : isAlgebraKind(unit.id) ? <AlgebraLab key={unit.id} kind={unit.id} /> : <InteractiveLab key={unit.id} kind={unit.id} />}</aside></div>
      <nav className="ai-reader-bottom" aria-label={ko ? '이전 다음 단원' : 'Previous and next units'}><Link href={index > 0 ? `${bookPath}${learningUnits[index - 1].id}/` : bookPath}><ArrowLeft size={17} aria-hidden="true" /><span>{index > 0 ? learningUnits[index - 1].title[language] : (ko ? '교재 목차' : 'Book contents')}</span></Link>{index < learningUnits.length - 1 ? <Link href={`${bookPath}${learningUnits[index + 1].id}/`}><span>{learningUnits[index + 1].title[language]}</span><ArrowRight size={17} aria-hidden="true" /></Link> : <Link href={bookPath}><span>{ko ? '다른 단원 둘러보기' : 'Explore the other units'}</span><ArrowRight size={17} aria-hidden="true" /></Link>}</nav>
    </div>
  </div>;
}
