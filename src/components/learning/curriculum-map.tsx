'use client';
import { useEffect, useId, useState } from 'react';
import Link from 'next/link';
import { useLanguage } from '@/components/language-provider';
import { curriculumTopicCount } from '@/data/learning-curriculum';
import { bookPath, type LearningUnitId } from '@/data/interactive-book';
import { learningPaths, domainById, unitById, learningLocation, domainUnits, pathUnits, searchLearningPath } from '@/data/learning-paths';
import { useSavedState } from './use-saved-state';
const branchIds = new Set([...learningPaths.map(p => p.id), ...domainById.keys()].map(id => `branch:${id}`));
const validOpen = (value: unknown): value is string[] => Array.isArray(value) && value.length <= branchIds.size && value.every(id => typeof id === 'string' && branchIds.has(id)) && new Set(value).size === value.length;
const initialOpen: string[] = [];
export default function CurriculumMap({ currentUnitId, compact = false }: { currentUnitId?: LearningUnitId; compact?: boolean }) {
  const { language } = useLanguage(), ko = language === 'ko', id = useId();
  const [query, setQuery] = useState('');
  const [open, setOpen, ready, storage] = useSavedState('suanlab-learning:curriculum-tree', initialOpen, validOpen);
  const [completed, setCompleted] = useState<Set<string>>(new Set());
  const [last, setLast] = useState<LearningUnitId | null>(null);
  const location = currentUnitId ? learningLocation(currentUnitId) : null;
  const result = searchLearningPath(query), searching = !!query.trim();
  useEffect(() => {
    const read = () => {
      const marked = new Set<string>();
      try {
        for (const unitId of unitById.keys()) {
          try { const saved = JSON.parse(localStorage.getItem(`suanlab-learning:${unitId}:study`) ?? 'null'); if (saved?.version === 1 && saved.value?.complete === true) marked.add(unitId); } catch { /* Ignore an individual invalid record. */ }
        }
        const previous = localStorage.getItem('suanlab-learning:last-unit');
        if (previous && unitById.has(previous as LearningUnitId)) setLast(previous as LearningUnitId);
      } catch { /* The tree remains navigable without storage. */ }
      setCompleted(marked);
    };
    read(); window.addEventListener('storage', read); window.addEventListener('suanlab-learning-progress', read);
    return () => { window.removeEventListener('storage', read); window.removeEventListener('suanlab-learning-progress', read); };
  }, []);
  useEffect(() => { if (currentUnitId) try { localStorage.setItem('suanlab-learning:last-unit', currentUnitId); } catch { /* Optional convenience only. */ } }, [currentUnitId]);
  const count = (ids: readonly string[]) => ids.filter(id => completed.has(id)).length;
  const toggle = (branch: string) => setOpen(old => old.includes(branch) ? old.filter(id => id !== branch) : [...old, branch]);
  useEffect(() => {
    if (!ready || !currentUnitId) return;
    const at = learningLocation(currentUnitId);
    if (at) setOpen(old => [...new Set([...old, `branch:${at.path.id}`, `branch:${at.domain.id}`])]);
  }, [ready, currentUnitId, setOpen]);
  const expanded = (key: string) => searching || open.includes(key);
  return <section className={`ai-curriculum-section ai-course-tree${compact ? ' ai-course-tree-compact' : ''}`} aria-labelledby={`${id}-title`} data-course-tree={compact ? 'reader' : 'overview'}>
    <div className="ai-section-heading"><div><span className="ai-eyebrow">EXPLORE YOUR LEARNING PATH</span><h2 id={`${id}-title`}>{ko ? '과정을 펼치며 시작하세요' : 'Find your path through the course'}</h2></div><p>{ko ? `${learningPaths.length}개 과정 · ${curriculumTopicCount}개 주제` : `${learningPaths.length} paths · ${curriculumTopicCount} topics`}</p></div>
    <p>{ko ? '관심 있는 과정을 펼친 뒤 영역과 단원을 선택하세요. 필요한 배경은 단원 안에서 다시 확인할 수 있습니다.' : 'Open a path, then choose a domain and lesson. Each lesson links to the background it needs.'}</p>
    {location && <p className="ai-tree-location">{ko ? '현재 위치' : 'You are here'}: {location.path.title[language]} / {location.domain.title[language]} / <strong>{unitById.get(currentUnitId!)!.title[language]}</strong></p>}
    <div className="ai-tree-toolbar"><div><label htmlFor={`${id}-search`}>{ko ? '배우고 싶은 개념 찾기' : 'Find a concept you want to learn'}</label><input id={`${id}-search`} type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder={ko ? '예: 투영, Bayes, 양자 간섭' : 'Try projection, Bayes or interference'} /></div><button type="button" disabled={!query && !open.length} onClick={() => { setQuery(''); setOpen([]); }}>{ko ? '탐색 초기화' : 'Reset exploration'}</button></div>
    {last && !currentUnitId && <p><Link className="ai-tree-resume" href={`${bookPath}${last}/`}>{ko ? '이어서 읽기' : 'Continue reading'}: {unitById.get(last)!.title[language]}</Link></p>}
    <p className="ai-tree-status" role="status">{searching ? (ko ? `검색된 단원 ${result.unitCount}개. 관련 가지를 펼쳤습니다.` : `${result.unitCount} matching lessons. Relevant branches are expanded.`) : (ko ? `학습 표시 ${completed.size} / ${unitById.size}단원 · 원하는 가지부터 탐색하세요.` : `${completed.size} / ${unitById.size} lessons marked complete. Start with any branch.`)}</p>
    <nav aria-label={ko ? '교육과정 트리' : 'Course tree'}><ul className="ai-tree-roots">{result.paths.map(({ path, domains }) => {
      const key = `branch:${path.id}`, isOpen = expanded(key), ids = pathUnits(path);
      return <li key={path.id} data-course-path={path.id}><button className="ai-tree-branch ai-tree-path" type="button" aria-expanded={isOpen} aria-controls={`${id}-${path.id}`} onClick={() => toggle(key)} disabled={!ready || searching}><span className="ai-tree-chevron" aria-hidden="true" /><span><strong>{path.title[language]}</strong><span className="ai-tree-purpose">{path.purpose[language]}</span></span><span className="ai-tree-count">{count(ids)} / {ids.length}</span></button>
        <ul id={`${id}-${path.id}`} hidden={!isOpen}>{domains.map(({ domain, unitIds }) => {
          const key = `branch:${domain.id}`, isOpen = expanded(key), ids = domainUnits(domain.id);
          return <li key={domain.id} data-course-domain={domain.id}><button className="ai-tree-branch" type="button" aria-expanded={isOpen} aria-controls={`${id}-${domain.id}`} onClick={() => toggle(key)} disabled={!ready || searching}><span className="ai-tree-chevron" aria-hidden="true" /><strong>{domain.title[language]}</strong><span className="ai-tree-count">{count(ids)} / {ids.length}</span></button>
            <div id={`${id}-${domain.id}`} hidden={!isOpen} className="ai-tree-domain-content">
              {domain.prerequisiteIds.length > 0 && <p className="ai-tree-prerequisites">{ko ? '관련 배경' : 'Helpful background'}: {domain.prerequisiteIds.map(id => domainById.get(id)!.title[language]).join(' · ')}</p>}
              <details className="ai-tree-topics"><summary>{ko ? `여기서 배우는 주제 ${domain.topics.length}개` : `${domain.topics.length} topics in this domain`}</summary><ul>{domain.topics.map((topic, i) => <li key={i}>{topic[language]}</li>)}</ul></details>
              <ul className="ai-tree-lessons">{unitIds.map(unitId => { const unit = unitById.get(unitId)!; return <li key={unitId}><Link href={`${bookPath}${unitId}/`} aria-current={unitId === currentUnitId ? 'page' : undefined} data-tree-unit={unitId}><span aria-hidden="true" className={`ai-tree-marker${completed.has(unitId) ? ' is-complete' : ''}`} /><span>{unit.title[language]}<small>{unit.subtitle[language]}</small></span>{completed.has(unitId) && <span className="ai-tree-completed-label">{ko ? '학습 표시됨' : 'Marked complete'}</span>}</Link></li>; })}</ul>
            </div>
          </li>;
        })}</ul>
      </li>;
    })}</ul></nav>
    {!result.paths.length && <p>{ko ? '일치하는 개념이 없습니다. 다른 표현으로 검색해보세요.' : 'No matching concept. Try another name.'}</p>}
    {!storage && <p className="ai-small">{ko ? '이 브라우저에서는 펼침 상태를 저장할 수 없습니다.' : 'This browser cannot save expansion preferences.'}</p>}
  </section>;
}
