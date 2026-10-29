'use client';
import { useId, useState } from 'react';
import Link from 'next/link';
import { useLanguage } from '@/components/language-provider';
import { curriculumDomains, curriculumTopicCount } from '@/data/learning-curriculum';
import { bookPath, learningUnits } from '@/data/interactive-book';

export default function CurriculumMap() {
 const {language}=useLanguage(),ko=language==='ko',id=useId();const [query,setQuery]=useState('');
 const needle=query.trim().toLocaleLowerCase();const visible=curriculumDomains.filter(domain=>[domain.title.ko,domain.title.en,...domain.topics.flatMap(t=>[t.ko,t.en])].some(t=>t.toLocaleLowerCase().includes(needle)));
 return <section className="ai-curriculum-section" aria-labelledby={`${id}-title`}>
 <div className="ai-section-heading"><div><span className="ai-eyebrow">THE FULL CURRICULUM</span><h2 id={`${id}-title`}>{ko?'전체 교육과정':'Explore the full curriculum'}</h2></div><p>{ko?`${curriculumDomains.length}개 영역 · ${curriculumTopicCount}개 세부 주제`:`${curriculumDomains.length} domains · ${curriculumTopicCount} topics`}</p></div>
 <details className="ai-curriculum"><summary>{ko?'전체 범위와 읽을 수 있는 단원 보기':'View the full scope and available readings'}</summary><div className="ai-curriculum-body">
 <p>{ko?'이 목록은 주차·분량 제한 없는 집필 범위입니다. 연결된 입문 단원이 있어도 해당 영역의 모든 주제와 심화 내용이 완성되었다는 뜻은 아닙니다.':'This is an authoring scope with no fixed duration or size. Related introductory readings do not mean every topic or advanced treatment in that domain is complete.'}</p>
 <label htmlFor={`${id}-search`}>{ko?'영역 또는 주제 찾기':'Find a domain or topic'}</label><input id={`${id}-search`} type="search" value={query} onChange={e=>setQuery(e.target.value)} placeholder={ko?'예: 양자, 측도, 최적화':'For example: quantum, measure, optimization'}/>
 <p className="ai-small" role="status">{ko?`${curriculumDomains.length}개 영역 중 ${visible.length}개 표시`:`Showing ${visible.length} of ${curriculumDomains.length} domains`}</p>
 <div className="ai-curriculum-grid">{visible.map(domain=><details className="ai-curriculum-domain" key={domain.id} id={`curriculum-${domain.id}`}>
 <summary><strong>{domain.title[language]}</strong><span>{ko?`${domain.topics.length}개 주제 · ${domain.unitIds.length?`관련 입문 단원 ${domain.unitIds.length}개`:'집필 예정'}`:`${domain.topics.length} topics · ${domain.unitIds.length?`${domain.unitIds.length} related readings`:'Planned'}`}</span></summary>
 <div className="ai-curriculum-domain-body">{domain.prerequisiteIds.length>0&&<p className="ai-small">{ko?'관련 선수 영역: ':'Relevant prerequisites: '}{domain.prerequisiteIds.map(id=>curriculumDomains.find(d=>d.id===id)!.title[language]).join(' · ')}</p>}
 <ol>{domain.topics.map((topic,i)=><li key={i}>{topic[language]}</li>)}</ol>
 {domain.unitIds.length>0?<><h3>{ko?'연결된 입문 단원':'Related introductory readings'}</h3><ul>{domain.unitIds.map(unitId=><li key={unitId}><Link href={`${bookPath}${unitId}/`}>{learningUnits.find(u=>u.id===unitId)!.title[language]}</Link></li>)}</ul></>:<p className="ai-small">{ko?'상세 원고와 실습을 집필할 예정입니다.':'Detailed chapters and experiments are planned.'}</p>}
 </div></details>)}</div>{visible.length===0&&<p>{ko?'일치하는 영역이나 주제가 없습니다.':'No matching domain or topic.'}</p>}
 </div></details></section>;
}
