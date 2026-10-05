'use client';
import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useLanguage } from '@/components/language-provider';
import { Button } from '@/components/ui/button';
import { searchItems, type SearchItem } from '@/lib/search';
const labels: Record<string, [string, string]> = { all: ['전체', 'All'], blog: ['블로그', 'Blog'], publication: ['연구 성과', 'Publications'], lecture: ['강의', 'Lectures'], research: ['연구 분야', 'Research'], project: ['프로젝트', 'Projects'], book: ['온라인 도서', 'Online books'], youtube: ['영상 강좌', 'Videos'], prompt: ['프롬프트', 'Prompts'] };
export default function SearchClient({ items }: { items: SearchItem[] }) {
  const { language } = useLanguage(); const lang = language === 'ko' ? 0 : 1;
  const [query, setQuery] = useState(''); const [type, setType] = useState('all'); const [limit, setLimit] = useState(40);
  useEffect(() => {
    const sync = () => { const params = new URLSearchParams(location.search); setQuery(params.get('q') || ''); setType(params.get('type') || 'all'); };
    sync(); window.addEventListener('popstate', sync); return () => window.removeEventListener('popstate', sync);
  }, []);
  const results = useMemo(() => searchItems(items, query, type), [items, query, type]);
  function update(value: string, category: string) { setQuery(value); setType(category); setLimit(40); const params = new URLSearchParams({ q: value, type: category }); history.replaceState(null, '', `?${params}`); }
  return <section className="container py-12"><div className="mx-auto max-w-4xl">
    <label className="block font-medium">{lang === 0 ? '통합 검색' : 'Search SuanLab'}<input type="search" className="mt-3 w-full rounded-xl border bg-background p-4 text-lg" value={query} onChange={(e) => update(e.target.value, type)} placeholder={lang === 0 ? '제목, 주제, 저자 (최소 2자)' : 'Title, topic, author (2+ characters)'} /></label>
    <div className="my-6 flex flex-wrap gap-2">{Object.entries(labels).map(([key, label]) => <Button key={key} size="sm" variant={key === type ? 'default' : 'outline'} aria-pressed={key === type} onClick={() => update(query, key)}>{label[lang]}</Button>)}</div>
    <p className="mb-6 text-sm text-muted-foreground" role="status">{query.trim().length < 2 ? (lang === 0 ? '검색어를 두 글자 이상 입력하세요.' : 'Enter at least two characters.') : `${results.length} ${lang === 0 ? '개 결과' : 'results'}`}</p>
    <div className="space-y-3">{results.slice(0, limit).map((item) => <Link key={`${item.type}:${item.href}:${item.title}`} href={item.href} className="block rounded-xl border bg-card p-5 transition-colors hover:border-primary"><span className="text-xs font-medium text-primary">{labels[item.type]?.[lang] || item.type}</span><h2 className="mt-2 font-semibold">{item.title}</h2><p className="mt-2 line-clamp-2 text-sm text-muted-foreground">{item.description}</p></Link>)}</div>
    {results.length > limit && <Button className="mt-6" variant="outline" onClick={() => setLimit(limit + 40)}>{lang === 0 ? '더 보기' : 'Show more'}</Button>}
  </div></section>;
}
