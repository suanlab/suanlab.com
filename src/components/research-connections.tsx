import Link from 'next/link';
import { lectures } from '@/data/lectures';
import { projects } from '@/data/projects';
import { researchKeywordMap } from '@/data/research';
import { researchLinks } from '@/data/editorial';
import { publications } from '@/data/publications';
import { getAllPosts } from '@/lib/blog';

export default function ResearchConnections({ slug }: { slug: string }) {
  const keywords = researchKeywordMap[slug] || [];
  const matches = (text: string) => keywords.some(keyword => text.toLowerCase().includes(keyword.toLowerCase()));
  const links = researchLinks[slug];
  const groups = [
    { title: '연구 논문 / Publications', items: (links?.publications || []).map(id => publications.find(p => p.id === id)!).map(p => ({ title: p.title, href: `/publication/?q=${encodeURIComponent(p.title)}#publication-${p.id}` })) },
    { title: '관련 강의 / Courses', items: (links?.lectures || []).map(slug => lectures.find(p => p.slug === slug)!).map(p => ({ title: p.titleKo, href: `/lecture/${p.slug}/` })) },
    { title: '관련 프로젝트 / Projects', items: (links?.projects || []).map(id => projects.find(p => p.id === id)!).map(p => ({ title: p.title, href: `/project/#project-${p.id}` })) },
    { title: '키워드 기반 읽을거리 / Suggested reading', items: getAllPosts().filter(p => matches(`${p.title} ${p.tags.join(' ')}`)).slice(0, 3).map(p => ({ title: p.title, href: `/blog/${p.slug}/` })) },
  ];
  return <section className="border-t bg-muted/20 py-12"><div className="container"><h2 className="text-2xl font-semibold">연구에서 학습으로 / Explore further</h2><p className="mt-2 text-sm text-muted-foreground">논문·강의·프로젝트는 연구 주제로 연결했습니다. 읽을거리는 키워드 추천이며 연구 성과 목록과 구분됩니다.<br />Courses, papers, and projects share research topics. Suggested reading is matched by keywords.</p><div className="mt-6 grid gap-6 md:grid-cols-2 xl:grid-cols-4">{groups.map(group => <div key={group.title} className="min-w-0 rounded-xl border bg-card p-5"><h3 className="mb-4 font-medium">{group.title}</h3><ul className="space-y-4">{group.items.map(item => <li key={item.href}><Link className="break-words text-sm hover:text-primary hover:underline" href={item.href}>{item.title}</Link></li>)}</ul>{!group.items.length && <p className="text-sm text-muted-foreground">연결된 콘텐츠가 없습니다.</p>}</div>)}</div></div></section>;
}
