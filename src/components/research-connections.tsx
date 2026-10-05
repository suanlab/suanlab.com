import Link from 'next/link';
import { lectures } from '@/data/lectures';
import { projects } from '@/data/projects';
import { researchKeywordMap } from '@/data/research';
import { getAllPosts } from '@/lib/blog';

export default function ResearchConnections({ slug }: { slug: string }) {
  const keywords = researchKeywordMap[slug] || [];
  const matches = (text: string) => keywords.some(keyword => text.toLowerCase().includes(keyword.toLowerCase()));
  const groups = [
    { title: '관련 강의 / Courses', items: lectures.filter(p => matches(`${p.titleEn} ${p.description} ${p.topics.join(' ')}`)).slice(0, 3).map(p => ({ title: p.titleKo, href: `/lecture/${p.slug}/` })) },
    { title: '관련 프로젝트 / Projects', items: projects.filter(p => matches(`${p.title} ${p.items.join(' ')}`)).slice(0, 3).map(p => ({ title: p.title, href: `/project/#project-${p.id}` })) },
    { title: '관련 글 / Reading', items: getAllPosts().filter(p => matches(`${p.title} ${p.tags.join(' ')}`)).slice(0, 3).map(p => ({ title: p.title, href: `/blog/${p.slug}/` })) },
  ];
  return <section className="border-t bg-muted/20 py-12"><div className="container"><h2 className="text-2xl font-semibold">연구에서 학습으로 / Explore further</h2><p className="mt-2 text-sm text-muted-foreground">연구 분야의 키워드를 기준으로 연결한 콘텐츠입니다.</p><div className="mt-6 grid gap-6 md:grid-cols-3">{groups.map(group => <div key={group.title} className="rounded-xl border bg-card p-5"><h3 className="mb-4 font-medium">{group.title}</h3><ul className="space-y-4">{group.items.map(item => <li key={item.href}><Link className="text-sm hover:text-primary hover:underline" href={item.href}>{item.title}</Link></li>)}</ul>{!group.items.length && <p className="text-sm text-muted-foreground">연결된 콘텐츠가 없습니다.</p>}</div>)}</div></div></section>;
}
