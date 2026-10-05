import { researchAreas } from '@/data/research';
import { projects } from '@/data/projects';
import { playlists } from '@/data/youtube';
import { getAllBookPosts } from '@/lib/books';
import { promptBuilders, promptSnippets } from '@/data/prompts';
import type { SearchItem } from '@/lib/search';
import { site } from '@/config/site';
import { Metadata } from 'next';
import PageHeader from '@/components/layout/PageHeader';
import SearchClient from './SearchClient';
import { getAllPosts } from '@/lib/blog';
import { publications } from '@/data/publications';
import { lectures } from '@/data/lectures';

const BASE_URL = site.url;

export const metadata: Metadata = {
  title: '검색',
  description: 'SuanLab에서 연구 성과, 프로젝트, 강의, 도서와 블로그를 검색하세요.',
  openGraph: {
    title: '검색 | SuanLab',
    description: 'SuanLab에서 연구 성과, 프로젝트, 강의, 도서와 블로그를 검색하세요.',
    url: `${BASE_URL}/search`,
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: '검색 | SuanLab',
    description: 'SuanLab에서 연구 성과, 프로젝트, 강의, 도서와 블로그를 검색하세요.',
  },
  alternates: {
    canonical: `${BASE_URL}/search/`,
  },
  robots: {
    index: false,
    follow: true,
  },
};

export default function SearchPage() {
  // Load all searchable data at build time
  const items: SearchItem[] = [
    ...getAllPosts().map(p => ({ type: 'blog', title: p.title, description: `${p.excerpt} ${p.tags.join(' ')}`, href: `/blog/${p.slug}/` })),
    ...publications.map(p => ({ type: 'publication', title: p.title, description: `${p.authors} · ${p.venue} ${p.keywords || ''}`, href: `/publication/?q=${encodeURIComponent(p.title)}#publication-${p.id}` })),
    ...lectures.map(p => ({ type: 'lecture', title: `${p.titleKo} · ${p.titleEn}`, description: `${p.descriptionKo} ${p.topics.join(' ')}`, href: `/lecture/${p.slug}/` })),
    ...researchAreas.map(p => ({ type: 'research', title: `${p.titleKo} · ${p.titleEn}`, description: p.overview, href: `/research/${p.slug}/` })),
    ...projects.map(p => ({ type: 'project', title: p.title, description: `${p.organization} ${p.items.join(' ')}`, href: `/project/#project-${p.id}` })),
    ...playlists.map(p => ({ type: 'youtube', title: `${p.titleKo} · ${p.titleEn}`, description: 'YouTube 교육 영상 강좌', href: `/youtube/${p.slug}/` })),
    ...getAllBookPosts().map(p => ({ type: 'book', title: p.title, description: `${p.subtitle || ''} ${p.author}`, href: `/book/online/${p.slug}/` })),
    ...[...promptBuilders, ...promptSnippets].map(p => ({ type: 'prompt', title: `${p.title.ko} · ${p.title.en}`, description: `${p.description.ko} ${p.description.en} ${p.tags.join(' ')}`, href: '/prompts/' })),
  ];

  return (
    <>
      <PageHeader
        title="검색"
        subtitle="연구 성과, 프로젝트, 강의, 도서와 블로그를 검색하세요"
        breadcrumbs={[{ label: '검색' }]}
      />
      <SearchClient items={items} />
    </>
  );
}
