import { bookPath, interactiveBook, learningUnits } from '@/data/interactive-book';
import { englishText } from '@/data/i18n/translate';
import { projectEnglish } from '@/data/i18n/projects';
import { researchEnglish } from '@/data/i18n/research';
import { catalogEnglish } from '@/data/i18n/catalog';
import { researchAreas } from '@/data/research';
import { projects } from '@/data/projects';
import { playlists } from '@/data/youtube';
import { getAllBookPosts } from '@/lib/books';
import { promptBuilders, promptSnippets } from '@/data/prompts';
import type { SearchItem } from '@/lib/search';
import { promptHref } from '@/lib/prompt-state';
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
    { type: 'book', title: interactiveBook.title.ko, titleEn: interactiveBook.title.en, description: interactiveBook.description.ko, descriptionEn: interactiveBook.description.en, href: bookPath },
    ...learningUnits.map(unit => ({ type: 'book', title: unit.title.ko, titleEn: unit.title.en, description: unit.subtitle.ko, descriptionEn: unit.subtitle.en, href: `${bookPath}${unit.id}/` })),
    ...getAllPosts().map(p => ({ type: 'blog', title: p.title, description: `${p.excerpt} ${p.tags.join(' ')}`, href: `/blog/${p.slug}/` })),
    ...publications.map(p => ({ type: 'publication', title: p.title, description: `${p.authors} · ${p.venue} ${p.keywords || ''}`, href: `/publication/?q=${encodeURIComponent(p.title)}#publication-${p.id}` })),
    ...lectures.map(p => ({ type: 'lecture', titleEn: p.titleEn, descriptionEn: p.description, title: `${p.titleKo} · ${p.titleEn}`, description: `${p.descriptionKo} ${p.topics.join(' ')}`, href: `/lecture/${p.slug}/` })),
    ...researchAreas.map(p => ({ type: 'research', titleEn: p.titleEn, descriptionEn: p.overview.split('\n\n').map(part => englishText(part, researchEnglish)).join(' '), title: `${p.titleKo} · ${p.titleEn}`, description: p.overview, href: `/research/${p.slug}/` })),
    ...projects.map(p => ({ type: 'project', titleEn: englishText(p.title, projectEnglish), descriptionEn: p.items.map(item => englishText(item, projectEnglish)).join(' '), title: p.title, description: `${p.organization} ${p.items.join(' ')}`, href: `/project/#project-${p.id}` })),
    ...playlists.map(p => ({ type: 'youtube', titleEn: p.titleEn, descriptionEn: 'YouTube educational video courses', title: `${p.titleKo} · ${p.titleEn}`, description: 'YouTube 교육 영상 강좌', href: `/youtube/${p.slug}/` })),
    ...getAllBookPosts().map(p => ({ type: 'book', titleEn: englishText(p.title, catalogEnglish), descriptionEn: p.subtitle ? englishText(p.subtitle, catalogEnglish) : '', title: p.title, description: `${p.subtitle || ''} ${p.author}`, href: `/book/online/${p.slug}/` })),
    ...promptBuilders.map(p => ({ type: 'prompt', titleEn: p.title.en, descriptionEn: p.description.en, title: `${p.title.ko} · ${p.title.en}`, description: `${p.description.ko} ${p.description.en} ${p.tags.join(' ')}`, href: promptHref('builder', p.id) })),
    ...promptSnippets.map(p => ({ type: 'prompt', titleEn: p.title.en, descriptionEn: p.description.en, title: `${p.title.ko} · ${p.title.en}`, description: `${p.description.ko} ${p.description.en} ${p.tags.join(' ')}`, href: promptHref('snippet', p.id) })),
  ];

  return (
    <>
      <PageHeader
        title="검색"
        titleKey="nav.search"
        subtitleKey="search.subtitle"
        breadcrumbs={[{ label: '검색', labelKey: 'nav.search' }]}
      />
      <SearchClient items={items} />
    </>
  );
}
