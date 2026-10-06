import { site } from '@/config/site';
import type { Metadata } from 'next';
import Script from 'next/script';
import PageHeader from '@/components/layout/PageHeader';
import PromptsClient from './PromptsClient';
import { promptBuilders, promptSnippets } from '@/data/prompts';

const BASE_URL = site.url;

export const metadata: Metadata = {
  title: 'Superintelligence Research Prompts',
  description:
    'LLM, RAG, 에이전트, 멀티모달, Text-to-SQL의 설계·평가·운영을 위한 프롬프트 빌더, 라이브러리, 연구 워크플로우',
  keywords: [
    'AI 프롬프트',
    'Prompt Engineering',
    'Claude CLI',
    '연구 프롬프트',
    '논문 작성',
    '실험 설계',
    'Rebuttal',
    'AI Research',
    'Prompt Library',
  ],
  openGraph: {
    title: 'Superintelligence Research Prompts | SuanLab',
    description: '초지능 연구와 실무 시스템을 위한 프롬프트 빌더·평가·워크플로우',
    url: `${BASE_URL}/prompts`,
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Superintelligence Research Prompts | SuanLab',
    description: '초지능 연구와 실무 시스템을 위한 프롬프트 툴킷',
  },
  alternates: {
    canonical: `${BASE_URL}/prompts`,
  },
};

export default function PromptsPage() {
  const itemList = [...promptBuilders, ...promptSnippets].map((item, i) => ({
    '@type': 'ListItem',
    position: i + 1,
    name: item.title.ko,
  }));

  return (
    <>
      <Script
        id="prompts-jsonld"
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'ItemList',
            name: 'Superintelligence Research Prompts',
            description: '초지능 연구와 실무 시스템을 위한 프롬프트 툴킷',
            itemListElement: itemList,
          }),
        }}
      />
      <PageHeader
        title="Superintelligence Research Prompts"
        subtitleKey="pageheader.prompts.subtitle"
        breadcrumbs={[{ label: 'Prompts' }]}
      />
      <section className="py-12 md:py-16">
        <div className="container">
          <PromptsClient />
        </div>
      </section>
    </>
  );
}
