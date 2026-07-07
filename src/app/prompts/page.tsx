import type { Metadata } from 'next';
import Script from 'next/script';
import PageHeader from '@/components/layout/PageHeader';
import PromptsClient from './PromptsClient';
import { promptBuilders, promptSnippets } from '@/data/prompts';

const BASE_URL = 'https://suanlab.com';

export const metadata: Metadata = {
  title: 'AI Research Prompts',
  description:
    'AI/ML 연구를 위한 프롬프트 툴킷 — 돌파구 전략, 품질 감사, 논문 리뷰, 실험 설계, 논문 작성, 학회 전략, Rebuttal 등 13개 빌더와 즉시 복사 가능한 큐레이션 프롬프트 라이브러리',
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
    title: 'AI Research Prompts | SuanLab',
    description: 'AI/ML 연구를 위한 프롬프트 툴킷 — 13개 빌더 + 큐레이션 라이브러리',
    url: `${BASE_URL}/prompts`,
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'AI Research Prompts | SuanLab',
    description: 'AI/ML 연구를 위한 프롬프트 툴킷',
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
            name: 'AI Research Prompts',
            description: 'AI/ML 연구를 위한 프롬프트 툴킷',
            itemListElement: itemList,
          }),
        }}
      />
      <PageHeader
        title="AI Research Prompts"
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
