import { Metadata } from 'next';
import PageHeader from '@/components/layout/PageHeader';
import { conferences, conferenceCategories } from '@/data/conferences';
import DeadlinesClient from './DeadlinesClient';

export const metadata: Metadata = {
  title: 'Conference Deadlines | SuanLab',
  description:
    'AI, 머신러닝, NLP, 컴퓨터 비전, 데이터 마이닝 등 주요 학술대회 논문 마감일 및 일정 - SuanLab',
  keywords: [
    'Conference Deadlines',
    '학회 마감일',
    'AAAI',
    'NeurIPS',
    'ICML',
    'ICLR',
    'CVPR',
    'ACL',
    'EMNLP',
    'KDD',
    'SIGIR',
    'ICRA',
    'AI 학회',
    '논문 마감',
    'Call for Papers',
  ],
  openGraph: {
    title: 'Conference Deadlines | SuanLab',
    description:
      'AI, 머신러닝, NLP, 컴퓨터 비전 등 주요 학술대회 논문 마감일 및 일정',
    url: 'https://suanlab.com/deadlines',
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Conference Deadlines | SuanLab',
    description:
      'AI, 머신러닝, NLP, 컴퓨터 비전 등 주요 학술대회 논문 마감일 및 일정',
  },
  alternates: {
    canonical: 'https://suanlab.com/deadlines',
  },
};

export default function DeadlinesPage() {
  return (
    <>
      <PageHeader
        title="Conference Deadlines"
        subtitleKey="pageheader.deadlines.subtitle"
        breadcrumbs={[{ label: 'Deadlines' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <DeadlinesClient conferences={conferences} categories={conferenceCategories} />
        </div>
      </section>
    </>
  );
}
