import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Publications',
  description:
    '이수안 교수의 학술 논문 및 연구 성과 - SCI/SCIE 국제 저널, 국내 학술지, 학술대회 논문 목록',
  keywords: [
    '논문',
    'Publications',
    'SCI',
    'SCIE',
    '학술논문',
    '연구성과',
    '이수안',
    'Deep Learning',
    'Machine Learning',
  ],
  openGraph: {
    title: 'Publications | SuanLab',
    description: '이수안 교수의 학술 논문 및 연구 성과',
    url: 'https://suanlab.com/publication',
  },
};

export default function PublicationLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
