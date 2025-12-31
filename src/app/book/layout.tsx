import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Books',
  description:
    '이수안 교수의 저서 및 온라인 도서 - 파이썬, 데이터 사이언스, 머신러닝, 딥러닝 교재',
  keywords: [
    '저서',
    'Books',
    '교재',
    '파이썬',
    'Python',
    '데이터 사이언스',
    '머신러닝',
    '딥러닝',
    '이수안',
  ],
  openGraph: {
    title: 'Books | SuanLab',
    description: '이수안 교수의 저서 및 온라인 도서',
    url: 'https://suanlab.com/book',
  },
};

export default function BookLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
