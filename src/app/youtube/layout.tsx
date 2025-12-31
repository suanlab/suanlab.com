import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'YouTube',
  description:
    '이수안 교수의 데이터 사이언스, 머신러닝, 딥러닝 유튜브 강의 - 파이썬, PyTorch, TensorFlow 튜토리얼',
  keywords: [
    'YouTube',
    '유튜브',
    '강의',
    '튜토리얼',
    '파이썬',
    'Python',
    'PyTorch',
    'TensorFlow',
    'Deep Learning',
    'Machine Learning',
    '이수안',
  ],
  openGraph: {
    title: 'YouTube | SuanLab',
    description: '이수안 교수의 데이터 사이언스, 머신러닝, 딥러닝 유튜브 강의',
    url: 'https://suanlab.com/youtube',
  },
};

export default function YouTubeLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
