import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Projects',
  description:
    '이수안 교수의 연구 프로젝트 - 정부 R&D, 산학협력, AI/빅데이터 연구개발 과제',
  keywords: [
    '연구 프로젝트',
    'Projects',
    'R&D',
    '산학협력',
    'AI 연구',
    '빅데이터',
    '정부과제',
    '이수안',
  ],
  openGraph: {
    title: 'Projects | SuanLab',
    description: '이수안 교수의 연구 프로젝트',
    url: 'https://suanlab.com/project',
  },
};

export default function ProjectLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
