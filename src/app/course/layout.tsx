import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Courses & Seminars',
  description:
    '이수안 교수의 교육 과정 및 세미나 - 데이터 사이언스, AI, 머신러닝, 딥러닝 강의 및 워크샵',
  keywords: [
    '강의',
    '세미나',
    'Courses',
    'Seminars',
    '교육',
    '워크샵',
    'AI 교육',
    '데이터 사이언스',
    '이수안',
  ],
  openGraph: {
    title: 'Courses & Seminars | SuanLab',
    description: '이수안 교수의 교육 과정 및 세미나',
    url: 'https://suanlab.com/course',
  },
};

export default function CourseLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
