import { Metadata } from 'next';
import PageHeader from '@/components/layout/PageHeader';
import { EventCollectionJsonLd } from '@/components/seo/JsonLd';
import { courses, courseCategories, Course } from '@/data/courses';
import CourseClient from './CourseClient';

export const metadata: Metadata = {
  title: 'Course & Seminar | SuanLab',
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
    '특강',
    '캠프',
    '연수',
  ],
  openGraph: {
    title: 'Course & Seminar | SuanLab',
    description:
      '이수안 교수의 교육 과정 및 세미나 - 데이터 사이언스, AI, 머신러닝, 딥러닝 강의 및 워크샵',
    url: 'https://suanlab.com/course',
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Course & Seminar | SuanLab',
    description:
      '이수안 교수의 교육 과정 및 세미나 - 데이터 사이언스, AI, 머신러닝, 딥러닝 강의 및 워크샵',
  },
  alternates: {
    canonical: 'https://suanlab.com/course',
  },
};

function getCourseStats(courseList: Course[]) {
  const years = [...new Set(courseList.map((c) => c.date.substring(0, 4)))].sort((a, b) =>
    b.localeCompare(a)
  );

  const categoryCounts: Record<string, number> = {
    all: courseList.length,
    seminar: courseList.filter((c) => c.category === 'seminar').length,
    lecture: courseList.filter((c) => c.category === 'lecture').length,
    camp: courseList.filter((c) => c.category === 'camp').length,
    training: courseList.filter((c) => c.category === 'training').length,
  };

  return { years, categoryCounts };
}

export default function CoursePage() {
  const { years, categoryCounts } = getCourseStats(courses);

  return (
    <>
      <EventCollectionJsonLd events={courses.map((c) => ({
        name: c.title,
        date: c.date,
        venue: c.organization,
      }))} />
      <PageHeader
        title="Course"
        subtitleKey="pageheader.course.subtitle"
        breadcrumbs={[{ label: 'Course' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <CourseClient
            courses={courses}
            categories={courseCategories}
            years={years}
            categoryCounts={categoryCounts}
          />
        </div>
      </section>
    </>
  );
}
