import LocalizedContent from '@/components/localized-content';
import { catalogEnglish } from '@/data/i18n/catalog';
import { englishText } from '@/data/i18n/translate';
import { localizedMarkup } from '@/lib/localized-markup';
import { getLectureContent } from '@/lib/lecture-content';
import { notFound } from 'next/navigation';
import { lectures, getLectureBySlug } from '@/data/lectures';
import { buildLectureSlides } from '@/lib/presentation';
import LectureDeck from '@/components/presentation/LectureDeck';
import { site } from '@/config/site';
export function generateStaticParams() { return lectures.map(({ slug }) => ({ slug })); }
export function generateMetadata({ params }: { params: { slug: string } }) {
  const lecture = getLectureBySlug(params.slug);
  return { title: `${lecture?.titleKo || 'Lecture'} · 발표`, alternates: { canonical: `${site.url}/lecture/${params.slug}/present/` } };
}
export default async function PresentationPage({ params }: { params: { slug: string } }) {
  const lecture = getLectureBySlug(params.slug); if (!lecture) notFound();
  const content = await getLectureContent(lecture.slug);
  const translate = (text: string) => englishText(text, catalogEnglish);
  const slides = (content?.slides || buildLectureSlides(lecture)).map(slide => ({ ...slide,
    titleEn: translate(slide.title), pointsEn: slide.points.map(translate), notesEn: slide.notes ? translate(slide.notes) : undefined,
    contentHtmlEn: slide.contentHtml ? localizedMarkup(slide.contentHtml, translate) : undefined,
  }));
  return <LocalizedContent translations={catalogEnglish}><LectureDeck slides={slides} slug={lecture.slug} title={lecture.titleKo} /></LocalizedContent>;
}
