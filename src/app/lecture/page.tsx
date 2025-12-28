import Image from 'next/image';
import Link from 'next/link';
import { GraduationCap, Building, BookOpen, Youtube, ChevronRight } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { lectures, graduateCourses, teachingHistory } from '@/data/lectures';

export const metadata = {
  title: 'Lecture | SuanLab',
  description: 'University courses taught by Professor Suan Lee - AI, Deep Learning, NLP, Computer Vision, and more',
};

export default function LecturePage() {
  return (
    <>
      <PageHeader
        title="Lecture"
        subtitle="University courses taught by Professor Suan Lee"
        breadcrumbs={[{ label: 'Lecture' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          {/* Teaching Summary */}
          <div className="mb-12">
            <h2 className="text-2xl font-bold mb-6">Teaching Experience</h2>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {Object.values(teachingHistory).map((univ) => (
                <Card key={univ.name}>
                  <CardContent className="p-4">
                    <div className="flex items-start gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 shrink-0">
                        <Building className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold">{univ.name}</h3>
                        <p className="text-sm text-muted-foreground">{univ.role}</p>
                        <p className="text-xs text-muted-foreground mt-1">{univ.period}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Course Categories */}
          <div className="mb-12">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">Courses</h2>
              <Badge variant="outline" className="text-sm">
                {lectures.length} courses
              </Badge>
            </div>

            <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
              {lectures.map((lecture) => (
                <Link key={lecture.slug} href={`/lecture/${lecture.slug}`}>
                  <Card className="group h-full overflow-hidden transition-all hover:shadow-lg hover:-translate-y-1">
                    <div className="relative aspect-[16/10] overflow-hidden">
                      <Image
                        src={lecture.image}
                        alt={lecture.titleEn}
                        fill
                        className="object-cover transition-transform group-hover:scale-105"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent" />
                      <div className="absolute bottom-3 left-3 right-3">
                        <h3 className="text-base font-semibold text-white">
                          {lecture.titleKo}
                        </h3>
                        <p className="text-xs text-white/80">{lecture.titleEn}</p>
                      </div>
                    </div>
                    <CardContent className="p-3">
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {lecture.descriptionKo}
                      </p>
                      <div className="flex items-center justify-between mt-3">
                        <Badge variant="secondary" className="text-xs">
                          {lecture.history.length} semesters
                        </Badge>
                        <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          </div>

          {/* Graduate Courses */}
          <div className="mb-12">
            <div className="flex items-center gap-2 mb-6">
              <GraduationCap className="h-6 w-6 text-primary" />
              <h2 className="text-2xl font-bold">Graduate Courses</h2>
            </div>
            <Card>
              <CardContent className="p-6">
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {graduateCourses.map((course) => (
                    <div key={course.name} className="flex items-start gap-3">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 shrink-0">
                        <BookOpen className="h-4 w-4 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium text-sm">{course.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {course.semesters.join(', ')}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Related YouTube */}
          <Card className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-950/20 dark:to-orange-950/20 border-red-200 dark:border-red-900/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Youtube className="h-5 w-5 text-red-600" />
                Related YouTube Lectures
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground mb-4">
                강의 내용과 관련된 무료 YouTube 강좌를 확인하세요.
              </p>
              <Link
                href="/youtube"
                className="inline-flex items-center gap-2 text-red-600 hover:text-red-700 font-medium"
              >
                View YouTube Lectures
                <ChevronRight className="h-4 w-4" />
              </Link>
            </CardContent>
          </Card>
        </div>
      </section>
    </>
  );
}
