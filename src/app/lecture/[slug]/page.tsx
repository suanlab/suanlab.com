import Image from 'next/image';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { Building, Calendar, BookOpen, CheckCircle, Youtube, ArrowRight, GraduationCap, ChevronRight } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { lectures, getLectureBySlug } from '@/data/lectures';
import { getPlaylistBySlug } from '@/data/youtube';

export async function generateStaticParams() {
  return lectures.map((lecture) => ({
    slug: lecture.slug,
  }));
}

export async function generateMetadata({ params }: { params: { slug: string } }) {
  const lecture = getLectureBySlug(params.slug);
  if (!lecture) return { title: 'Not Found' };

  return {
    title: `${lecture.titleEn} | Lecture | SuanLab`,
    description: lecture.description,
  };
}

export default function LectureDetailPage({ params }: { params: { slug: string } }) {
  const lecture = getLectureBySlug(params.slug);

  if (!lecture) {
    notFound();
  }

  return (
    <>
      <PageHeader
        title={lecture.titleEn}
        subtitle={lecture.titleKo}
        breadcrumbs={[
          { label: 'Lecture', href: '/lecture' },
          { label: lecture.titleEn },
        ]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="grid gap-12 lg:grid-cols-3">
            {/* Left: Course Info */}
            <div className="lg:col-span-2">
              {/* Hero Image & Description */}
              <div className="relative aspect-video overflow-hidden rounded-xl shadow-lg mb-8">
                <Image
                  src={lecture.image}
                  alt={lecture.titleEn}
                  fill
                  className="object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <div className="absolute bottom-4 left-4">
                  <Badge className="bg-primary text-primary-foreground">
                    {lecture.titleKo}
                  </Badge>
                </div>
              </div>

              {/* Description */}
              <div className="mb-8">
                <h2 className="text-2xl font-bold mb-4">Course Overview</h2>
                <p className="text-muted-foreground leading-relaxed mb-4">
                  {lecture.description}
                </p>
                <p className="text-muted-foreground leading-relaxed">
                  {lecture.descriptionKo}
                </p>
              </div>

              {/* Topics */}
              <Card className="mb-8">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BookOpen className="h-5 w-5 text-primary" />
                    Course Topics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-3 sm:grid-cols-2">
                    {lecture.topics.map((topic, idx) => (
                      <div key={idx} className="flex items-start gap-2">
                        <CheckCircle className="h-5 w-5 text-green-500 shrink-0 mt-0.5" />
                        <span className="text-sm">{topic}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Teaching History */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Calendar className="h-5 w-5 text-primary" />
                    Teaching History
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {lecture.history.map((item, idx) => (
                      <div key={idx} className="flex items-start gap-4 pb-4 border-b last:border-0 last:pb-0">
                        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 shrink-0">
                          <GraduationCap className="h-5 w-5 text-primary" />
                        </div>
                        <div className="flex-1">
                          <p className="font-medium">{item.semester}</p>
                          <p className="text-sm text-muted-foreground">{item.institution}</p>
                          <div className="flex flex-wrap gap-2 mt-2">
                            {item.courses.map((course, cidx) => (
                              <Badge key={cidx} variant="secondary" className="text-xs">
                                {course}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right: Sidebar */}
            <div className="lg:col-span-1">
              <div className="sticky top-24 space-y-6">
                {/* Quick Info */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Course Info</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                        <Building className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Institution</p>
                        <p className="text-sm font-medium">세명대학교</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                        <Calendar className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Total Semesters</p>
                        <p className="text-sm font-medium">{lecture.history.length} semesters</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Related YouTube */}
                {lecture.relatedYoutube && lecture.relatedYoutube.length > 0 && (
                  <Card className="border-red-200 dark:border-red-900/30">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <Youtube className="h-5 w-5 text-red-600" />
                        Related YouTube
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground mb-4">
                        이 강의와 관련된 무료 YouTube 강좌를 확인하세요.
                      </p>
                      <div className="space-y-2">
                        {lecture.relatedYoutube.map((url) => {
                          const slug = url.replace('/youtube/', '');
                          const playlist = getPlaylistBySlug(slug);
                          const displayName = playlist?.titleKo || slug.toUpperCase();
                          return (
                            <Link
                              key={url}
                              href={url}
                              className="flex items-center justify-between p-2 rounded-lg hover:bg-muted transition-colors"
                            >
                              <span className="text-sm text-primary">{displayName}</span>
                              <ChevronRight className="h-4 w-4 text-muted-foreground" />
                            </Link>
                          );
                        })}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* CTA */}
                <Card className="bg-gradient-to-br from-primary/5 to-primary/10">
                  <CardContent className="p-6 text-center">
                    <h3 className="font-semibold mb-2">Explore More Lectures</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      다른 강의들도 확인해보세요.
                    </p>
                    <Button className="w-full" asChild>
                      <Link href="/lecture">
                        All Lectures
                        <ArrowRight className="ml-2 h-4 w-4" />
                      </Link>
                    </Button>
                  </CardContent>
                </Card>

                {/* YouTube CTA */}
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/youtube">
                    <Youtube className="mr-2 h-4 w-4 text-red-600" />
                    YouTube Lectures
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
