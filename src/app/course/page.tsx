'use client';

import { useState } from 'react';
import { Calendar, MapPin, FileText, ExternalLink, Presentation, Code, Building2, BookOpen, GraduationCap, Tent } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { courses, courseCategories, Course } from '@/data/courses';

const categoryIcons = {
  seminar: Presentation,
  lecture: BookOpen,
  camp: Tent,
  training: GraduationCap,
};

const categoryColors = {
  seminar: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
  lecture: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
  camp: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300',
  training: 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-300',
};

export default function CoursePage() {
  const [selectedCategory, setSelectedCategory] = useState<Course['category'] | 'all'>('all');
  const [selectedYear, setSelectedYear] = useState<string>('all');

  // Get unique years from courses
  const years = [...new Set(courses.map((c) => c.date.substring(0, 4)))].sort((a, b) => b.localeCompare(a));

  // Filter courses
  const filteredCourses = courses.filter((course) => {
    const categoryMatch = selectedCategory === 'all' || course.category === selectedCategory;
    const yearMatch = selectedYear === 'all' || course.date.includes(selectedYear);
    return categoryMatch && yearMatch;
  });

  // Count by category
  const categoryCounts = {
    all: courses.length,
    seminar: courses.filter((c) => c.category === 'seminar').length,
    lecture: courses.filter((c) => c.category === 'lecture').length,
    camp: courses.filter((c) => c.category === 'camp').length,
    training: courses.filter((c) => c.category === 'training').length,
  };

  return (
    <>
      <PageHeader
        title="Course"
        subtitle="특강 및 세미나 경력"
        breadcrumbs={[{ label: 'Course' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          {/* Summary Stats */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
            <Card
              className={`cursor-pointer transition-all ${selectedCategory === 'all' ? 'ring-2 ring-primary' : 'hover:shadow-md'}`}
              onClick={() => setSelectedCategory('all')}
            >
              <CardContent className="p-4 text-center">
                <p className="text-2xl font-bold text-primary">{categoryCounts.all}</p>
                <p className="text-sm text-muted-foreground">전체</p>
              </CardContent>
            </Card>
            {(Object.keys(courseCategories) as Course['category'][]).map((category) => {
              const Icon = categoryIcons[category];
              return (
                <Card
                  key={category}
                  className={`cursor-pointer transition-all ${selectedCategory === category ? 'ring-2 ring-primary' : 'hover:shadow-md'}`}
                  onClick={() => setSelectedCategory(category)}
                >
                  <CardContent className="p-4 text-center">
                    <Icon className="h-5 w-5 mx-auto mb-1 text-muted-foreground" />
                    <p className="text-2xl font-bold">{categoryCounts[category]}</p>
                    <p className="text-xs text-muted-foreground">{courseCategories[category].label}</p>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* Year Filter */}
          <div className="flex flex-wrap gap-2 mb-8">
            <Button
              variant={selectedYear === 'all' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedYear('all')}
            >
              전체 연도
            </Button>
            {years.map((year) => (
              <Button
                key={year}
                variant={selectedYear === year ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedYear(year)}
              >
                {year}
              </Button>
            ))}
          </div>

          {/* Results count */}
          <div className="mb-6">
            <Badge variant="outline" className="text-sm">
              {filteredCourses.length}개 표시 중
            </Badge>
          </div>

          {/* Course Grid */}
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {filteredCourses.map((course) => {
              const CategoryIcon = categoryIcons[course.category];
              return (
                <Card key={course.id} className="h-full flex flex-col">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between mb-2">
                      <Badge className={categoryColors[course.category]}>
                        <CategoryIcon className="mr-1 h-3 w-3" />
                        {courseCategories[course.category].label}
                      </Badge>
                      <span className="text-xs text-muted-foreground">{course.date.substring(0, 4)}</span>
                    </div>
                    <CardTitle className="text-lg leading-tight">{course.title}</CardTitle>
                    {course.subtitle && (
                      <p className="text-sm text-muted-foreground">{course.subtitle}</p>
                    )}
                  </CardHeader>
                  <CardContent className="flex-1 flex flex-col">
                    <div className="space-y-2 text-sm text-muted-foreground mb-4">
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 shrink-0" />
                        <span>{course.date}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <MapPin className="h-4 w-4 shrink-0" />
                        <span>{course.location}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Building2 className="h-4 w-4 shrink-0" />
                        <span>{course.organization}</span>
                      </div>
                    </div>

                    {course.materials && course.materials.length > 0 && (
                      <div className="mt-auto space-y-2">
                        <p className="text-sm font-medium">자료:</p>
                        <div className="flex flex-wrap gap-2">
                          {course.materials.map((material, idx) => (
                            <Button
                              key={idx}
                              variant="outline"
                              size="sm"
                              asChild
                              className="h-auto py-1 px-2 text-xs"
                            >
                              <a href={material.url} target="_blank" rel="noopener noreferrer">
                                {material.type === 'colab' ? (
                                  <Code className="mr-1 h-3 w-3" />
                                ) : material.type === 'pdf' ? (
                                  <FileText className="mr-1 h-3 w-3" />
                                ) : (
                                  <ExternalLink className="mr-1 h-3 w-3" />
                                )}
                                {material.title}
                              </a>
                            </Button>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </section>
    </>
  );
}
