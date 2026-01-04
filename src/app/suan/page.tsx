import Image from 'next/image';
import { GraduationCap, Building, Mail, Phone, MapPin, Check, Globe, Instagram, Facebook, Linkedin, Youtube, Calendar, Award, BookOpen, Plane, Users, Briefcase, Heart, Scale } from 'lucide-react';
import { academicActivities, journalMemberships, journalReviews, journalReviewStats, conferenceReviews, advisoryActivities, activityCategories } from '@/data/academic-activities';
import { awards, awardStats, awardCategories } from '@/data/awards';
import { Trophy } from 'lucide-react';
import { visitedCountries, overseasExperiences, overseasStats } from '@/data/overseas-experiences';
import { networkCategories, networkStats } from '@/data/networks';

// X (formerly Twitter) icon
const XIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 24 24" className={className} fill="currentColor">
    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
  </svg>
);
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export const metadata = {
  title: 'Suan | SuanLab',
  description: 'Professor Suan Lee - Data Science & Artificial Intelligence Researcher',
};

const skills = [
  { name: 'Deep Learning & ML', detail: 'TensorFlow, Keras, PyTorch', percent: 100 },
  { name: 'Big Data', detail: 'Hadoop, Spark, HBase, Hive, ZooKeeper', percent: 95 },
  { name: 'Database', detail: 'Oracle, PostgreSQL, MySQL, MariaDB', percent: 95 },
  { name: 'Programming', detail: 'Python, Java, Scala, C++, C', percent: 90 },
  { name: 'Statistics & Data Mining', detail: 'R, MATLAB, Weka, RapidMiner', percent: 95 },
  { name: 'Web', detail: 'HTML5, JavaScript, CSS, Node.js', percent: 85 },
  { name: 'Multimedia', detail: 'Photoshop, Illustrator, Premiere', percent: 80 },
];

const socialLinks = [
  { icon: Youtube, label: 'YouTube (이수안컴퓨터연구소, 구독자 3.7만+)', url: 'https://www.youtube.com/@suanlab' },
  { icon: Linkedin, label: 'LinkedIn', url: 'https://www.linkedin.com/in/suanlab' },
  { icon: Instagram, label: 'Instagram', url: 'https://www.instagram.com/suanlab/' },
  { icon: Facebook, label: 'Facebook', url: 'http://www.facebook.com/suanlab' },
  { icon: XIcon, label: 'X', url: 'https://x.com/leesuanlab' },
];

const researchInterests = [
  'Data Science & Big Data',
  'Deep Learning & Machine Learning',
  'Natural Language Processing',
  'Computer Vision',
  'Graphs and Tensors',
  'Spatio-Temporal Analysis',
  'Audio & Speech Processing',
];

export default function SuanPage() {
  return (
    <>
      <PageHeader
        title="Professor Suan Lee"
        subtitle="Data Science & Artificial Intelligence Researcher"
        breadcrumbs={[{ label: 'Suan' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="grid gap-12 lg:grid-cols-3">
            {/* Left Sidebar */}
            <div className="lg:col-span-1">
              {/* Profile Card */}
              <Card className="mb-6 overflow-hidden">
                <div className="relative aspect-square bg-gradient-to-br from-cyan-600 via-blue-600 to-indigo-700">
                  <Image
                    src="/assets/images/suan/profile.jpg"
                    alt="Suan Lee"
                    fill
                    className="object-cover object-top"
                  />
                </div>
                <CardContent className="p-6 text-center">
                  <h2 className="text-xl font-bold">Suan Lee</h2>
                  <p className="text-sm text-muted-foreground mt-1">
                    <GraduationCap className="inline h-4 w-4 mr-1" />
                    Assistant Professor / Ph.D
                  </p>
                  <div className="flex justify-center gap-3 mt-4">
                    {socialLinks.map((link) => (
                      <a
                        key={link.label}
                        href={link.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex h-9 w-9 items-center justify-center rounded-full bg-muted hover:bg-primary hover:text-primary-foreground transition-colors"
                        aria-label={link.label}
                      >
                        <link.icon className="h-4 w-4" />
                      </a>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Business Cards */}
              <Card className="mb-6 overflow-hidden">
                <Image
                  src="/assets/images/suan/card-kor.png"
                  alt="Business Card (Korean)"
                  width={300}
                  height={180}
                  className="w-full h-auto"
                />
                <Image
                  src="/assets/images/suan/card-eng.png"
                  alt="Business Card (English)"
                  width={300}
                  height={180}
                  className="w-full h-auto"
                />
              </Card>

              {/* Skills */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Skills</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {skills.map((skill) => (
                    <div key={skill.name}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium">{skill.name}</span>
                        <span className="text-muted-foreground">{skill.percent}%</span>
                      </div>
                      <p className="text-xs text-muted-foreground mb-2">{skill.detail}</p>
                      <div className="h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full bg-primary rounded-full transition-all"
                          style={{ width: `${skill.percent}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              {/* Networks */}
              <Card className="mt-6">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Users className="h-5 w-5 text-primary" />
                    Networks
                  </CardTitle>
                  <p className="text-xs text-muted-foreground mt-1">
                    {networkStats.totalConnections}+ Professional Connections
                  </p>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="space-y-4">
                    {networkCategories.map((category) => {
                      const IconComponent = category.icon === 'GraduationCap' ? GraduationCap
                        : category.icon === 'Building' ? Building
                        : category.icon === 'Briefcase' ? Briefcase
                        : category.icon === 'Scale' ? Scale
                        : category.icon === 'Heart' ? Heart
                        : Users;
                      return (
                        <details key={category.name} className="group">
                          <summary className="flex items-center justify-between cursor-pointer list-none py-2 hover:bg-muted/50 rounded-md px-2 -mx-2">
                            <div className="flex items-center gap-2">
                              <IconComponent className="h-4 w-4 text-muted-foreground" />
                              <span className="font-medium text-sm">{category.nameKo}</span>
                              <span className="text-xs text-muted-foreground">({category.items.length})</span>
                            </div>
                            <svg className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                          </summary>
                          <div className="mt-2 pl-6 max-h-60 overflow-y-auto">
                            <div className="flex flex-wrap gap-1">
                              {category.items.map((item, idx) => (
                                <span
                                  key={idx}
                                  className={`inline-block px-2 py-0.5 rounded text-xs ${category.color}`}
                                >
                                  {item}
                                </span>
                              ))}
                            </div>
                          </div>
                        </details>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>

              {/* Visited Countries */}
              <Card className="mt-6">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Globe className="h-5 w-5 text-primary" />
                    Visited Countries
                  </CardTitle>
                  <div className="flex gap-4 mt-2 text-sm text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Plane className="h-4 w-4" />
                      {overseasStats.totalTrips}회
                    </span>
                    <span>{overseasStats.totalCountries}개국</span>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  {/* Countries by Continent */}
                  <div className="space-y-3">
                    {(['Europe', 'Asia', 'America', 'Oceania', 'Africa'] as const).map((continent) => {
                      const countries = visitedCountries.filter(c => c.continent === continent);
                      if (countries.length === 0) return null;
                      return (
                        <div key={continent}>
                          <p className="text-xs font-medium text-muted-foreground mb-1.5">{continent}</p>
                          <div className="flex flex-wrap gap-1">
                            {countries.map((country) => (
                              <span
                                key={country.code}
                                className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-muted text-foreground"
                                title={country.name}
                              >
                                <span>{country.flag}</span>
                                <span className="hidden sm:inline">{country.name}</span>
                              </span>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Recent Trips */}
                  <div className="mt-4 pt-3 border-t">
                    <p className="text-xs font-medium text-muted-foreground mb-2">Recent Trips</p>
                    <div className="space-y-1.5">
                      {overseasExperiences.slice(0, 5).map((exp) => (
                        <div key={exp.id} className="flex items-center gap-2 text-xs">
                          <span className="text-muted-foreground w-16 shrink-0">{exp.period.split(' ~ ')[0]}</span>
                          <span className="font-medium truncate">{exp.countries.join(', ')}</span>
                          <span className="text-muted-foreground truncate">({exp.purpose})</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right Content */}
            <div className="lg:col-span-2">
              {/* About Me */}
              <div className="mb-12">
                <h2 className="text-2xl font-bold mb-6">
                  About <span className="text-primary">Me</span>
                </h2>
                <p className="text-lg text-muted-foreground leading-relaxed mb-4">
                  &quot;데이터와 AI를 이용해 세상을 이롭게하자!&quot;라는 생각을 가진 데이터 과학자이자 AI 연구자입니다.
                </p>
                <p className="text-muted-foreground leading-relaxed">
                  새로운 연구와 기술에 흥미를 가지며, 인공지능, 머신러닝, 딥러닝, 자연어처리, 컴퓨터비전,
                  오디오음성처리, 빅데이터에 관심이 많습니다. 머신러닝, 딥러닝, 데이터마이닝, 데이터웨어하우스,
                  데이터베이스 분야에서 19년간 연구하였고, 인메모리 데이터베이스와 실시간 스트림 데이터 처리 엔진,
                  빅데이터 플랫폼과 관련해 3년 이상의 개발 경력을 쌓았습니다.
                </p>
              </div>

              {/* Experience */}
              <Card className="mb-12">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Building className="h-5 w-5 text-primary" />
                    Work Experience
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="border-l-2 border-primary pl-4">
                    <p className="font-medium">세명대학교 IT엔지니어링대학 컴퓨터학부</p>
                    <p className="text-sm text-muted-foreground">조교수 (2021.03 - 현재)</p>
                  </div>
                  <div className="border-l-2 border-muted pl-4">
                    <p className="font-medium">인하대학교 VOICE AI 연구소</p>
                    <p className="text-sm text-muted-foreground">책임연구원 (2019.12 - 2020.12)</p>
                  </div>
                  <div className="border-l-2 border-muted pl-4">
                    <p className="font-medium">강원대학교 SW중심대학</p>
                    <p className="text-sm text-muted-foreground">연구교수/객원교수 (2019.02 - 2019.11)</p>
                  </div>
                  <div className="border-l-2 border-muted pl-4">
                    <p className="font-medium">강원대학교 정보통신연구소</p>
                    <p className="text-sm text-muted-foreground">선임연구원 Post-Doc. (2018.02 - 2019.01)</p>
                  </div>
                  <div className="border-l-2 border-muted pl-4">
                    <p className="font-medium">㈜알티베이스 개발연구본부</p>
                    <p className="text-sm text-muted-foreground">연구원 (2012.10 - 2015.12)</p>
                  </div>
                </CardContent>
              </Card>

              {/* Education & Research */}
              <div className="grid gap-6 md:grid-cols-2 mb-12">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <GraduationCap className="h-5 w-5 text-primary" />
                      Education
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="border-l-2 border-primary pl-4">
                      <p className="font-medium">Ph.D in Computer Science</p>
                      <p className="text-sm text-muted-foreground">Kangwon National University (2010-2017)</p>
                    </div>
                    <div className="border-l-2 border-muted pl-4">
                      <p className="font-medium">M.S in Computer Science</p>
                      <p className="text-sm text-muted-foreground">Kangwon National University (2008-2010)</p>
                    </div>
                    <div className="border-l-2 border-muted pl-4">
                      <p className="font-medium">B.S in Computer Science</p>
                      <p className="text-sm text-muted-foreground">Kangwon National University (2004-2008)</p>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Building className="h-5 w-5 text-primary" />
                      Research Interests
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {researchInterests.map((interest) => (
                        <li key={interest} className="flex items-center gap-2">
                          <Check className="h-4 w-4 text-green-500" />
                          <span className="text-sm">{interest}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              </div>

              {/* Awards */}
              <div className="mb-12">
                <h2 className="text-2xl font-bold mb-6">
                  <Trophy className="inline h-6 w-6 text-primary mr-2" />
                  Awards & <span className="text-primary">Honors</span>
                </h2>

                {/* Award Statistics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <Card className="text-center">
                    <CardContent className="p-4">
                      <p className="text-3xl font-bold text-primary">{awardStats.total}</p>
                      <p className="text-sm text-muted-foreground">총 수상</p>
                    </CardContent>
                  </Card>
                  <Card className="text-center">
                    <CardContent className="p-4">
                      <p className="text-3xl font-bold text-blue-600">{awardStats.paper}</p>
                      <p className="text-sm text-muted-foreground">논문상</p>
                    </CardContent>
                  </Card>
                  <Card className="text-center">
                    <CardContent className="p-4">
                      <p className="text-3xl font-bold text-green-600">{awardStats.contribution}</p>
                      <p className="text-sm text-muted-foreground">공로상</p>
                    </CardContent>
                  </Card>
                  <Card className="text-center">
                    <CardContent className="p-4">
                      <p className="text-3xl font-bold text-purple-600">{awardStats.teaching}</p>
                      <p className="text-sm text-muted-foreground">우수교원</p>
                    </CardContent>
                  </Card>
                </div>

                {/* Awards List */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Award className="h-5 w-5 text-primary" />
                      수상 내역
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="max-h-96 overflow-y-auto pr-2 space-y-3">
                      {awards.map((award) => (
                        <div key={award.id} className="flex items-start gap-3 pb-3 border-b border-muted last:border-0">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium shrink-0 ${awardCategories[award.category].color}`}>
                            {awardCategories[award.category].label}
                          </span>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-sm">{award.award}</p>
                            {award.title && (
                              <p className="text-xs text-muted-foreground mt-0.5">&quot;{award.title}&quot;</p>
                            )}
                            {award.authors && (
                              <p className="text-xs text-primary mt-0.5">{award.authors}</p>
                            )}
                            <p className="text-xs text-muted-foreground">{award.date} · {award.organization}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground mt-4 text-center">
                      총 {awards.length}건 (2017 - 2025)
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Academic Activities */}
              <div className="mb-12">
                <h2 className="text-2xl font-bold mb-6">
                  Academic <span className="text-primary">Activities</span>
                </h2>

                {/* Conference & Workshop Activities */}
                <Card className="mb-6">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Calendar className="h-5 w-5 text-primary" />
                      국내외 학회, 워크샵, 포럼 활동
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="max-h-80 overflow-y-auto pr-2 space-y-3">
                      {academicActivities.map((activity) => (
                        <div key={activity.id} className="flex items-start gap-3 pb-3 border-b border-muted last:border-0">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium shrink-0
                            ${activity.category === 'conference' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300' : ''}
                            ${activity.category === 'workshop' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300' : ''}
                            ${activity.category === 'forum' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300' : ''}
                          `}>
                            {activityCategories[activity.category].label}
                          </span>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-sm">{activity.organization}</p>
                            <p className="text-xs text-muted-foreground">{activity.period} · {activity.role}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground mt-4 text-center">
                      총 {academicActivities.length}건
                    </p>
                  </CardContent>
                </Card>

                {/* Paper Review Activities */}
                <Card className="mb-6">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BookOpen className="h-5 w-5 text-primary" />
                      논문 리뷰 활동
                    </CardTitle>
                    <p className="text-sm text-muted-foreground mt-1">
                      {conferenceReviews.length}개 학회, {journalReviewStats.totalJournals}개 저널, 총 {journalReviewStats.totalReviews}건+ 리뷰
                    </p>
                  </CardHeader>
                  <CardContent>
                    {/* Conference Reviews */}
                    <div className="mb-4">
                      <p className="text-xs font-medium text-muted-foreground mb-2">Conference</p>
                      <div className="flex flex-wrap gap-1.5">
                        {conferenceReviews.map((item) => (
                          <span
                            key={item.id}
                            className="inline-flex items-center px-2.5 py-1 rounded-md bg-primary/10 text-primary text-xs font-medium"
                            title={item.fullName}
                          >
                            {item.conference}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Journal Reviews */}
                    <div className="pt-4 border-t">
                      <p className="text-xs font-medium text-muted-foreground mb-2">Journal</p>
                      <div className="max-h-72 overflow-y-auto pr-2 space-y-4">
                        {(() => {
                          // Group by publisher
                          const grouped = journalReviews.reduce((acc, item) => {
                            if (!acc[item.publisher]) {
                              acc[item.publisher] = { journals: [], totalReviews: 0 };
                            }
                            acc[item.publisher].journals.push(item);
                            acc[item.publisher].totalReviews += item.reviewCount;
                            return acc;
                          }, {} as Record<string, { journals: typeof journalReviews; totalReviews: number }>);

                          // Sort publishers by total review count
                          const sortedPublishers = Object.entries(grouped)
                            .sort(([, a], [, b]) => b.totalReviews - a.totalReviews);

                          return sortedPublishers.map(([publisher, data]) => (
                            <div key={publisher} className="pb-3 border-b border-muted last:border-0">
                              <div className="flex items-center justify-between mb-2">
                                <span className="font-semibold text-sm">{publisher}</span>
                                <span className="text-xs text-muted-foreground">
                                  {data.journals.length}개 저널 · {data.totalReviews}건
                                </span>
                              </div>
                              <div className="flex flex-wrap gap-1.5">
                                {data.journals.map((item) => (
                                  <span
                                    key={item.id}
                                    className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-muted text-xs"
                                    title={`${item.journal} - ${item.reviewCount}건 리뷰`}
                                  >
                                    <span className="font-bold text-primary">{item.reviewCount}</span>
                                    <span className="truncate max-w-[180px]">{item.journal}</span>
                                  </span>
                                ))}
                              </div>
                            </div>
                          ));
                        })()}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Membership */}
                <Card className="mb-6">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Award className="h-5 w-5 text-primary" />
                      학회 멤버십
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {journalMemberships.map((item) => (
                        <div key={item.id} className="flex items-start gap-3 py-2 border-b border-muted last:border-0">
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium shrink-0 bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300">
                            {item.role}
                          </span>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-sm">{item.organization}</p>
                            <p className="text-xs text-muted-foreground">{item.period}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Advisory Activities */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Award className="h-5 w-5 text-primary" />
                      자문, 심사, 평가위원 활동
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="max-h-80 overflow-y-auto pr-2 space-y-3">
                      {advisoryActivities.map((activity) => (
                        <div key={activity.id} className="flex items-start gap-3 pb-3 border-b border-muted last:border-0">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium shrink-0
                            ${activity.role.includes('평가') ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300' : ''}
                            ${activity.role.includes('자문') ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300' : ''}
                            ${activity.role.includes('심사') ? 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300' : ''}
                            ${activity.role.includes('면접') ? 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300' : ''}
                            ${activity.role.includes('멘토') ? 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-300' : ''}
                            ${activity.role.includes('외부') ? 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300' : ''}
                            ${activity.role.includes('기획') ? 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300' : ''}
                            ${activity.role.includes('집필') || activity.role.includes('감수') || activity.role.includes('검수') ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300' : ''}
                            ${activity.role.includes('전문위원') ? 'bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-300' : ''}
                          `}>
                            {activity.role}
                          </span>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-sm">{activity.topic}</p>
                            <p className="text-xs text-muted-foreground">{activity.period} · {activity.organization}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground mt-4 text-center">
                      총 {advisoryActivities.length}건
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Contact */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Mail className="h-5 w-5 text-primary" />
                    Contact
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-6 md:grid-cols-2">
                    <div className="space-y-4">
                      <div className="flex items-start gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                          <Building className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Office</p>
                          <p className="font-medium">School of Computer Science, Semyung University</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                          <MapPin className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Address</p>
                          <p className="font-medium">65 Semyung-ro, Jecheon-si, Chungcheongbuk-do, 27136 Korea</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                          <Phone className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Phone</p>
                          <p className="font-medium">+82-43-649-1273</p>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-start gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                          <Mail className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Email</p>
                          <a href="mailto:suanlab@gmail.com" className="font-medium text-primary hover:underline">
                            suanlab@gmail.com
                          </a>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                          <Mail className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Work Email</p>
                          <a href="mailto:suanlee@semyung.ac.kr" className="font-medium text-primary hover:underline">
                            suanlee@semyung.ac.kr
                          </a>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
