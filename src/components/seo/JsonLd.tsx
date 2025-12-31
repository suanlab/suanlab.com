import Script from 'next/script';

const BASE_URL = 'https://suanlab.com';

// Organization Schema
export function OrganizationJsonLd() {
  const schema = {
    '@context': 'https://schema.org',
    '@type': 'ResearchOrganization',
    name: 'SuanLab',
    alternateName: '수안랩',
    url: BASE_URL,
    logo: `${BASE_URL}/assets/images/logo.png`,
    description:
      '이수안 교수의 데이터 사이언스 연구실 - 딥러닝, 머신러닝, 빅데이터, 자연어처리, 컴퓨터 비전 연구',
    foundingDate: '2021',
    founder: {
      '@type': 'Person',
      name: '이수안',
      alternateName: 'Suan Lee',
    },
    sameAs: [
      'https://www.youtube.com/@suloopy',
      'https://github.com/suanlab',
    ],
    knowsAbout: [
      'Data Science',
      'Deep Learning',
      'Machine Learning',
      'Natural Language Processing',
      'Computer Vision',
      'Big Data',
      'Artificial Intelligence',
    ],
  };

  return (
    <Script
      id="organization-jsonld"
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(schema) }}
    />
  );
}

// Person Schema for Professor Suan
export function PersonJsonLd() {
  const schema = {
    '@context': 'https://schema.org',
    '@type': 'Person',
    name: '이수안',
    alternateName: 'Suan Lee',
    url: `${BASE_URL}/suan`,
    image: `${BASE_URL}/assets/images/profile/suan.jpg`,
    jobTitle: '교수',
    worksFor: {
      '@type': 'Organization',
      name: '세명대학교',
      alternateName: 'Semyung University',
    },
    alumniOf: [
      {
        '@type': 'CollegeOrUniversity',
        name: '서울대학교',
        alternateName: 'Seoul National University',
      },
    ],
    knowsAbout: [
      'Data Science',
      'Deep Learning',
      'Machine Learning',
      'Natural Language Processing',
      'Computer Vision',
      'Big Data',
      'Python',
      'PyTorch',
      'TensorFlow',
    ],
    sameAs: [
      'https://www.youtube.com/@suloopy',
      'https://github.com/suanlab',
      'https://scholar.google.com/citations?user=YOUR_ID',
    ],
  };

  return (
    <Script
      id="person-jsonld"
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(schema) }}
    />
  );
}

// WebSite Schema
export function WebSiteJsonLd() {
  const schema = {
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    name: 'SuanLab',
    alternateName: '수안랩',
    url: BASE_URL,
    description:
      '이수안 교수의 데이터 사이언스 연구실 - 딥러닝, 머신러닝, 빅데이터 연구 및 교육',
    inLanguage: 'ko-KR',
    publisher: {
      '@type': 'Person',
      name: '이수안',
    },
    potentialAction: {
      '@type': 'SearchAction',
      target: {
        '@type': 'EntryPoint',
        urlTemplate: `${BASE_URL}/blog?q={search_term_string}`,
      },
      'query-input': 'required name=search_term_string',
    },
  };

  return (
    <Script
      id="website-jsonld"
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(schema) }}
    />
  );
}

// Article Schema for blog posts
interface ArticleJsonLdProps {
  title: string;
  description: string;
  url: string;
  datePublished: string;
  dateModified?: string;
  author?: string;
  image?: string;
}

export function ArticleJsonLd({
  title,
  description,
  url,
  datePublished,
  dateModified,
  author = '이수안',
  image,
}: ArticleJsonLdProps) {
  const schema = {
    '@context': 'https://schema.org',
    '@type': 'Article',
    headline: title,
    description: description,
    url: url,
    datePublished: datePublished,
    dateModified: dateModified || datePublished,
    author: {
      '@type': 'Person',
      name: author,
      url: `${BASE_URL}/suan`,
    },
    publisher: {
      '@type': 'Organization',
      name: 'SuanLab',
      logo: {
        '@type': 'ImageObject',
        url: `${BASE_URL}/assets/images/logo.png`,
      },
    },
    image: image || `${BASE_URL}/assets/images/og-image.jpg`,
    mainEntityOfPage: {
      '@type': 'WebPage',
      '@id': url,
    },
  };

  return (
    <Script
      id="article-jsonld"
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(schema) }}
    />
  );
}

// Educational Course Schema
interface CourseJsonLdProps {
  name: string;
  description: string;
  url: string;
  provider?: string;
}

export function CourseJsonLd({
  name,
  description,
  url,
  provider = 'SuanLab',
}: CourseJsonLdProps) {
  const schema = {
    '@context': 'https://schema.org',
    '@type': 'Course',
    name: name,
    description: description,
    url: url,
    provider: {
      '@type': 'Organization',
      name: provider,
      url: BASE_URL,
    },
    instructor: {
      '@type': 'Person',
      name: '이수안',
    },
    inLanguage: 'ko',
  };

  return (
    <Script
      id="course-jsonld"
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(schema) }}
    />
  );
}

// BreadcrumbList Schema
interface BreadcrumbItem {
  name: string;
  url: string;
}

export function BreadcrumbJsonLd({ items }: { items: BreadcrumbItem[] }) {
  const schema = {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: items.map((item, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: item.name,
      item: item.url,
    })),
  };

  return (
    <Script
      id="breadcrumb-jsonld"
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(schema) }}
    />
  );
}
