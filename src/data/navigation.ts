import { NavItem, SocialLink, ContactInfo } from '@/types';

export const navigation: NavItem[] = [
  {
    label: 'SUAN',
    href: '/suan',
    icon: 'et-profile-male',
  },
  {
    label: 'RESEARCH',
    href: '/research',
    icon: 'et-search',
    children: [
      { label: 'Data Science & Big Data', href: '/research/ds', icon: 'et-gears' },
      { label: 'Deep Learning & Machine Learning', href: '/research/dl', icon: 'et-layers' },
      { label: 'Natural Language Processing', href: '/research/nlp', icon: 'et-document' },
      { label: 'Computer Vision', href: '/research/cv', icon: 'et-pictures' },
      { label: 'Graphs and Tensors', href: '/research/graphs', icon: 'et-linegraph' },
      { label: 'Spatio-Temporal', href: '/research/st', icon: 'et-map' },
    ],
  },
  {
    label: 'PROJECT',
    href: '/project',
    icon: 'et-notebook',
  },
  {
    label: 'PUBLICATION',
    href: '/publication',
    icon: 'et-newspaper',
  },
  {
    label: 'BOOK',
    href: '/book',
    icon: 'et-book-open',
    children: [
      { label: 'Online Book', href: '/book/online', icon: 'fa fa-globe' },
      { label: 'Published Book', href: '/book/published', icon: 'fa fa-book' },
    ],
  },
  {
    label: 'LECTURE',
    href: '/lecture',
    icon: 'et-documents',
    children: [
      { label: 'Artificial Intelligence', href: '/lecture/ai', icon: 'fa fa-brain' },
      { label: 'Deep Learning', href: '/lecture/dl', icon: 'fa fa-layer-group' },
      { label: 'Machine Learning', href: '/lecture/ml', icon: 'fa fa-cogs' },
      { label: 'Natural Language Processing', href: '/lecture/nlp', icon: 'fa fa-comments' },
      { label: 'Computer Vision', href: '/lecture/cv', icon: 'fa fa-eye' },
      { label: 'Audio Signal Processing', href: '/lecture/asp', icon: 'fa fa-music' },
      { label: 'Big Data Analysis', href: '/lecture/bd', icon: 'fa fa-chart-bar' },
      { label: 'Database', href: '/lecture/db', icon: 'fa fa-database' },
    ],
  },
  {
    label: 'COURSE',
    href: '/course',
    icon: 'et-presentation',
  },
  {
    label: 'YOUTUBE',
    href: '/youtube',
    icon: 'et-video',
    children: [
      { label: 'Python Programming', href: '/youtube/pp', icon: 'fa fa-code' },
      { label: 'Python Game', href: '/youtube/pg', icon: 'fa fa-tasks' },
      { label: 'Web Programming', href: '/youtube/web', icon: 'fa fa-html5' },
      { label: 'Data Science', href: '/youtube/ds', icon: 'fa fa-flask' },
      { label: 'Data Collection', href: '/youtube/dc', icon: 'fa fa-gears' },
      { label: 'Data Analysis', href: '/youtube/da', icon: 'fa fa-wrench' },
      { label: 'Data Visualization', href: '/youtube/dv', icon: 'fa fa-bar-chart' },
      { label: 'Database', href: '/youtube/db', icon: 'fa fa-database' },
      { label: 'Big Data', href: '/youtube/bd', icon: 'fa fa-server' },
      { label: 'Machine Learning', href: '/youtube/ml', icon: 'fa fa-gears' },
      { label: 'Deep Learning', href: '/youtube/dl', icon: 'fa fa-sliders' },
      { label: 'Deep Learning Framework', href: '/youtube/dlf', icon: 'fa fa-magic' },
      { label: 'Computer Vision', href: '/youtube/cv', icon: 'fa fa-desktop' },
      { label: 'Natural Language Processing', href: '/youtube/nlp', icon: 'fa fa-search' },
      { label: 'Audio Speech Processing', href: '/youtube/asp', icon: 'fa fa-tasks' },
    ],
  },
];

export const socialLinks: SocialLink[] = [
  {
    name: 'YouTube',
    url: 'https://www.youtube.com/channel/UCFfALXX0DOx7zv6VeR5U_Bg',
    icon: 'youtube',
  },
  {
    name: 'Facebook',
    url: 'https://www.facebook.com/suanlab',
    icon: 'facebook',
  },
  {
    name: 'Instagram',
    url: 'https://www.instagram.com/suanlab',
    icon: 'instagram',
  },
  {
    name: 'LinkedIn',
    url: 'https://www.linkedin.com/in/suan-lee-46aaa15b/',
    icon: 'linkedin',
  },
  {
    name: 'Twitter',
    url: 'https://twitter.com/webdizen',
    icon: 'twitter',
  },
];

export const contactInfo: ContactInfo = {
  address: '65 Semyung-ro, Jecheon-si, Chungcheongbuk-do, 27136 Korea',
  phone: '+82-43-649-1273',
  emails: ['suanlab@gmail.com', 'suanlee@semyung.ac.kr'],
};

export const siteDescription = `SuanLab is professor Suan's personal website, and conducting research on data science and artificial intelligence and shares various information, lecture, and youtube contents.
Currently, I am an assistant professor in the School of Computer Science at Semyung University and teach artificial intelligence, image processing, big data, and database lecture.
Contact me if anyone or company wants to join or collaborate on an interesting data science & artificial intelligence research.`;
