export type ConferenceCategory =
  | 'AI'
  | 'ML'
  | 'NLP'
  | 'CV'
  | 'Data'
  | 'IR'
  | 'Robotics'
  | 'Speech'
  | 'Web'
  | 'Knowledge';

export interface ConferenceCategoryInfo {
  id: ConferenceCategory;
  label: string;
  color: string;
}

export interface ConferenceDeadline {
  type: string;
  date: string;
  note?: string;
}

export interface Conference {
  id: string;
  name: string;
  year: number;
  full_name: string;
  categories: ConferenceCategory[];
  location: string;
  url: string;
  timezone?: string;
  deadlines: ConferenceDeadline[];
}

export const conferenceCategories: ConferenceCategoryInfo[] = [
  { id: 'AI', label: 'AI', color: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300' },
  { id: 'ML', label: 'ML', color: 'bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300' },
  { id: 'NLP', label: 'NLP', color: 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300' },
  { id: 'CV', label: 'CV', color: 'bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300' },
  { id: 'Data', label: 'Data', color: 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300' },
  { id: 'IR', label: 'IR', color: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/40 dark:text-cyan-300' },
  { id: 'Robotics', label: 'Robotics', color: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300' },
  { id: 'Speech', label: 'Speech', color: 'bg-pink-100 text-pink-800 dark:bg-pink-900/40 dark:text-pink-300' },
  { id: 'Web', label: 'Web', color: 'bg-teal-100 text-teal-800 dark:bg-teal-900/40 dark:text-teal-300' },
  { id: 'Knowledge', label: 'Knowledge', color: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/40 dark:text-indigo-300' },
];

export const conferences: Conference[] = [
  {
    id: 'aaai-2027',
    name: 'AAAI',
    year: 2027,
    full_name: 'AAAI Conference on Artificial Intelligence',
    categories: ['AI', 'ML', 'NLP', 'CV', 'Knowledge'],
    location: 'Montréal, Canada',
    url: 'https://aaai.org/conference/aaai/aaai-27/',
    timezone: 'AoE',
    deadlines: [
      { type: 'Abstract', date: '2026-07-21' },
      { type: 'Full Paper', date: '2026-07-28' },
      { type: 'Supplementary/Code', date: '2026-07-31' },
      { type: 'Phase 1 Rejection Notice', date: '2026-09-24' },
      { type: 'Rebuttal Window Opens', date: '2026-10-19' },
      { type: 'Rebuttal Window Ends', date: '2026-10-25' },
      { type: 'Final Notification', date: '2026-11-30' },
      { type: 'Camera Ready', date: '2026-12-14' },
      { type: 'Conference', date: '2027-02-16', note: 'Feb 16-23, 2027' },
    ],
  },
  {
    id: 'wsdm-2027',
    name: 'WSDM',
    year: 2027,
    full_name: 'ACM International Conference on Web Search and Data Mining',
    categories: ['IR', 'Web', 'Data', 'ML'],
    location: 'Hong Kong, China',
    url: 'https://www.wsdm-conference.org/2027/',
    deadlines: [
      { type: 'Conference', date: '2027-02-15', note: 'Feb 15-19, 2027 (20th Anniversary)' },
    ],
  },
  {
    id: 'icml-2027',
    name: 'ICML',
    year: 2027,
    full_name: 'International Conference on Machine Learning',
    categories: ['ML', 'AI'],
    location: 'South America (city TBA)',
    url: 'https://icml.cc/',
    deadlines: [
      { type: 'Conference', date: '2027-07-01', note: 'Exact dates TBA (typically July)' },
    ],
  },
  {
    id: 'iclr-2027',
    name: 'ICLR',
    year: 2027,
    full_name: 'International Conference on Learning Representations',
    categories: ['ML', 'AI', 'CV', 'NLP'],
    location: 'TBA',
    url: 'https://iclr.cc/',
    deadlines: [
      { type: 'Conference', date: '2027-04-15', note: 'Exact dates TBA (typically April-May)' },
    ],
  },
  {
    id: 'acl-2027',
    name: 'ACL',
    year: 2027,
    full_name: 'Annual Meeting of the Association for Computational Linguistics',
    categories: ['NLP', 'ML'],
    location: 'Japan (city TBA)',
    url: 'https://www.aclweb.org/portal/',
    deadlines: [
      { type: 'Conference', date: '2027-07-01', note: 'Exact dates TBA (typically July)' },
    ],
  },
  {
    id: 'cvpr-2027',
    name: 'CVPR',
    year: 2027,
    full_name: 'IEEE/CVF Conference on Computer Vision and Pattern Recognition',
    categories: ['CV', 'ML', 'AI'],
    location: 'Seattle, USA',
    url: 'https://cvpr.thecvf.com/',
    deadlines: [
      { type: 'Conference', date: '2027-06-01', note: 'Exact dates TBA (typically June). GC: Deva Ramanan, A. Ross' },
    ],
  },
  {
    id: 'iccv-2027',
    name: 'ICCV',
    year: 2027,
    full_name: 'International Conference on Computer Vision',
    categories: ['CV', 'ML', 'AI'],
    location: 'Hong Kong',
    url: 'https://iccv.thecvf.com/',
    deadlines: [
      { type: 'Conference', date: '2027-10-15', note: 'Exact dates TBA (typically October). GC: Gang Hua, William Scheirer, Nuria Oliver' },
    ],
  },
  {
    id: 'icassp-2027',
    name: 'ICASSP',
    year: 2027,
    full_name: 'IEEE International Conference on Acoustics, Speech and Signal Processing',
    categories: ['Speech', 'ML', 'AI'],
    location: 'Toronto, Canada',
    url: 'https://2027.ieeeicassp.org/',
    timezone: 'AoE',
    deadlines: [
      { type: 'Full Paper', date: '2026-09-16' },
      { type: 'Conference', date: '2027-05-16', note: 'May 16-21, 2027' },
    ],
  },
  {
    id: 'icra-2027',
    name: 'ICRA',
    year: 2027,
    full_name: 'IEEE International Conference on Robotics and Automation',
    categories: ['Robotics'],
    location: 'Seoul, Korea (Coex)',
    url: 'https://2027.ieee-icra.org/',
    deadlines: [
      { type: 'Conference', date: '2027-05-24', note: 'May 24-28, 2027' },
    ],
  },
  {
    id: 'kdd-2027',
    name: 'KDD',
    year: 2027,
    full_name: 'ACM SIGKDD Conference on Knowledge Discovery and Data Mining',
    categories: ['Data', 'ML', 'AI'],
    location: 'San Jose, USA',
    url: 'https://kdd2027.kdd.org/',
    timezone: 'UTC',
    deadlines: [
      { type: 'Abstract (Cycle 1)', date: '2027-05-20' },
      { type: 'Full Paper (Cycle 1)', date: '2027-06-19' },
      { type: 'Conference', date: '2027-08-25', note: 'Aug 25-29, 2027 (approx)' },
    ],
  },
  {
    id: 'www-2027',
    name: 'WWW',
    year: 2027,
    full_name: 'The Web Conference',
    categories: ['Web', 'IR', 'Data'],
    location: 'Dublin, Ireland',
    url: 'https://thewebconf.org/',
    deadlines: [
      { type: 'Conference', date: '2027-04-01', note: 'Exact dates TBA' },
    ],
  },
  {
    id: 'coling-2027',
    name: 'COLING',
    year: 2027,
    full_name: 'International Conference on Computational Linguistics',
    categories: ['NLP'],
    location: 'Macau, China',
    url: 'https://2027.coling-iccl.org/',
    deadlines: [
      { type: 'Conference', date: '2027-05-09', note: 'May 9-14, 2027' },
    ],
  },
  {
    id: 'neurips-2026',
    name: 'NeurIPS',
    year: 2026,
    full_name: 'Conference on Neural Information Processing Systems',
    categories: ['ML', 'AI', 'NLP', 'CV'],
    location: 'Atlanta, USA',
    url: 'https://neurips.cc/Conferences/2026',
    timezone: 'AoE',
    deadlines: [
      { type: 'Abstract', date: '2026-05-04' },
      { type: 'Full Paper', date: '2026-05-06' },
      { type: 'Discussion Opens', date: '2026-07-27' },
      { type: 'Discussion Ends', date: '2026-08-10' },
      { type: 'Notification', date: '2026-09-24' },
      { type: 'Conference', date: '2026-12-06', note: 'Dec 6-12, 2026 (40th NeurIPS)' },
    ],
  },
  {
    id: 'emnlp-2026',
    name: 'EMNLP',
    year: 2026,
    full_name: 'Conference on Empirical Methods in Natural Language Processing',
    categories: ['NLP', 'ML'],
    location: 'Budapest, Hungary',
    url: 'https://2026.emnlp.org/',
    timezone: 'AoE',
    deadlines: [
      { type: 'ARR Submission', date: '2026-05-25' },
      { type: 'Author Response', date: '2026-07-07', note: 'Jul 7-13' },
      { type: 'Commitment Deadline', date: '2026-08-02' },
      { type: 'Notification', date: '2026-08-20' },
      { type: 'Camera Ready', date: '2026-08-30' },
      { type: 'Conference', date: '2026-10-24', note: 'Oct 24-29, 2026' },
    ],
  },
  {
    id: 'colm-2026',
    name: 'COLM',
    year: 2026,
    full_name: 'Conference on Language Modeling',
    categories: ['NLP', 'ML', 'AI'],
    location: 'San Francisco, USA',
    url: 'https://colmweb.org/',
    timezone: 'AoE',
    deadlines: [
      { type: 'Abstract', date: '2026-03-26' },
      { type: 'Full Paper', date: '2026-03-31' },
      { type: 'Rebuttal Period', date: '2026-05-22', note: 'May 22 - Jun 8' },
      { type: 'Notification', date: '2026-07-08' },
      { type: 'Conference', date: '2026-10-06', note: 'Oct 6-9, 2026' },
    ],
  },
  {
    id: 'eccv-2026',
    name: 'ECCV',
    year: 2026,
    full_name: 'European Conference on Computer Vision',
    categories: ['CV', 'ML'],
    location: 'Malmö, Sweden',
    url: 'https://eccv.ecva.net/Conferences/2026',
    timezone: 'UTC',
    deadlines: [
      { type: 'Paper Registration (Abstract)', date: '2026-02-26' },
      { type: 'Full Paper', date: '2026-03-05' },
      { type: 'Supplemental Materials', date: '2026-03-12' },
      { type: 'Rebuttal Deadline', date: '2026-05-11' },
      { type: 'Camera Ready', date: '2026-06-27' },
      { type: 'Conference', date: '2026-09-08', note: 'Sep 8-12, 2026 (36th ECCV)' },
    ],
  },
  {
    id: 'cikm-2026',
    name: 'CIKM',
    year: 2026,
    full_name: 'ACM International Conference on Information and Knowledge Management',
    categories: ['IR', 'Data', 'Knowledge'],
    location: 'Rome, Italy',
    url: 'https://cikm2026.diag.uniroma1.it/',
    timezone: 'AoE',
    deadlines: [
      { type: 'Full Paper Abstract', date: '2026-05-16' },
      { type: 'Full Paper Submission', date: '2026-05-23' },
      { type: 'Short Paper Abstract', date: '2026-05-30' },
      { type: 'Short Paper Submission', date: '2026-06-06' },
      { type: 'Notification', date: '2026-08-07' },
      { type: 'Conference', date: '2026-11-07', note: 'Nov 7-11, 2026' },
    ],
  },
  {
    id: 'icdm-2026',
    name: 'ICDM',
    year: 2026,
    full_name: 'IEEE International Conference on Data Mining',
    categories: ['Data', 'ML', 'AI'],
    location: 'Shenyang, China',
    url: 'https://icdm2026.neu.edu.cn/',
    timezone: 'AoE',
    deadlines: [
      { type: 'Full Paper', date: '2026-06-06' },
      { type: 'Notification', date: '2026-08-16' },
      { type: 'Conference', date: '2026-11-12', note: 'Nov 12-15, 2026 (26th IEEE ICDM)' },
    ],
  },
  {
    id: 'recsys-2026',
    name: 'RecSys',
    year: 2026,
    full_name: 'ACM Conference on Recommender Systems',
    categories: ['IR', 'ML', 'Web'],
    location: 'Minneapolis, USA',
    url: 'https://recsys.acm.org/recsys26/',
    timezone: 'AoE',
    deadlines: [
      { type: 'Abstract', date: '2026-04-28' },
      { type: 'Full Paper', date: '2026-05-05' },
      { type: 'Notification', date: '2026-07-09' },
      { type: 'Camera Ready', date: '2026-07-27' },
      { type: 'Conference', date: '2026-09-27', note: 'Sep 27 - Oct 2, 2026 (20th RecSys)' },
    ],
  },
  {
    id: 'corl-2026',
    name: 'CoRL',
    year: 2026,
    full_name: 'Conference on Robot Learning',
    categories: ['Robotics', 'ML', 'AI'],
    location: 'Austin, USA',
    url: 'https://www.corl.org/',
    timezone: 'UTC',
    deadlines: [
      { type: 'Abstract', date: '2026-05-25' },
      { type: 'Full Paper', date: '2026-05-28' },
      { type: 'Conference', date: '2026-11-10', note: 'Nov 10-12, 2026 (main)' },
    ],
  },
  {
    id: 'interspeech-2026',
    name: 'INTERSPEECH',
    year: 2026,
    full_name: 'INTERSPEECH Conference',
    categories: ['Speech', 'ML'],
    location: 'Sydney, Australia',
    url: 'https://interspeech2026.org/',
    timezone: 'AoE',
    deadlines: [
      { type: 'Camera Ready', date: '2026-06-19' },
      { type: 'Early-bird Registration', date: '2026-07-15' },
      { type: 'Conference', date: '2026-09-28', note: 'Sep 28 - Oct 1, 2026 (Theme: Speaking Together)' },
    ],
  },
  {
    id: 'ismir-2026',
    name: 'ISMIR',
    year: 2026,
    full_name: 'International Society for Music Information Retrieval Conference',
    categories: ['Speech', 'ML', 'AI'],
    location: 'Abu Dhabi, UAE',
    url: 'https://ismir2026.ismir.net/',
    timezone: 'AoE',
    deadlines: [
      { type: 'Abstract', date: '2026-04-20' },
      { type: 'Full Paper', date: '2026-04-27' },
      { type: 'Conference', date: '2026-11-08', note: 'Nov 8-12, 2026 (27th ISMIR)' },
    ],
  },
  {
    id: 'ecmlpkdd-2026',
    name: 'ECML PKDD',
    year: 2026,
    full_name: 'European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases',
    categories: ['ML', 'Data', 'Knowledge'],
    location: 'Naples, Italy',
    url: 'https://ecmlpkdd.org/2026/',
    timezone: 'AoE',
    deadlines: [
      { type: 'Abstract', date: '2026-03-05' },
      { type: 'Full Paper', date: '2026-03-12' },
      { type: 'Notification', date: '2026-07-15' },
      { type: 'Conference', date: '2026-09-07', note: 'Sep 7-11, 2026' },
    ],
  },
  {
    id: 'acmmm-2026',
    name: 'ACM MM',
    year: 2026,
    full_name: 'ACM Multimedia Conference',
    categories: ['CV', 'ML', 'Web', 'AI'],
    location: 'Rio de Janeiro, Brazil',
    url: 'https://2026.acmmm.org/',
    timezone: 'AoE',
    deadlines: [
      { type: 'Abstract', date: '2026-03-25' },
      { type: 'Full Paper', date: '2026-04-01' },
      { type: 'Supplementary', date: '2026-04-08' },
      { type: 'Notification', date: '2026-07-07' },
      { type: 'Camera Ready', date: '2026-08-06' },
      { type: 'Conference', date: '2026-11-10', note: 'Nov 10-14, 2026 (34th ACM MM)' },
    ],
  },
  {
    id: 'lrec-2026',
    name: 'LREC',
    year: 2026,
    full_name: 'International Conference on Language Resources and Evaluation',
    categories: ['NLP'],
    location: 'Palma de Mallorca, Spain',
    url: 'https://lrec2026.info/',
    timezone: 'AoE',
    deadlines: [
      { type: 'Paper Submission', date: '2025-10-17' },
      { type: 'Workshop Notification', date: '2025-11-17' },
      { type: 'Conference', date: '2026-05-11', note: 'May 11-16, 2026 (15th LREC)' },
    ],
  },
];
