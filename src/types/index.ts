export interface NavItem {
  label: string;
  href: string;
  icon: string;
  children?: NavItem[];
}

export interface SocialLink {
  name: string;
  url: string;
  icon: string;
}

export interface ContactInfo {
  address: string;
  phone: string;
  emails: string[];
}
