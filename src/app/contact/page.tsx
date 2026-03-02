import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { Mail, Phone, MapPin } from 'lucide-react';
import ContactForm from './contact-form';

export const metadata = {
  title: '문의하기 | SuanLab',
  description: 'SuanLab에 문의하기',
};

const contactInfo = {
  address: '65 Semyung-ro, Jecheon-si, Chungcheongbuk-do, 27136 Korea',
  phone: '+82-43-649-1273',
  emails: ['suanlab@gmail.com', 'suanlee@semyung.ac.kr'],
};

export default function ContactPage() {
  return (
    <>
      <PageHeader
        title="문의하기"
        subtitle="SuanLab에 문의하세요"
        breadcrumbs={[{ label: '문의하기' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="grid gap-12 lg:grid-cols-3">
            {/* Contact Info Sidebar */}
            <aside className="lg:col-span-1">
              <div className="sticky top-24 space-y-6">
                {/* Email */}
                <Card>
                  <CardContent className="p-6">
                    <div className="flex gap-4">
                      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                        <Mail className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold mb-2">Email</h3>
                        <div className="space-y-1">
                          {contactInfo.emails.map((email) => (
                            <a
                              key={email}
                              href={`mailto:${email}`}
                              className="block text-sm text-muted-foreground hover:text-primary transition-colors"
                            >
                              {email}
                            </a>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Phone */}
                <Card>
                  <CardContent className="p-6">
                    <div className="flex gap-4">
                      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                        <Phone className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold mb-2">Phone</h3>
                        <a
                          href={`tel:${contactInfo.phone.replace(/[^\d+]/g, '')}`}
                          className="text-sm text-muted-foreground hover:text-primary transition-colors"
                        >
                          {contactInfo.phone}
                        </a>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Address */}
                <Card>
                  <CardContent className="p-6">
                    <div className="flex gap-4">
                      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                        <MapPin className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold mb-2">Address</h3>
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {contactInfo.address}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </aside>

            {/* Contact Form */}
            <div className="lg:col-span-2">
              <ContactForm />
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
