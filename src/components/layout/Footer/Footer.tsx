import Image from 'next/image';
import Link from 'next/link';
import { socialLinks, contactInfo, siteDescription } from '@/data/navigation';

export default function Footer() {
  return (
    <footer id="footer">
      <div className="container">
        <div className="row">
          {/* Left Column: Logo & Social */}
          <div className="col-md-5">
            <Image
              src="/assets/images/logo-footer.png"
              alt="SuanLab"
              width={180}
              height={50}
              className="footer-logo footer-2"
            />

            <div className="clearfix">
              {socialLinks.map((social) => (
                <a
                  key={social.name}
                  href={social.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`social-icon social-icon-sm social-icon-transparent social-${social.icon} pull-left`}
                  title={social.name}
                >
                  <i className={`icon-${social.icon}`}></i>
                  <i className={`icon-${social.icon}`}></i>
                </a>
              ))}
            </div>

            <p>{siteDescription}</p>
          </div>

          {/* Middle Column: Contact Info */}
          <div className="col-md-3">
            <h3>CONTACT INFO</h3>

            <address>
              <ul className="list-unstyled">
                <li className="footer-sprite address">{contactInfo.address}</li>
                <li className="footer-sprite phone">Phone: {contactInfo.phone}</li>
                <li className="footer-sprite email">
                  {contactInfo.emails.map((email, index) => (
                    <span key={email}>
                      <a href={`mailto:${email}`}>{email}</a>
                      {index < contactInfo.emails.length - 1 && <br />}
                    </span>
                  ))}
                </li>
              </ul>
            </address>
          </div>

          {/* Right Column: Map */}
          <div className="col-md-4">
            <a
              href="https://map.kakao.com/?urlX=765405&urlY=1024145&urlLevel=4&map_type=TYPE_SKYVIEW&map_hybrid=true"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Image
                src="https://map2.daum.net/map/skyviewmapservice?FORMAT=PNG&RDR=HybridRender&SCALE=5&MX=765405&MY=1024145&S=0&IW=504&IH=310&LANG=0&COORDSTM=WCONGNAMUL&logo=kakao_logo"
                alt="Location Map"
                width={360}
                height={204}
                style={{ border: '1px solid #ccc' }}
                unoptimized
              />
            </a>
          </div>
        </div>
      </div>

      {/* Copyright */}
      <div className="copyright">
        <div className="container">
          <ul className="pull-right nomargin list-inline mobile-block">
            <li>
              <Link href="#">Terms &amp; Conditions</Link>
            </li>
            <li>&bull;</li>
            <li>
              <Link href="#">Privacy</Link>
            </li>
          </ul>
          &copy; All Rights Reserved, SuanLab
        </div>
      </div>
    </footer>
  );
}
