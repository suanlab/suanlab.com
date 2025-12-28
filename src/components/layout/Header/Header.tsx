'use client';

import { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { navigation } from '@/data/navigation';
import { NavItem } from '@/types';

export default function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };

  const renderNavItem = (item: NavItem) => {
    const hasChildren = item.children && item.children.length > 0;

    return (
      <li key={item.href} className={hasChildren ? 'dropdown' : ''}>
        <Link
          href={item.href}
          className={hasChildren ? 'dropdown-toggle' : ''}
        >
          <span className="topMain-icon">
            <i className={item.icon}></i>
            {item.label}
          </span>
        </Link>
        {hasChildren && (
          <ul className="dropdown-menu">
            {item.children!.map((child) => (
              <li key={child.href}>
                <Link href={child.href}>
                  <i className={child.icon}></i> {child.label}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </li>
    );
  };

  return (
    <div id="header" className="sticky clearfix">
      <header id="topNav">
        <div className="container">
          {/* Mobile Menu Button */}
          <button
            className="btn btn-mobile"
            onClick={toggleMobileMenu}
            aria-label="Toggle navigation menu"
          >
            <i className="fa fa-bars"></i>
          </button>

          {/* Logo */}
          <Link href="/" className="logo pull-left">
            <Image
              src="/assets/images/logo.png"
              alt="SuanLab"
              width={180}
              height={50}
              priority
            />
          </Link>

          {/* Navigation */}
          <div
            className={`navbar-collapse pull-right nav-main-collapse ${
              mobileMenuOpen ? 'in' : 'collapse'
            }`}
          >
            <nav className="nav-main">
              <ul id="topMain" className="nav nav-pills nav-main">
                {navigation.map(renderNavItem)}
              </ul>
            </nav>
          </div>
        </div>
      </header>
    </div>
  );
}
