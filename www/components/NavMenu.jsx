"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  ["/", "Overview"],
  ["/dataset", "Dataset"],
  ["/benchmarks", "Benchmarks"],
  ["/attacks", "Attacks"],
  ["/models", "Models"],
  ["/instances", "Instances"],
];

export default function NavMenu() {
  const pathname = usePathname();
  return (
    <nav className="menu">
      {links.map(([href, label]) => {
        const active = pathname === href || (href !== "/" && pathname.startsWith(href));
        return (
          <Link key={href} href={href} className={active ? "active" : ""}>
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
