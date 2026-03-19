"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Beaker, Gauge, Layers, ListTree, Radar, RefreshCcw, ShieldAlert, Telescope } from "lucide-react";

const links = [
  ["/", "Overview", Gauge],
  ["/dataset", "Dataset", Layers],
  ["/benchmarks", "Benchmarks", Telescope],
  ["/attacks", "Attacks", ShieldAlert],
  ["/retrain-strategy", "Retrain Strategy", RefreshCcw],
  ["/models", "Models", Radar],
  ["/instances", "Instances", ListTree],
  ["/experiments", "Experiments", Beaker],
];

export default function NavMenu() {
  const pathname = usePathname();
  return (
    <nav className="menu">
      {links.map(([href, label, Icon]) => {
        const active = pathname === href || (href !== "/" && pathname.startsWith(href));
        return (
          <Link key={href} href={href} className={active ? "active" : ""}>
            <Icon size={15} />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
