"use client";

import { usePathname, useSearchParams } from "next/navigation";

export default function ContentViewport({ children }) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const dataset = searchParams.get("dataset") || "";

  // Remount main content when dataset changes so client-page effects run again.
  const contentKey = `${pathname}::${dataset}`;

  return (
    <main key={contentKey} className="content">
      {children}
    </main>
  );
}
