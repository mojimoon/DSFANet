import "./globals.css";
import { Suspense } from "react";
import NavMenu from "@/components/NavMenu";
import ContentViewport from "@/components/ContentViewport";

export const metadata = {
  title: "IDS Dashboard",
  description: "Multi-page dashboard for ensemble anomaly detection",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="layout">
          <aside className="sidebar">
            <h1>IDS Dashboard</h1>
            <NavMenu />
          </aside>
          <Suspense fallback={<main className="content">{children}</main>}>
            <ContentViewport>{children}</ContentViewport>
          </Suspense>
        </div>
      </body>
    </html>
  );
}
