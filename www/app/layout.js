import "./globals.css";
import NavMenu from "@/components/NavMenu";

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
            <p className="subtle">Frontend: Next.js (pnpm)</p>
          </aside>
          <main className="content">{children}</main>
        </div>
      </body>
    </html>
  );
}
