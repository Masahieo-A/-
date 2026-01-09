import "./globals.css";

export const metadata = {
  title: "Lesson Prep Agent",
  description: "教科書PDFから配布物・スライド・語彙表を自動生成",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja">
      <body>
        <div className="app-shell">
          <header className="header">
            <div>
              <p className="app-title">Lesson Prep Agent</p>
              <p className="app-subtitle">
                教科書PDFをアップロードして配布物・スライド・語彙表を生成
              </p>
            </div>
            <button className="ghost-button" type="button">
              スタイル設定
            </button>
          </header>
          <main className="content">{children}</main>
        </div>
      </body>
    </html>
  );
}
