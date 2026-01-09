const steps = [
  {
    title: "1. PDFアップロード",
    description:
      "ドラッグ&ドロップで教科書PDFをアップロード。ページサムネイルを確認します。",
  },
  {
    title: "2. 領域調整",
    description:
      "自動提案されたメイン本文・語彙の矩形をドラッグで調整します。",
  },
  {
    title: "3. OCR & エクスポート",
    description:
      "OCR結果を編集し、Word/Slides/Excelを出力します。",
  },
];

const outputs = [
  {
    label: "Wordハンドアウト",
    detail: "B4横・左側に語彙表と本文を配置",
  },
  {
    label: "Google Slides",
    detail: "1文/1枚、色・フォント固定",
  },
  {
    label: "Excel語彙表",
    detail: "用語・品詞・日本語の表形式",
  },
];

export default function HomePage() {
  return (
    <div className="stack">
      <section className="hero">
        <div>
          <p className="eyebrow">最短で授業準備を完了</p>
          <h1>PDFから自動で教材を生成</h1>
          <p className="lead">
            本文と語彙を検出し、OCR結果を確認してそのまま配布物やスライドを作成できます。
          </p>
        </div>
        <div className="upload-card">
          <p className="upload-title">PDFをアップロード</p>
          <p className="upload-subtitle">最大50ページまで / 300DPIで画像化</p>
          <button className="primary-button" type="button">
            ファイルを選択
          </button>
          <p className="upload-hint">またはここにドラッグ&ドロップ</p>
        </div>
      </section>

      <section className="card-grid">
        {steps.map((step) => (
          <div className="card" key={step.title}>
            <h3>{step.title}</h3>
            <p>{step.description}</p>
          </div>
        ))}
      </section>

      <section className="output-section">
        <h2>出力形式</h2>
        <div className="output-grid">
          {outputs.map((output) => (
            <div className="output-card" key={output.label}>
              <p className="output-label">{output.label}</p>
              <p className="output-detail">{output.detail}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
