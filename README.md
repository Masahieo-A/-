# Insight-Style Grammar Maker

英文を入力すると、空欄補充・答え・Tip（思考誘導）・解説を生成する Next.js アプリです。

Google スプレッドシートで共有された文字起こしデータの列構成（日本語、英文（穴埋め）、答え、Tip、解説、Words to Use）と、Tip が「何に注目すべきか」を疑問文で誘導するトーンを参考にしています。

## Features

- 複数英文をまとめて入力し、1文ごとに問題化
- 現在完了形、進行形、未来表現、受動態、不定詞、条件節を簡易検出
- few-shot prompt プレビューを画面内に表示
- ルールで検出できない英文も、文構造・語法問題としてフォールバック生成

## Frontend

```bash
cd frontend
npm install
npm run build
npm run dev
```

## Backend

既存の FastAPI OCR API は `backend/app/main.py` に残しています。PDF OCR や Word/Excel エクスポート用の API として利用できます。

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
