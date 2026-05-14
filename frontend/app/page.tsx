"use client";

import { useMemo, useState } from "react";

type GrammarFocus = {
  id: string;
  label: string;
  description: string;
  detector: (sentence: string) => boolean;
  blanker: (sentence: string) => ExerciseDraft | null;
};

type ExerciseDraft = {
  japanese: string;
  cloze: string;
  answer: string;
  tip: string;
  explanation: string;
  wordsToUse: string;
};

const fewShotExamples = [
  {
    focus: "現在形：習慣・不変の事実",
    japanese: "父はよくネットで買い物をする。",
    cloze: "My father often ( )( ).",
    answer: "shops / online",
    tip: "「現在の習慣」を表すとき、時制はどうする? 「買い物をする」を1語の自動詞で表すと?",
  },
  {
    focus: "現在進行形：今まさに行われている動作",
    japanese: "その女性は花に水をやっている。",
    cloze: "The woman ( )( ) some flowers.",
    answer: "is / watering",
    tip: "「現在行っている動作」を表す形は? 「〜に水をやる」を表す他動詞は?",
  },
  {
    focus: "過去形：過去の一時点・過去を示す語句",
    japanese: "おととしの夏、家族で富士山に登った。",
    cloze: "I ( ) Mt. Fuji with my family two ( ) ago.",
    answer: "climbed / summers",
    tip: "「過去を示す表現」と共に使われる時制は? 日本語を英語らしい語順に置き換えると?",
  },
  {
    focus: "未来表現：予想される未来",
    japanese: "自動運転車は交通をより安全にするだろう。",
    cloze: "Self-driving cars ( )( ) transportation safer.",
    answer: "will / make",
    tip: "「〜するだろう」と未来に起こると予想される事柄を表す一般的な表現は?",
  },
];

const sampleInput =
  "She has lived in Kyoto for three years.\nIf it rains tomorrow, we will stay home.\nI am studying English now.";

const tokenPatterns = {
  presentPerfect: /\b(has|have)\s+([a-z]+(?:ed|en)|been|done|gone|seen|written|known|lived|studied|visited|worked)\b/i,
  conditional: /\bIf\s+([^,.]+),\s*([^.!?]+)/i,
  progressive: /\b(am|are|is)\s+([a-z]+ing)\b/i,
  future: /\b(will)\s+([a-z]+)\b/i,
  passive: /\b(am|are|is|was|were|be|been)\s+([a-z]+ed|made|known|seen|written|built|given|called)\b/i,
  infinitive: /\b(to)\s+([a-z]+)\b/i,
};

const grammarFocuses: GrammarFocus[] = [
  {
    id: "conditional",
    label: "時・条件の副詞節",
    description: "if / when 節では、未来の内容でも現在形を使う点に注目します。",
    detector: (sentence) => tokenPatterns.conditional.test(sentence),
    blanker: (sentence) => {
      const match = sentence.match(tokenPatterns.conditional);
      if (!match) return null;
      return {
        japanese: "もし明日雨が降れば、私たちは家にいるだろう。",
        cloze: sentence.replace(/\bIf\b/i, "( )").replace(/\bwill\b/i, "( )"),
        answer: "If / will",
        tip: "条件を表すまとまりはどこまで? 主節で未来の内容を表す助動詞は?",
        explanation:
          "条件を表す if 節では、未来の内容でも現在形を使う。主節では will do を用いて、未来に起こると考えられることを表す。",
        wordsToUse: "if / will / present tense / main clause",
      };
    },
  },
  {
    id: "present-perfect",
    label: "現在完了形",
    description: "過去から現在へのつながりを has/have done で考えます。",
    detector: (sentence) => tokenPatterns.presentPerfect.test(sentence),
    blanker: (sentence) => {
      const match = sentence.match(tokenPatterns.presentPerfect);
      if (!match) return null;
      const auxiliary = match[1];
      const participle = match[2];
      return {
        japanese: "過去に始まった状態・経験・完了が、現在とどうつながっているかを考える英文。",
        cloze: sentence.replace(match[0], `( ) ( )`),
        answer: `${auxiliary} / ${participle}`,
        tip: "過去の一点だけを述べている? それとも現在とのつながりを述べている? 主語に合わせる助動詞は?",
        explanation:
          "現在完了形は has/have + 過去分詞で表す。for や since など期間を表す語句があるときは、過去から現在まで続く状態として考える。",
        wordsToUse: "has / have / past participle / for / since",
      };
    },
  },
  {
    id: "progressive",
    label: "進行形",
    description: "am/are/is doing で、今まさに進行中の動作を表します。",
    detector: (sentence) => tokenPatterns.progressive.test(sentence),
    blanker: (sentence) => {
      const match = sentence.match(tokenPatterns.progressive);
      if (!match) return null;
      return {
        japanese: "今まさに行われている動作を表す英文。",
        cloze: sentence.replace(match[0], "( ) ( )"),
        answer: `${match[1]} / ${match[2]}`,
        tip: "「今しているところ」を表す形は? be 動詞は主語に合わせてどう変える?",
        explanation:
          "現在進行形は am/are/is doing の形で、今まさに行われている動作や一時的な状況を表す。",
        wordsToUse: "am / are / is / doing / now",
      };
    },
  },
  {
    id: "future",
    label: "未来表現",
    description: "will do を使い、これから起こると考えられることを表します。",
    detector: (sentence) => tokenPatterns.future.test(sentence),
    blanker: (sentence) => {
      const match = sentence.match(tokenPatterns.future);
      if (!match) return null;
      return {
        japanese: "未来に起こると予想される事柄を表す英文。",
        cloze: sentence.replace(match[0], "( ) ( )"),
        answer: `${match[1]} / ${match[2]}`,
        tip: "「〜するだろう」と未来の予想を表す助動詞は? 助動詞の後ろの動詞の形は?",
        explanation:
          "will do は未来に起こると予想される事柄を表す一般的な形。助動詞 will の後ろには動詞の原形を置く。",
        wordsToUse: "will / base verb / tomorrow / next",
      };
    },
  },
  {
    id: "passive",
    label: "受動態",
    description: "be done で「〜される」を作るときの be 動詞と過去分詞に注目します。",
    detector: (sentence) => tokenPatterns.passive.test(sentence),
    blanker: (sentence) => {
      const match = sentence.match(tokenPatterns.passive);
      if (!match) return null;
      return {
        japanese: "主語が動作を受ける側になっている英文。",
        cloze: sentence.replace(match[0], "( ) ( )"),
        answer: `${match[1]} / ${match[2]}`,
        tip: "主語は動作をする側? される側? 「〜される」を表す基本の形は?",
        explanation:
          "受動態は be 動詞 + 過去分詞で表す。be 動詞は主語と時制に合わせて、過去分詞は動詞ごとの形を確認する。",
        wordsToUse: "be / past participle / by",
      };
    },
  },
  {
    id: "infinitive",
    label: "不定詞",
    description: "to do が目的・原因・名詞的用法など、文中で何の働きをするかを考えます。",
    detector: (sentence) => tokenPatterns.infinitive.test(sentence),
    blanker: (sentence) => {
      const match = sentence.match(tokenPatterns.infinitive);
      if (!match) return null;
      return {
        japanese: "to do のまとまりが文中で働いている英文。",
        cloze: sentence.replace(match[0], "( ) ( )"),
        answer: `${match[1]} / ${match[2]}`,
        tip: "to の後ろに続く動詞の形は? to do のまとまりは文の中で何の役割をしている?",
        explanation:
          "不定詞は to + 動詞の原形で表す。名詞・形容詞・副詞のように働き、文脈によって意味を判断する。",
        wordsToUse: "to / base verb / purpose / adjective use",
      };
    },
  },
];

function splitIntoSentences(input: string) {
  return input
    .split(/(?<=[.!?])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function fallbackExercise(sentence: string): ExerciseDraft {
  const words = sentence.match(/[A-Za-z']+|[^A-Za-z']+/g) ?? [sentence];
  const wordIndex = words.findIndex((word) => /^[A-Za-z']+$/.test(word) && word.length > 4);
  const targetIndex = wordIndex >= 0 ? wordIndex : words.findIndex((word) => /^[A-Za-z']+$/.test(word));
  const answer = targetIndex >= 0 ? words[targetIndex] : sentence;
  const cloze = targetIndex >= 0 ? words.map((word, index) => (index === targetIndex ? "( )" : word)).join("") : "( )";

  return {
    japanese: "入力英文の文構造を確認する問題。",
    cloze,
    answer,
    tip: "空欄の語は、文の中で主語・動詞・目的語・修飾語のどの役割をしている? 前後の語とのつながりは?",
    explanation:
      "英文を語順だけで暗記せず、空欄の前後にある語句との関係から必要な品詞と形を判断する。",
    wordsToUse: "subject / verb / object / modifier",
  };
}

function makeExercise(sentence: string) {
  const focus = grammarFocuses.find((item) => item.detector(sentence));
  const draft = focus?.blanker(sentence) ?? fallbackExercise(sentence);

  return {
    sentence,
    focus: focus?.label ?? "文構造・語法",
    description: focus?.description ?? "前後関係から必要な語・形・意味を考えます。",
    ...draft,
  };
}

function makeFewShotPrompt(input: string) {
  const examples = fewShotExamples
    .map(
      (example, index) =>
        `例${index + 1}\n文法事項: ${example.focus}\n日本語: ${example.japanese}\n英文（穴埋め）: ${example.cloze}\n答え: ${example.answer}\nTip（思考誘導）: ${example.tip}`,
    )
    .join("\n\n");

  return `あなたは高校英文法教材の作問者です。以下の例のトーンに合わせ、英文を空欄補充問題にします。\n\n${examples}\n\n制約:\n- 答えを直接教えすぎず、「何に注目するか」を疑問文で誘導する。\n- 空欄は文法判断が必要な語句に置く。\n- 解説は、なぜその形になるかを簡潔に述べる。\n\n入力英文:\n${input}`;
}

export default function HomePage() {
  const [input, setInput] = useState(sampleInput);
  const exercises = useMemo(() => splitIntoSentences(input).map(makeExercise), [input]);
  const prompt = useMemo(() => makeFewShotPrompt(input), [input]);

  return (
    <div className="stack">
      <section className="hero insight-hero">
        <div>
          <p className="eyebrow">Insight-style grammar generator</p>
          <h1>英文から「思考のヒント」付き空欄補充問題を生成</h1>
          <p className="lead">
            Google スプレッドシートで確認した文字起こしデータのトーンを few-shot prompt として整理し、入力英文から疑似 Vision Quest Insight 風の問題を作ります。
          </p>
        </div>
        <div className="upload-card prompt-card">
          <p className="upload-title">生成ポリシー</p>
          <p className="upload-subtitle">
            正解暗記ではなく、「どの文法判断をすればよいか」を問いかける Tip を中心に出力します。
          </p>
          <span className="pill">few-shot</span>
          <span className="pill">cloze</span>
          <span className="pill">thinking hint</span>
        </div>
      </section>

      <section className="workspace-grid">
        <div className="panel">
          <div className="section-heading">
            <p className="eyebrow">Input</p>
            <h2>英文を入力</h2>
          </div>
          <textarea
            className="sentence-input"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            rows={9}
            aria-label="問題化したい英文"
          />
          <p className="helper-text">
            1文ずつ改行、またはピリオド・疑問符・感嘆符で区切ると複数問を生成できます。
          </p>
        </div>

        <div className="panel prompt-preview">
          <div className="section-heading">
            <p className="eyebrow">Few-shot prompt</p>
            <h2>参照トーン</h2>
          </div>
          <pre>{prompt}</pre>
        </div>
      </section>

      <section className="output-section" id="generated-exercises">
        <div className="section-heading">
          <p className="eyebrow">Generated exercises</p>
          <h2>疑似 Insight 問題</h2>
        </div>
        <div className="exercise-list">
          {exercises.map((exercise, index) => (
            <article className="exercise-card" key={`${exercise.sentence}-${index}`}>
              <div className="exercise-header">
                <span className="question-number">Q{index + 1}</span>
                <div>
                  <h3>{exercise.focus}</h3>
                  <p>{exercise.description}</p>
                </div>
              </div>
              <dl className="exercise-detail">
                <div>
                  <dt>日本語</dt>
                  <dd>{exercise.japanese}</dd>
                </div>
                <div>
                  <dt>英文（穴埋め）</dt>
                  <dd className="cloze">{exercise.cloze}</dd>
                </div>
                <div>
                  <dt>答え</dt>
                  <dd>{exercise.answer}</dd>
                </div>
                <div>
                  <dt>Tip（思考誘導）</dt>
                  <dd>{exercise.tip}</dd>
                </div>
                <div>
                  <dt>解説</dt>
                  <dd>{exercise.explanation}</dd>
                </div>
                <div>
                  <dt>Words to Use</dt>
                  <dd>{exercise.wordsToUse}</dd>
                </div>
              </dl>
            </article>
          ))}
        </div>
      </section>

      <section className="card-grid">
        {fewShotExamples.map((example) => (
          <div className="card" key={example.focus}>
            <h3>{example.focus}</h3>
            <p className="mini-label">英文（穴埋め）</p>
            <p>{example.cloze}</p>
            <p className="mini-label">Tip</p>
            <p>{example.tip}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
