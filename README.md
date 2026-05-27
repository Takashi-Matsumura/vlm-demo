# VLM Demo — Qwen2.5-VL × llama.cpp

Vision Language Model (VLM) を**ローカルで動かして触りながら学ぶ**ためのデモアプリ。画像 1 枚または動画 1 本を入力すると、Qwen2.5-VL がその内容を日本語で言語化します。

学習目的のリポジトリなので、コードだけでなく **VLM そのものへの理解** をこの README に整理しています。

---

## 1. このデモで体験できること

| モード | 入力 | 処理 | 出力 |
|---|---|---|---|
| 画像モード | 画像 1 枚 (JPEG / PNG / WebP / GIF) | そのまま VLM に渡す | 画像の日本語説明（ストリーミング） |
| 動画モード | 動画 1 本 (MP4 / MOV / WebM など) | ブラウザ内で等間隔に 2〜12 枚のフレームを抽出 → 複数枚の画像として一度に VLM に渡す | 動画全体の時系列を踏まえた日本語説明（ストリーミング） |

---

## 2. セットアップ

### 前提

- macOS (Apple Silicon 推奨)
- Node.js 22+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (`brew install llama.cpp`)

### llama-server を起動（初回は HuggingFace から数 GB の自動ダウンロード）

```sh
llama-server -hf ggml-org/Qwen2.5-VL-7B-Instruct-GGUF \
  --host 127.0.0.1 --port 8080 -c 8192
```

### Next.js を起動

```sh
cp .env.example .env.local
npm install
npm run dev
```

`http://localhost:3000` を開く。

---

## 3. アーキテクチャ

```
[Browser]
  ├─ 画像 / 動画ファイルを選択
  ├─ 動画モード: <video> + <canvas> で N 枚 JPEG を抽出
  └─ FormData で POST /api/vlm
        ↓
[Next.js Route Handler (app/api/vlm/route.ts)]
  ├─ zod でバリデーション (mime / size / prompt)
  ├─ 画像/動画モードを frames[] フィールドの有無で判定
  ├─ Vercel AI SDK v6 の streamText で OpenAI 互換 API 呼び出し
  └─ toTextStreamResponse() でクライアントにストリーム返却
        ↓
[llama-server (llama.cpp)]
  └─ Qwen2.5-VL (GGUF + mmproj) でマルチモーダル推論
```

### 主要ファイル

| ファイル | 役割 |
|---|---|
| `app/api/vlm/route.ts` | Route Handler。画像 1 枚 (`image`) と動画フレーム複数 (`frames[]`) の両方を扱う |
| `app/page.tsx` | クライアント UI。ファイル種別の自動判定、動画プレビュー、フレーム抽出、ストリーム表示 |
| `lib/llm.ts` | Vercel AI SDK の OpenAI 互換プロバイダ設定 |
| `lib/video.ts` | ブラウザ内で動画 → JPEG フレーム抽出 (HTMLVideoElement + Canvas) |
| `.env.example` | 環境変数テンプレート (`LLAMA_SERVER_URL`, `LLAMA_MODEL_ID`) |

---

## 4. VLM について学んだこと

### 4.1 VLM とは

> **画像を入力に取れる LLM。** ViT 系の視覚エンコーダで画像を埋め込み、投影層で LLM のトークン空間にマッピングしてから、通常どおり LLM がテキスト生成する構造。

「画像の説明」だけが用途ではなく、以下のような **画像を入力に取るあらゆるテキストタスク** に使えます。

- 視覚的質問応答 ("この標識は何を意味する?")
- OCR (画像内のテキスト抽出)
- 物体の位置や個数の特定、構図の分析
- 画像を根拠にした推論やコード生成

ベース LLM の言語能力を「画像を見ながら」そのまま発揮できるイメージ。

### 4.2 動画の言語化アプローチ — 2 通りある

| 方式 | 内容 | このデモ |
|---|---|---|
| **フレーム多枚送信** | クライアントで N 枚抽出 → 独立した画像として送る。モデルはフレーム間の時間情報を **プロンプト順序からのみ推測** | ✅ |
| **ネイティブ動画入力** | モデルが動画トークン列を受け取り、時間軸を含む位置エンコーディング (Qwen2.5-VL の **M-RoPE = 時間 / 縦 / 横の 3 軸**) で処理。空間トークン圧縮も効いて効率的 | ❌ |

GPT-4o の動画理解も内部的にはフレーム多枚送信に近い方式で、現実的かつ広く使われているアプローチ。

### 4.3 なぜ llama.cpp 経由ではフレーム多枚方式なのか

- llama.cpp の **mtmd**（マルチモーダルパイプライン）は動画コンテナを直接扱えない
- OpenAI 互換 API の `image_url` スキーマも動画を想定していない
- Qwen2.5-VL **モデル自体は** HuggingFace transformers / vLLM 等の公式実装でネイティブ動画を扱えるが、**llama.cpp / GGUF エコシステムにはまだ届いていない**

**LM Studio や Ollama も内部で llama.cpp を使う**ため、同じ制約を引き継ぐ。「ローカル + GGUF」を選ぶと現状ネイティブ動画は使えない、と覚えておくのが正確。

### 4.4 フレーム多枚方式のトレードオフ

- ✅ 実装がシンプル。画像対応の VLM ならどれでもそのまま動画にも使える
- ❌ 抽出間隔より短い動きや事象は欠落する（例: 35 秒動画 / 12 フレーム = 約 3 秒間隔）
- ❌ 各フレームが独立した画像トークン列になるため、ネイティブ方式の空間トークン圧縮が効かず**トークン消費が大きい**
- ❌ 時間情報は「並び順」しか伝わらず、本物の時間軸エンコーディングではない

UI のフレーム数スライダー (2〜12) はこのトレードオフを **体感で比較する** ためのもの。

### 4.5 ネイティブ動画入力を試したい場合の選択肢

| 手段 | 備考 |
|---|---|
| **HuggingFace transformers + `qwen-vl-utils`** | 公式 Python 実装。`{type: "video", video: "path.mp4", fps: 1.0}` で渡せる。最も柔軟 |
| **vLLM** | OpenAI 互換 API を `video_url` という独自 content part 型で拡張。サーバとして本格運用しやすい |
| **NVIDIA NIM** | コンテナ 1 つで OpenAI 互換 API を起動 |
| **Alibaba DashScope** | Qwen 本家のホスト型 API |
| **Google Gemini API** | Gemini は最初からネイティブ動画対応 (音声まで含む) |

GPU を持っているなら（例: NVIDIA DGX Spark の 128 GB 統合メモリ + Blackwell）vLLM で Qwen2.5-VL 7B〜72B をネイティブ動画モードで動かし、本デモを薄いフロントとして残せば **「llama.cpp の多枚画像方式 vs vLLM のネイティブ動画方式」を並べて比較する**学習プラットフォームになる。

---

## 5. 技術スタック

- **Next.js 16** (App Router, Turbopack)
- **React 19** / **TypeScript** (strict)
- **Tailwind CSS v4**
- **Vercel AI SDK v6** (`ai`, `@ai-sdk/openai-compatible`)
- **zod** (バリデーション)
- **llama.cpp** (`llama-server`, mtmd マルチモーダル)
- **Qwen2.5-VL-7B-Instruct** (GGUF + mmproj)

---

## 6. API リファレンス

### `POST /api/vlm`

`multipart/form-data` で受信。

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `prompt` | string | ✅ | 1〜2000 文字 |
| `image` | File | ※ | 画像モード用。1 枚 |
| `frames` | File[] | ※ | 動画モード用。最大 12 枚 |

`image` か `frames` のいずれかが必須。両方あれば `frames` 優先。

| 受け入れる mime | `image/jpeg`, `image/png`, `image/webp`, `image/gif` |
|---|---|
| 1 枚あたり上限 | 8 MB |

**レスポンス**: 成功時は `text/plain; charset=utf-8` のチャンクストリーム。失敗時は JSON (`400` / `413` / `415` / `503` / `500`)。

---

## 7. ライセンス / 注意

学習目的のサンプル。モデルファイルのライセンスは HuggingFace 上の各リポジトリに従う。
