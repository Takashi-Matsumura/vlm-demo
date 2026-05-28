# VLM Demo — Qwen2.5-VL

Vision Language Model (VLM) を**ローカルで動かして、動画理解の仕組みを体感しながら学ぶ**デモアプリ。

画像 1 枚または動画 1 本を入力すると Qwen2.5-VL が日本語で説明します。  
**バックエンドを 4 種類から選択**でき、特に「フレーム多枚画像方式 vs ネイティブ動画入力 (M-RoPE)」の違いを並列比較できます。

---

## バックエンドモード一覧

| モード | モデル | 動画の扱い | 特徴 |
|---|---|---|---|
| **llama.cpp 7B** | Qwen2.5-VL-7B-Q4_K_M | ブラウザでフレーム抽出 → 複数画像 | 高速・軽量 |
| **llama.cpp 72B** | Qwen2.5-VL-72B-Q4_K_M | ブラウザでフレーム抽出 → 複数画像 | 高精度・低速 |
| **mlx-vlm 72B** | Qwen2.5-VL-7B-8bit *(or 72B-bf16)* | 動画ファイルをそのまま送信 | M-RoPE ネイティブ動画 |
| **並列比較** | llama.cpp + mlx-vlm を同時実行 | 両方式を同時実行 | TTFT・出力の差を比較 |

---

## セットアップ

### 前提条件

- macOS (Apple Silicon)
- Node.js 22+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (`brew install llama.cpp`)
- [uv](https://docs.astral.sh/uv/) (mlx-vlm サービス用)
- [hf CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`pip install huggingface_hub`)

---

### 1. モデルのダウンロード

#### Qwen2.5-VL-7B (llama.cpp 用 GGUF)

```bash
hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF \
  Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf
```

#### Qwen2.5-VL-72B (llama.cpp 用 GGUF)

```bash
hf download ggml-org/Qwen2.5-VL-72B-Instruct-GGUF \
  Qwen2.5-VL-72B-Instruct-Q4_K_M.gguf \
  mmproj-Qwen2.5-VL-72B-Instruct-f16.gguf
```

> 72B は約 47 GB + 1.4 GB。ダウンロードに数十分かかる場合があります。

#### Qwen2.5-VL-7B (mlx-vlm 用、ネイティブ動画確認用)

```bash
hf download mlx-community/Qwen2.5-VL-7B-Instruct-8bit
```

#### Qwen2.5-VL-72B-bf16 (mlx-vlm 用、本番用)

```bash
hf download mlx-community/Qwen2.5-VL-72B-Instruct-bf16
```

> 約 145 GB。Mac Studio M3 Ultra (512 GB) での動作を確認。

---

### 2. llama-server を起動

ダウンロードされたモデルは `~/.cache/huggingface/hub/` に保存されます。

```bash
# 変数にスナップショットパスをセット
SNAP_7B=~/.cache/huggingface/hub/models--ggml-org--Qwen2.5-VL-7B-Instruct-GGUF/snapshots/<HASH>
SNAP_72B=~/.cache/huggingface/hub/models--ggml-org--Qwen2.5-VL-72B-Instruct-GGUF/snapshots/<HASH>

# 7B (port 8082)
llama-server \
  --model    "$SNAP_7B/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" \
  --mmproj   "$SNAP_7B/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" \
  --host 127.0.0.1 --port 8082 \
  --ctx-size 8192 --n-gpu-layers 99 --flash-attn on \
  --alias qwen2.5-vl-7b

# 72B (port 8083)
llama-server \
  --model    "$SNAP_72B/Qwen2.5-VL-72B-Instruct-Q4_K_M.gguf" \
  --mmproj   "$SNAP_72B/mmproj-Qwen2.5-VL-72B-Instruct-f16.gguf" \
  --host 127.0.0.1 --port 8083 \
  --ctx-size 8192 --n-gpu-layers 99 --flash-attn on \
  --alias qwen2.5-vl-72b
```

---

### 3. mlx-vlm サービスを起動

```bash
cd services/mlx-vlm-server

# uv がない場合はインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# Python 3.12 環境を作成 (mlx は Python 3.14 未対応)
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install mlx-vlm fastapi "uvicorn[standard]" python-multipart

# 起動 (7B で疎通確認)
MLX_MODEL_ID=mlx-community/Qwen2.5-VL-7B-Instruct-8bit \
  uvicorn main:app --host 127.0.0.1 --port 8001

# 起動 (72B 本番)
uvicorn main:app --host 127.0.0.1 --port 8001
```

"Application startup complete." が出たらモデルロード完了。

---

### 4. Next.js を起動

```bash
cp .env.example .env.local
# .env.local を編集してポート番号を確認
npm install
npm run dev
```

`http://localhost:3000` を開く。

---

## アーキテクチャ

```
[Browser]
  ├─ 画像 / 動画ファイルを選択
  ├─ フレーム多枚方式: <video> + <canvas> で N 枚 JPEG を抽出
  └─ FormData で POST /api/vlm (mode フィールドでバックエンド指定)
        ↓
[Next.js Route Handler  app/api/vlm/route.ts]
  ├─ mode=llama_cpp      → llama-server 7B  (port 8082)
  ├─ mode=llama_cpp_72b  → llama-server 72B (port 8083)
  ├─ mode=mlx_native     → mlx-vlm-server   (port 8001) ← 動画ファイルをプロキシ
  └─ mode=parallel       → llama + mlx を同時実行してストリームを返す
        ↓
[llama-server (llama.cpp)]           [mlx-vlm-server (FastAPI + mlx-vlm)]
  Qwen2.5-VL GGUF + mmproj              Qwen2.5-VL mlx weights
  フレーム多枚画像として推論             動画ネイティブ入力 + M-RoPE
```

### 主要ファイル

| ファイル | 役割 |
|---|---|
| `app/api/vlm/route.ts` | Route Handler。4 モードの分岐・バリデーション・ストリーム返却 |
| `app/page.tsx` | クライアント UI。バックエンド選択、動画プレビュー、フレーム抽出、並列比較表示 |
| `lib/llm.ts` | Vercel AI SDK OpenAI 互換プロバイダ設定 |
| `lib/video.ts` | ブラウザ内で動画 → JPEG フレーム抽出 (HTMLVideoElement + Canvas) |
| `services/mlx-vlm-server/inference.py` | mlx-vlm モデルロードと推論ジェネレータ |
| `services/mlx-vlm-server/main.py` | FastAPI エントリポイント（ストリーミング推論）|
| `.env.example` | 環境変数テンプレート |

---

## VLM と動画理解について

### VLM とは

画像を入力に取れる LLM。ViT 系の視覚エンコーダで画像をトークン列に変換し、LLM のトークン空間に投影してからテキスト生成する構造です。

### 動画言語化の 2 方式と M-RoPE

| 方式 | 仕組み | このデモ |
|---|---|---|
| **フレーム多枚送信** | ブラウザで N 枚の JPEG を抽出し、複数の独立した画像として VLM に渡す | llama.cpp モード |
| **ネイティブ動画入力** | 動画ファイルをモデルに直接渡し、**M-RoPE (Multimodal Rotary Position Embedding)** で時間・縦・横の 3 軸エンコーディングを適用 | mlx-vlm モード |

**M-RoPE の特徴:**
- 通常の RoPE が 1 次元（トークン位置）なのに対し、M-RoPE は時間 (t) / 高さ (y) / 幅 (x) の 3 軸で位置を表現
- フレーム間の時間的変化がモデルに直接伝わる
- 空間トークン圧縮 (temporal compression) で同一フレームの視覚トークンを削減し、長い動画も処理できる

**なぜ llama.cpp ではネイティブ動画が使えないか:**
llama.cpp の mtmd（マルチモーダルパイプライン）と OpenAI 互換 API は動画コンテナを扱えません。LM Studio や Ollama も内部で llama.cpp を使うため同じ制約があります。M-RoPE を活かすには mlx-vlm / HuggingFace transformers / vLLM など専用スタックが必要です。

### フレーム多枚方式のトレードオフ

| | |
|---|---|
| ✅ | 画像対応の VLM ならすぐ動画対応できる |
| ✅ | 実装がシンプル（GPT-4o も内部的にはこれに近い） |
| ❌ | 抽出間隔より短い動きが欠落する（12 フレーム / 30 秒 = 約 2.5 秒間隔） |
| ❌ | フレームごとに独立したトークン列になり、ネイティブの空間圧縮が効かない |
| ❌ | 時間情報は「並び順」からの推測のみ |

UI のフレーム数スライダー (2〜12) と「並列比較」モードで両方式の出力差を体感できます。

---

## 環境変数

`.env.example` をコピーして `.env.local` を作成してください。

| 変数 | デフォルト | 説明 |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8082/v1` | llama-server 7B の URL |
| `LLAMA_MODEL_ID` | `qwen2.5-vl-7b` | llama-server に渡すモデル alias |
| `LLAMA_72B_SERVER_URL` | `http://127.0.0.1:8083/v1` | llama-server 72B の URL |
| `LLAMA_72B_MODEL_ID` | `qwen2.5-vl-72b` | 72B モデルの alias |
| `MLX_VLM_BASE_URL` | `http://127.0.0.1:8001` | mlx-vlm FastAPI サービスの URL |

---

## API リファレンス

### `GET /api/vlm`

全バックエンドのヘルスチェック結果を返します。

```json
{
  "llama_server_url": "http://127.0.0.1:8082/v1",
  "llama_72b_server_url": "http://127.0.0.1:8083/v1",
  "mlx": { "model_id": "...", "loaded": true, "error": null },
  "llama_72b": { "loaded": true }
}
```

### `POST /api/vlm`

`multipart/form-data` で送信。

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `prompt` | string | ✅ | 1〜2000 文字 |
| `mode` | string | – | `llama_cpp` / `llama_cpp_72b` / `mlx_native` (デフォルト: `llama_cpp`) |
| `image` | File | ※ | 画像モード用 (llama_cpp / mlx_native) |
| `frames` | File[] | ※ | 動画フレームモード用 (llama_cpp, 最大 12 枚) |
| `video` | File | ※ | 動画ファイル (mlx_native) |

レスポンスは `text/plain; charset=utf-8` のチャンクストリーム。

---

## 技術スタック

- **Next.js 16** (App Router, Turbopack)
- **React 19** / **TypeScript** (strict)
- **Tailwind CSS v4**
- **Vercel AI SDK v6** (`ai`, `@ai-sdk/openai-compatible`)
- **zod** (バリデーション)
- **llama.cpp** (`llama-server`, mtmd マルチモーダル)
- **mlx-vlm 0.5.0** (Apple Silicon ネイティブ推論)
- **FastAPI + uvicorn** (mlx-vlm サービス)
- **Qwen2.5-VL** 7B / 72B (GGUF + mlx weights)

---

## ライセンス

学習目的のサンプル。モデルファイルのライセンスは HuggingFace 上の各リポジトリに従います。
