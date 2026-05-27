# mlx-vlm 拡張設計 — Qwen2.5-VL-72B ネイティブ動画入力 on Apple Silicon

## 1. 背景とゴール

### 1.1 現状の制約

本デモは **llama.cpp の `llama-server`** を推論バックエンドに使い、動画はクライアント側で **N 枚のフレームに分解** → **複数枚の独立した画像** として VLM に送る方式で動作している。これは llama.cpp の `mtmd` パイプラインが動画コンテナを直接扱えないことに起因する制約であり、本質的に以下の損失がある:

- 抽出間隔より短い動きやイベントが欠落する
- 各フレームが独立した画像トークン列になり、ネイティブ方式の **空間トークン圧縮**（隣接フレーム間の冗長性削減）が効かない
- 時間情報は「並び順」のみで、本物の **時間軸位置エンコーディング (Qwen2.5-VL の M-RoPE)** ではない

### 1.2 拡張のゴール

**Mac Studio M3 Ultra 512GB** 上に **mlx-vlm + Qwen2.5-VL-72B** を立て、動画ファイルをそのままモデルに渡せる **ネイティブ動画入力モード** を本デモに追加する。

学習目的としては:

- M-RoPE が効いた動画理解の精度を体感する
- 72B クラスの品質を 7B (今回のデモ) と比較する
- トークン消費・レイテンシ・短時間モーション補足精度を **既存方式と並べて比較** する

### 1.3 非ゴール

- mlx-vlm を OpenAI 互換にフル準拠させる (薄い shim で十分)
- 本番運用 (デモはあくまで学習用)
- Web 公開 (LAN 内・ローカルのみ)
- 音声入力（後日 Qwen2.5-Omni 検討の余地あり）

---

## 2. アーキテクチャ全体像

```
[Browser]
   │  ファイル選択 + プロンプト + モード選択
   ▼
[Next.js (既存) — app/api/vlm/route.ts]
   │
   ├─ mode = 'llama_cpp'   (既存パス: フレーム多枚 / 単一画像)
   │     │
   │     ▼
   │  llama-server (llama.cpp + Qwen2.5-VL-7B GGUF)
   │
   └─ mode = 'mlx_native'  (新規: 動画ファイル直送 / 画像直送)
         │
         ▼
   [Python サービス (新規) — mlx-vlm-server]
         │
         ▼
   mlx-vlm  ──  Qwen2.5-VL-72B (mlx-community/Qwen2.5-VL-72B-Instruct-bf16)
```

**ポイント:**

- 既存の Next.js は **薄いフロント + BFF** として残し、Python 側は **推論専用** に切り離す。Node ↔ Python のプロトコルは独自 (OpenAI 完全互換は目指さない)
- 既存の llama.cpp パスと **共存** させ、UI のトグルで切り替え／同時実行できるようにする
- すべて localhost / LAN 内で完結

---

## 3. 設計判断

### 3.1 なぜ mlx-vlm か

| 選択肢 | Mac 対応 | ネイティブ動画 | 評価 |
|---|---|---|---|
| **mlx-vlm** | ✅ Apple ネイティブ | ✅ Qwen2.5-VL 対応済み | **本命** |
| transformers + MPS | △ 動くが ops 一部 CPU フォールバック | △ 公式実装に最も近いが遅い | 第二候補 |
| vLLM | ❌ CUDA / ROCm のみ | — | Mac では使えない |
| llama.cpp + Metal | ✅ | ❌（今回の制約） | 既存パスに使用 |

MLX は Apple Silicon の **統合メモリ + Metal** を前提に設計されており、72B のような巨大モデルでも統合メモリのおかげで CPU↔GPU 転送オーバーヘッドがない。Apple 純正の `mlx` フレームワーク上で構築された `mlx-vlm` は VLM 専用に Qwen2.5-VL の動画前処理 (`qwen-vl-utils` 相当) を含んでおり、最も Mac 寄りの選択肢。

### 3.2 なぜ FastAPI ラッパーを挟むのか

`mlx-vlm` 自体にもサーバーモード (`python -m mlx_vlm.server`) はあるが、

- API シェイプがバージョンで変動する
- 動画対応の有無や認証周りが不明確
- ストリーミング応答の制御が限定的

ため、**自分で薄い FastAPI を書いた方が結果として早い**。50〜100 行程度の shim。

### 3.3 なぜ 72B か

Mac Studio M3 Ultra 512GB のメモリは 72B-bf16 (約 150GB) を **快適に乗せられる** 規模であり、72B の動画理解は 7B と質的に違う。学習デモとして「**最大限の品質**」と「**最小限の品質 (7B 多枚画像方式)**」を比較できるのは価値が高い。

| モデル | bf16 サイズ | 推奨用途 |
|---|---|---|
| Qwen2.5-VL-7B | ~15 GB | 軽量テスト |
| Qwen2.5-VL-32B | ~65 GB | バランス |
| **Qwen2.5-VL-72B** | **~150 GB** | **本拡張のメイン** |

途中段階の確認用に 7B でも動かせるよう、モデル ID は環境変数で切り替え可能にする。

### 3.4 なぜ既存 Next.js を流用するか

新規実装を最小化するため。フロント UI とストリーミング配信ロジックは既に動くので、`/api/vlm` に **mode 分岐** を 1 段足して、新しい Python サービスへ proxy するだけで済む。

---

## 4. コンポーネント設計

### 4.1 Python サービス (新規)

#### 配置

```
services/
└── mlx-vlm-server/
    ├── pyproject.toml         # uv / pip-tools いずれでも
    ├── main.py                # FastAPI エントリポイント
    ├── inference.py           # mlx-vlm ロード + 推論
    └── README.md
```

別リポジトリにしてもよいが、**学習デモを 1 つの GitHub プロジェクトでまとめて読めるようにする**ため本リポジトリ内に同居させる。

#### 技術スタック

| コンポーネント | 役割 |
|---|---|
| Python 3.11+ | 言語 |
| `mlx-vlm` | VLM 推論 |
| `mlx` | テンソル基盤 (mlx-vlm の依存) |
| `fastapi` + `uvicorn` | HTTP サーバ |
| `python-multipart` | multipart/form-data 受信 |
| `av` (PyAV) | 動画コンテナデコード (mlx-vlm が内部で要求) |

#### API 設計

**`POST /v1/infer`** (multipart/form-data, streaming response)

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `prompt` | string | ✅ | ユーザープロンプト |
| `video` | File | ※ | 動画モード用 |
| `image` | File | ※ | 画像モード用 |
| `max_tokens` | int | – | デフォルト 1024 |

`video` か `image` のいずれかが必須。

**レスポンス**: `text/event-stream` (Server-Sent Events) もしくは `text/plain` のチャンクストリーム。デモの実装簡略化のため `text/plain; charset=utf-8` を採用 (既存の Next.js 側 `toTextStreamResponse` と同じ形式)。

**`GET /v1/health`**: 200 OK + モデル ID とロード状態の JSON

#### 推論フロー (擬似コード)

```python
# inference.py
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import process_vision_info  # Qwen 公式 utils 相当

MODEL_ID = os.environ.get("MLX_MODEL_ID", "mlx-community/Qwen2.5-VL-72B-Instruct-bf16")
model, processor = load(MODEL_ID)  # 起動時に 1 回ロード

def run(prompt: str, video_path: str | None, image_path: str | None):
    messages = build_messages(prompt, video_path, image_path)
    inputs = process_vision_info(messages, processor)
    for chunk in stream_generate(model, processor, **inputs, max_tokens=1024):
        yield chunk.text
```

> **VERIFY**: `mlx-vlm` の最新バージョンで `stream_generate` のシグネチャと動画入力の渡し方を確認すること。バージョン差分が大きく、`messages` 形式 (`{"type": "video", "video": path}`) と `process_vision_info` のインタフェースは流動的。pyproject.toml で **バージョン pin** する。

#### モデルダウンロードと起動

```bash
# 初回のみ (約 150GB、数十分〜数時間)
huggingface-cli download mlx-community/Qwen2.5-VL-72B-Instruct-bf16

# 起動
cd services/mlx-vlm-server
uvicorn main:app --host 127.0.0.1 --port 8001
```

メモリ常駐 + ロード時間が長いため、開発中はサービスを起動しっぱなしにする運用。

### 4.2 Next.js 側変更

#### 環境変数追加 (`.env.example`)

```
# 既存
LLAMA_SERVER_URL=http://127.0.0.1:8080/v1
LLAMA_MODEL_ID=qwen2.5-vl-7b

# 新規
MLX_VLM_BASE_URL=http://127.0.0.1:8001
```

#### Route Handler の改修

既存 `app/api/vlm/route.ts` に **`mode` フィールド** で分岐を追加する。

| `mode` | 受け取り | 転送先 | 備考 |
|---|---|---|---|
| `llama_cpp` (デフォルト) | `image` 単発 or `frames[]` 多枚 | llama-server | 既存パス、後方互換 |
| `mlx_native` | `image` 単発 or `video` 1 本 | Python サービス `/v1/infer` | 新規パス |

Python サービスへの転送は `fetch` で multipart をそのまま中継する。AI SDK は経由しない (OpenAI 互換ではないため)。

#### UI 改修 (`app/page.tsx`)

3 つの動作モードをラジオで提供:

1. **🟢 llama.cpp 多枚画像** (現状、デフォルト) — 既存の動画フレーム抽出方式
2. **🔵 mlx-vlm ネイティブ動画** — 動画ファイルをそのまま送信
3. **⚖️ 並列比較** — 同じ入力を 2 つのバックエンドに同時投入し、結果を左右に並べる

並列比較モードのとき、出力エリアを 2 カラムに分割し、レイテンシ・最終トークン数・最初の出力までの時間 (TTFT) を簡易表示する。**これが本拡張の最大の学習価値**。

---

## 5. 実装フェーズ

### Phase 1: Python サービスの単体検証

- `mlx-vlm` を CLI で動画 1 本に対して推論させ、`Qwen2.5-VL-7B` でまず疎通確認 (72B は DL に時間がかかるので後回し)
- 7B での動作確認後、72B の DL とロードに進む
- ロード時間・メモリ占有を計測

### Phase 2: FastAPI ラッパー

- `POST /v1/infer` を実装 (画像のみ → 動画追加の順)
- chunked streaming で text を返せること
- `curl -N -F` で疎通確認

### Phase 3: Next.js から接続

- `.env.local` に `MLX_VLM_BASE_URL` を追加
- `app/api/vlm/route.ts` に `mode` 分岐を追加し、`mlx_native` で Python に proxy
- UI に「バックエンド: llama_cpp / mlx_native」のラジオを追加 (まずは並列比較なし)

### Phase 4: 並列比較モード

- UI に「並列比較」モードを追加
- 出力エリアを 2 カラムに、メトリクス (TTFT / 総トークン / 完了時間) を表示
- 同一入力で両方を fire-and-stream

### Phase 5: 検証 + ドキュメント

- 同一動画 (例: `~/Downloads/vlm-test.mp4`) で 2 方式の差を観察
- README に観察結果を追記 (「速いシーンは多枚画像で欠落するが、ネイティブだと拾えた」等)

---

## 6. 検証項目

| 項目 | 確認方法 |
|---|---|
| mlx-vlm が 72B を 512GB 上でロードできる | サービス起動ログ + `top` でメモリ確認 |
| 動画 1 本でストリーミング応答が返る | `curl -N -F video=@... -F prompt=... http://127.0.0.1:8001/v1/infer` |
| Next.js から mlx_native モードで応答が返る | ブラウザで送信、開発者ツールで Network 確認 |
| 並列比較で TTFT と総トークン数が表示される | ブラウザでモード切替、ストップウォッチ手動確認 |
| 短時間モーションがネイティブ方式で拾えるか | 動きの速い動画（スポーツ等）で比較 |

---

## 7. 注意点 / 想定リスク

### モデル DL とストレージ

- 72B-bf16 は約 **150GB**。HuggingFace キャッシュ (`~/.cache/huggingface/`) が置かれるディスクの空き容量を要確認
- 量子化版 (`mlx-community/Qwen2.5-VL-72B-Instruct-4bit` 等) は半分以下になるが、学習目的なら bf16 推奨

### mlx-vlm の API 流動性

- `mlx-vlm` はまだ **頻繁に API が変わる**。`pyproject.toml` で `mlx-vlm==<具体的版>` を pin する
- 動画入力の messages 形式 (`{"type": "video", "video": ...}`) は本家 transformers の `qwen-vl-utils` と同型だが、mlx-vlm 側で先行/遅延がある

### ストリーミングの実装

- `mlx-vlm` の `stream_generate` がジェネレータで返るので、FastAPI 側は `StreamingResponse(generator())` で薄くラップ
- Node ↔ Python 間は `fetch` の `Response.body.pipeThrough(TextDecoderStream)` でそのまま中継

### メモリ使用

72B + KV cache + 動画フレーム埋め込みで **概算 200〜260 GB**。512GB に対して余裕はあるが、他のアプリ (Docker, Chrome 大量タブ等) との同時使用には注意。

### 推論速度

| モデル | M3 Ultra での目安 |
|---|---|
| 7B | 30〜60 tok/s |
| 32B | 10〜25 tok/s |
| **72B** | **5〜10 tok/s** |

72B は遅いが、学習デモとしては許容できる範囲。動画のような長文出力で 30 秒〜数分かかることを想定。

### 動画長

`max_model_len` 相当 (mlx-vlm の context window) を超える長尺動画は **フレーム間引き** がより強く効くため、極端な精度向上が出ない場合がある。**まずは 30 秒〜2 分程度の動画** で評価する。

---

## 8. ファイル構成 (この拡張で追加されるもの)

```
vlm-demo/
├── app/
│   ├── api/
│   │   └── vlm/
│   │       └── route.ts          # mode 分岐を追加 (既存)
│   └── page.tsx                  # バックエンドトグル + 並列比較UI (既存)
├── docs/
│   └── mlx-vlm-extension.md      # 本ドキュメント
├── lib/
│   ├── llm.ts                    # MLX baseURL を追加 (既存)
│   └── video.ts                  # 既存、変更なし
├── services/
│   └── mlx-vlm-server/           # 新規 Python サービス
│       ├── pyproject.toml
│       ├── main.py
│       ├── inference.py
│       └── README.md
└── .env.example                  # MLX_VLM_BASE_URL を追記 (既存)
```

---

## 9. 将来の発展余地

- **Qwen2.5-Omni** (音声 + 動画 + 画像) への切替で、動画の **音声トラックも含めた理解** が可能に
- マルチターン会話化 (動画 + 質問の連続)
- 並列比較モードのメトリクスを自動収集して CSV 出力 → 学習レポート用
- 量子化版 (4bit / 8bit) との品質・速度比較
- transformers + MPS バックエンドを 3 つ目の比較対象として追加

---

## 10. 参考リンク

- [mlx-vlm (GitHub)](https://github.com/Blaizzy/mlx-vlm)
- [mlx-community on HuggingFace](https://huggingface.co/mlx-community)
- [Qwen2.5-VL 公式 (Hugging Face)](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)
- [Qwen2.5-VL 論文 (arXiv)](https://arxiv.org/abs/2502.13923)
- [MLX 公式](https://github.com/ml-explore/mlx)
