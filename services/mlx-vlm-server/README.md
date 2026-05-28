# mlx-vlm-server

Qwen2.5-VL の**ネイティブ動画入力 (M-RoPE)** を提供する FastAPI サービス。  
mlx-vlm を使って動画・画像を Apple Silicon 上でネイティブ推論する。

---

## セットアップ

### 1. uv のインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Python 3.12 環境の作成

システム Python が 3.14 の場合、mlx の C 拡張が未対応なため Python 3.12 を使用する。

```bash
cd services/mlx-vlm-server
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 3. 依存パッケージのインストール

```bash
uv pip install "mlx-vlm==0.5.0" "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9"
```

---

## モデルのダウンロード

```bash
# 7B 量子化版 (約 8 GB) — 疎通確認用
hf download mlx-community/Qwen2.5-VL-7B-Instruct-8bit

# 72B bf16 (約 145 GB) — 本番用 (Mac Studio M3 Ultra 512 GB で動作確認済み)
hf download mlx-community/Qwen2.5-VL-72B-Instruct-bf16
```

---

## 起動

```bash
source .venv/bin/activate

# 7B で起動 (疎通確認用)
MLX_MODEL_ID=mlx-community/Qwen2.5-VL-7B-Instruct-8bit \
  uvicorn main:app --host 127.0.0.1 --port 8001 --timeout-keep-alive 600

# 72B で起動 (本番・デフォルト)
uvicorn main:app --host 127.0.0.1 --port 8001 --timeout-keep-alive 600
```

"Application startup complete." が出たらモデルロード完了。72B はロードに **1〜3 分**かかる。

---

## 疎通確認

```bash
# ヘルスチェック
curl http://127.0.0.1:8001/v1/health
# → {"model_id":"...","loaded":true,"error":null}

# 画像推論
curl -N \
  -F 'prompt=この画像を日本語で詳しく説明してください' \
  -F 'image=@/path/to/test.jpg' \
  http://127.0.0.1:8001/v1/infer

# 動画推論 (ネイティブ入力)
curl -N \
  -F 'prompt=この動画で何が起きているか時系列で日本語で説明してください' \
  -F 'video=@/path/to/test.mp4' \
  http://127.0.0.1:8001/v1/infer
```

レスポンスは `text/plain; charset=utf-8` のチャンクストリーム。

---

## API

### `GET /v1/health`

```json
{
  "model_id": "mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
  "loaded": true,
  "error": null
}
```

### `POST /v1/infer`

`multipart/form-data` で送信。

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `prompt` | string | ✅ | ユーザープロンプト |
| `video` | File | ※ | 動画ファイル (`video` か `image` のいずれか必須) |
| `image` | File | ※ | 画像ファイル |
| `max_tokens` | int | – | デフォルト 1024 |

レスポンスはテキストのチャンクストリーム (`text/plain; charset=utf-8`)。

---

## 実装上の注意点

- **Metal Stream のスレッド制約**: mlx の Metal Stream はスレッドローカルのため、`run_in_executor` などでスレッドをまたぐと `RuntimeError: There is no Stream(gpu, 1)` が発生する。そのため推論は async generator 内のメインスレッドで実行し、各トークン yield 後に `await asyncio.sleep(0)` で制御を返す方式を採用。
- **chat template**: `processor.apply_chat_template` に `{"type": "video", "video": path}` を含む messages を渡すことで、Qwen2.5-VL の動画トークンが正しく埋め込まれる。

---

## 推論速度の目安 (Mac Studio M3 Ultra)

| モデル | prefill | 生成速度 |
|---|---|---|
| Qwen2.5-VL-7B-Instruct-8bit | ~1000 tok/s | 30〜60 tok/s |
| Qwen2.5-VL-72B-Instruct-bf16 | ~200 tok/s | 5〜10 tok/s |

72B は動画の長文出力で 30 秒〜数分かかることがある。
