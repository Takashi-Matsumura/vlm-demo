import { streamText, type ModelMessage } from 'ai';
import { z } from 'zod';
import { llama, VLM_MODEL, LLAMA_SERVER_URL } from '@/lib/llm';

const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const MAX_FRAMES = 12;
const ALLOWED_MIME = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/gif']);

const promptSchema = z.string().trim().min(1).max(2000);

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8' },
  });
}

async function fileToImagePart(file: File): Promise<{
  type: 'image';
  image: Uint8Array;
  mediaType: string;
}> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  return { type: 'image', image: bytes, mediaType: file.type };
}

function validateImageFile(file: File): { ok: true } | { ok: false; status: number; body: unknown } {
  if (!ALLOWED_MIME.has(file.type)) {
    return {
      ok: false,
      status: 415,
      body: {
        error: `unsupported media type: ${file.type || 'unknown'}`,
        allowed: [...ALLOWED_MIME],
      },
    };
  }
  if (file.size > MAX_IMAGE_BYTES) {
    return {
      ok: false,
      status: 413,
      body: { error: 'image too large', limitBytes: MAX_IMAGE_BYTES, actualBytes: file.size },
    };
  }
  return { ok: true };
}

export async function POST(request: Request): Promise<Response> {
  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return json(400, { error: 'invalid multipart/form-data body' });
  }

  const promptRaw = form.get('prompt');
  if (typeof promptRaw !== 'string') {
    return json(400, { error: '`prompt` field is required (string)' });
  }
  const promptResult = promptSchema.safeParse(promptRaw);
  if (!promptResult.success) {
    return json(400, { error: 'invalid prompt', detail: promptResult.error.issues });
  }
  const prompt = promptResult.data;

  const frameFields = form.getAll('frames').filter((v): v is File => v instanceof File);
  const imageField = form.get('image');
  const isVideoMode = frameFields.length > 0;

  if (!isVideoMode && !(imageField instanceof File)) {
    return json(400, {
      error: '`image` (single File) or `frames` (multiple File) field is required',
    });
  }

  const files = isVideoMode ? frameFields : [imageField as File];

  if (files.length > MAX_FRAMES) {
    return json(413, { error: `too many frames`, limit: MAX_FRAMES, actual: files.length });
  }

  for (const f of files) {
    const v = validateImageFile(f);
    if (!v.ok) return json(v.status, v.body);
  }

  const imageParts = await Promise.all(files.map(fileToImagePart));

  const messages: ModelMessage[] = isVideoMode
    ? [
        {
          role: 'system',
          content:
            `あなたは動画を理解できる視覚言語モデルです。` +
            `ユーザーは 1 本の動画から時系列順に等間隔で抽出された ${files.length} 枚のフレームを送ります。` +
            `これらを連続するフレームとして扱い、動画全体で何が起きているかを統合的に説明してください。` +
            `静止画ごとに個別の説明を並べるのではなく、時間的変化を踏まえた 1 つの記述にまとめてください。`,
        },
        {
          role: 'user',
          content: [{ type: 'text', text: prompt }, ...imageParts],
        },
      ]
    : [
        {
          role: 'user',
          content: [{ type: 'text', text: prompt }, ...imageParts],
        },
      ];

  try {
    const result = streamText({
      model: llama(VLM_MODEL),
      messages,
    });
    return result.toTextStreamResponse();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const isConnRefused = /ECONNREFUSED|fetch failed|ENOTFOUND/i.test(message);
    if (isConnRefused) {
      return json(503, {
        error: 'llama-server unreachable',
        url: LLAMA_SERVER_URL,
        detail: message,
      });
    }
    return json(500, { error: 'inference failed', detail: message });
  }
}
