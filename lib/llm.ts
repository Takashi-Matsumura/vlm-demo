import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

const baseURL = process.env.LLAMA_SERVER_URL ?? 'http://127.0.0.1:8080/v1';

export const llama = createOpenAICompatible({
  name: 'llama-server',
  baseURL,
});

export const VLM_MODEL = process.env.LLAMA_MODEL_ID ?? 'qwen2.5-vl-7b';
export const LLAMA_SERVER_URL = baseURL;
export const MLX_VLM_BASE_URL = process.env.MLX_VLM_BASE_URL ?? 'http://127.0.0.1:8001';

const base72B = process.env.LLAMA_72B_SERVER_URL ?? 'http://127.0.0.1:8083/v1';
export const llama72b = createOpenAICompatible({ name: 'llama-server-72b', baseURL: base72B });
export const VLM_72B_MODEL = process.env.LLAMA_72B_MODEL_ID ?? 'qwen2.5-vl-72b';
export const LLAMA_72B_SERVER_URL = base72B;
