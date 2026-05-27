import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

const baseURL = process.env.LLAMA_SERVER_URL ?? 'http://127.0.0.1:8080/v1';

export const llama = createOpenAICompatible({
  name: 'llama-server',
  baseURL,
});

export const VLM_MODEL = process.env.LLAMA_MODEL_ID ?? 'qwen2.5-vl-7b';
export const LLAMA_SERVER_URL = baseURL;
