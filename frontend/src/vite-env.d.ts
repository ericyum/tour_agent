/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  readonly VITE_GOOGLE_CLIENT_ID: string
  // 필요한 env 변수 여기에 추가
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}