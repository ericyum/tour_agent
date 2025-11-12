# 🎪 FestMoment - React 프론트엔드 완벽 가이드

> AI 기반 축제 가이드 서비스 - React + FastAPI로 재구축

---

## 📖 목차

1. [빠른 시작](#-빠른-시작)
2. [아키텍처](#-아키텍처)
3. [필수 요구사항](#-필수-요구사항)
4. [상세 설치 가이드](#-상세-설치-가이드)
5. [프로젝트 구조](#-프로젝트-구조)
6. [기술 스택](#-기술-스택)
7. [주요 기능](#-주요-기능)
8. [API 엔드포인트](#-api-엔드포인트)
9. [개발 가이드](#-개발-가이드)
10. [트러블슈팅](#-트러블슈팅)
11. [배포 가이드](#-배포-가이드)

---

## 🚀 빠른 시작

### ⚡ 한 눈에 보는 실행 방법

#### 1️⃣ 백엔드 서버 실행 (터미널 1)

```bash
# 프로젝트 루트 디렉토리에서
python api_server.py
```

✅ **실행 확인**: http://localhost:8000
📚 **API 문서**: http://localhost:8000/docs

#### 2️⃣ 프론트엔드 실행 (터미널 2)

```bash
# frontend 폴더로 이동
cd frontend

# 최초 1회만 실행 (의존성 설치)
npm install

# 개발 서버 시작
npm run dev
```

✅ **실행 확인**: http://localhost:3000

#### 3️⃣ 브라우저 접속

웹 브라우저에서 **http://localhost:3000** 접속! 🎉

---

## 🏗️ 아키텍처

### 시스템 구조

```
┌─────────────────────────────────────────┐
│   사용자 브라우저                         │
│   http://localhost:3000                 │
└────────────────┬────────────────────────┘
                 │ HTTP/REST API
┌────────────────▼────────────────────────┐
│   React Frontend (Port 3000)            │
│   ├─ Vite Dev Server                    │
│   ├─ React 18 + TypeScript              │
│   ├─ Tailwind CSS + Framer Motion      │
│   ├─ React Query (Data Fetching)       │
│   └─ Zustand (State Management)        │
└────────────────┬────────────────────────┘
                 │ Proxy to /api/*
┌────────────────▼────────────────────────┐
│   FastAPI Backend (Port 8000)           │
│   ├─ RESTful API Endpoints              │
│   ├─ CORS Middleware                    │
│   ├─ Request/Response Models            │
│   └─ Business Logic Integration         │
└────────────────┬────────────────────────┘
                 │ Function Calls
┌────────────────▼────────────────────────┐
│   Existing Backend Services             │
│   ├─ LangGraph Agents                   │
│   │   ├─ DB Search Agent                │
│   │   ├─ Naver Review Agent             │
│   │   ├─ Validation Agent               │
│   │   └─ Sentiment Analysis             │
│   ├─ Google Gemini LLM                  │
│   ├─ SQLite Database (tour.db)          │
│   └─ External APIs                      │
│       ├─ Naver Blog/Trend API           │
│       └─ Google Maps API                │
└─────────────────────────────────────────┘
```

### 데이터 흐름

```
사용자 클릭 → React Component → API Call (axios)
                    ↓
          React Query (캐싱/상태관리)
                    ↓
          FastAPI Endpoint (/api/*)
                    ↓
          Use Case Layer (비즈니스 로직)
                    ↓
          Agent Layer (LangGraph)
                    ↓
          외부 API / Database
                    ↓
          Response → React Query → Component 업데이트
```

---

## 📋 필수 요구사항

### 소프트웨어

| 항목 | 버전 | 설치 링크 |
|------|------|----------|
| **Python** | 3.11 이상 | https://www.python.org/downloads/ |
| **Node.js** | 18.0 이상 | https://nodejs.org/ |
| **npm** | 9.0 이상 | Node.js와 함께 설치됨 |

### API 키 (기존과 동일)

`.env` 파일에 다음 키들이 필요합니다:

```env
# Google Gemini (필수)
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MAPS_KEY=your_google_maps_key_here

# Naver (필수)
NAVER_CLIENT_ID=your_naver_client_id_here
NAVER_CLIENT_SECRET=your_naver_client_secret_here
NAVER_TREND_CLIENT_ID=your_trend_client_id_here
NAVER_TREND_CLIENT_SECRET=your_trend_client_secret_here
```

---

## 📥 상세 설치 가이드

### 1단계: 저장소 클론 (이미 있다면 생략)

```bash
git clone <repository-url>
cd tour_agent
```

### 2단계: Python 환경 설정

#### 가상환경 생성 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### Python 의존성 설치

```bash
pip install -r requirements.txt
```

**설치되는 주요 패키지:**
- `fastapi` - 백엔드 웹 프레임워크
- `uvicorn` - ASGI 서버
- `langchain-google-genai` - Gemini LLM
- `langgraph` - 에이전트 프레임워크
- 기타 기존 패키지들...

### 3단계: 환경 변수 설정

프로젝트 루트에 `.env` 파일이 있는지 확인:

```bash
# Windows
type .env

# macOS/Linux
cat .env
```

없다면 `.env.example`을 복사하여 생성:

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

그리고 API 키를 입력하세요.

### 4단계: 데이터베이스 초기화

```bash
# api_server.py를 실행하면 자동으로 초기화됩니다
# 또는 직접 초기화:
python -c "from src.infrastructure.persistence.database import init_db; init_db()"
```

### 5단계: 백엔드 서버 실행

```bash
# 방법 1: 직접 실행 (권장)
python api_server.py

# 방법 2: uvicorn으로 실행
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

**서버 실행 확인:**
- Health Check: http://localhost:8000
- API 문서: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

### 6단계: Node.js 환경 설정

새 터미널을 열고:

```bash
cd frontend
npm install
```

**설치되는 주요 패키지:**
- `react` & `react-dom` - React 라이브러리
- `react-router-dom` - 라우팅
- `@tanstack/react-query` - 데이터 페칭
- `zustand` - 상태 관리
- `axios` - HTTP 클라이언트
- `framer-motion` - 애니메이션
- `tailwindcss` - 스타일링

### 7단계: 프론트엔드 개발 서버 실행

```bash
npm run dev
```

**브라우저에서 접속:**
http://localhost:3000

---

## 📁 프로젝트 구조

```
tour_agent/
├── 📄 api_server.py              # FastAPI 백엔드 서버
├── 📄 requirements.txt           # Python 의존성
├── 📄 .env                       # 환경 변수 (git ignore)
├── 📄 tour.db                    # SQLite 데이터베이스
├── 📂 archive_gradio/            # 기존 Gradio UI (백업)
│   ├── app.py
│   └── presentation/
├── 📂 src/                       # 백엔드 소스코드
│   ├── 📂 application/           # 비즈니스 로직
│   │   ├── agents/              # LangGraph 에이전트
│   │   ├── core/                # 핵심 로직
│   │   ├── services/            # 서비스 레이어
│   │   ├── supervisors/         # 에이전트 조율
│   │   └── use_cases/           # 유스케이스
│   ├── 📂 domain/                # 도메인 모델
│   ├── 📂 infrastructure/        # 인프라
│   │   ├── config/              # 설정
│   │   ├── external_services/   # 외부 API
│   │   ├── persistence/         # 데이터베이스
│   │   └── reporting/           # 차트, 워드클라우드
│   └── 📂 presentation/          # (삭제됨, archive로 이동)
├── 📂 frontend/                  # React 프론트엔드 ⭐
│   ├── 📂 src/
│   │   ├── 📂 components/       # 재사용 가능한 컴포넌트
│   │   │   ├── layout/         # Header, Footer, Layout
│   │   │   ├── search/         # SearchFilters
│   │   │   └── festival/       # FestivalCard
│   │   ├── 📂 pages/            # 페이지 컴포넌트
│   │   │   ├── HomePage.tsx
│   │   │   ├── SearchPage.tsx
│   │   │   ├── FestivalDetailPage.tsx
│   │   │   └── MyCoursePage.tsx
│   │   ├── 📂 lib/              # 유틸리티
│   │   │   ├── api.ts          # API 클라이언트
│   │   │   └── utils.ts        # 헬퍼 함수
│   │   ├── 📂 store/            # 상태 관리
│   │   │   └── useCourseStore.ts
│   │   ├── 📂 types/            # TypeScript 타입
│   │   ├── App.tsx             # 메인 앱
│   │   ├── main.tsx            # 진입점
│   │   └── index.css           # 글로벌 스타일
│   ├── 📄 package.json
│   ├── 📄 vite.config.ts
│   ├── 📄 tsconfig.json
│   ├── 📄 tailwind.config.js
│   └── 📄 index.html
├── 📂 database/                  # DB 초기화 CSV
├── 📂 dic/                       # 감성 분석 사전
├── 📂 assets/                    # 정적 자산
└── 📂 best_images_and_icons/    # 축제 이미지/아이콘
```

---

## 🛠️ 기술 스택

### 백엔드

| 기술 | 버전 | 용도 |
|------|------|------|
| **FastAPI** | 0.115.0 | REST API 프레임워크 |
| **Uvicorn** | 0.32.0 | ASGI 웹 서버 |
| **LangGraph** | 1.0.1 | 에이전트 워크플로우 |
| **Gemini** | latest | LLM (Google) |
| **SQLite** | 3 | 데이터베이스 |
| **Pandas** | 2.3.3 | 데이터 처리 |
| **Playwright** | 1.55.0 | 웹 스크래핑 |

### 프론트엔드

| 기술 | 버전 | 용도 |
|------|------|------|
| **React** | 18.3.1 | UI 라이브러리 |
| **TypeScript** | 5.2.2 | 타입 안정성 |
| **Vite** | 5.3.1 | 빌드 도구 |
| **Tailwind CSS** | 3.4.10 | 스타일링 |
| **Framer Motion** | 11.5.4 | 애니메이션 |
| **React Router** | 6.26.0 | 라우팅 |
| **TanStack Query** | 5.56.0 | 데이터 페칭 |
| **Zustand** | 4.5.5 | 상태 관리 |
| **Axios** | 1.7.7 | HTTP 클라이언트 |
| **Recharts** | 2.12.7 | 차트 라이브러리 |
| **React Icons** | 5.3.0 | 아이콘 |
| **date-fns** | 3.6.0 | 날짜 처리 |

---

## 🎯 주요 기능

### ✅ 구현 완료

#### 1. 🏠 홈페이지
- **영웅 섹션**: 애니메이션과 그라데이션 효과
- **기능 소개**: 6가지 주요 기능 카드
- **CTA**: 검색 페이지로 이동 버튼
- **반응형**: 모바일/태블릿/데스크톱 최적화

#### 2. 🔍 축제 검색
- **다중 필터**:
  - 지역 선택 (시/도)
  - 시/군/구 선택 (지역에 종속)
  - 대/중/소 카테고리 (계층적 종속)
  - 진행 상태 (전체/진행중/예정/종료)
- **검색 결과**:
  - 반응형 카드 그리드 레이아웃
  - 호버 애니메이션
  - 상태 배지 (진행중/예정/종료)
  - '내 코스에 추가' 버튼
- **페이지네이션**: 이전/다음 버튼 및 페이지 번호
- **빈 상태 처리**: 결과 없음, 검색 전 안내

#### 3. 📋 축제 상세
- **히어로 섹션**: 대표 이미지 with 그라데이션
- **기본 정보**:
  - 축제명, 일정, 주소
  - 전화번호, 홈페이지 링크
  - 이용요금, 행사장소
- **상세 설명**: 축제 소개, 프로그램
- **액션**: 내 코스에 추가
- **알림**: 일정 변경 가능성 안내

#### 4. 📍 내 코스
- **코스 관리**:
  - 추가/삭제 기능
  - 순서 표시 (번호 배지)
  - LocalStorage 자동 저장
- **여행 설정**: 기간 선택 (당일~4박5일)
- **AI 검증**:
  - 코스 현실성 검토
  - 최적 동선 제안
  - 시간 배분 조언
- **UI/UX**:
  - 애니메이션 효과
  - 빈 상태 안내
  - 로딩 상태 표시

#### 5. 🎨 디자인 시스템
- **색상 테마**:
  - Primary: Blue (#0ea5e9)
  - Accent: Purple (#d946ef)
  - 그라데이션 조합
- **컴포넌트**:
  - 글래스모피즘 카드
  - 그라데이션 버튼
  - 호버/클릭 애니메이션
- **타이포그래피**: Inter 폰트
- **아이콘**: React Icons 사용

### 🚧 진행 중 (API는 있으나 UI 미완성)

#### 6. 📊 트렌드 분석
- ✅ API: `GET /api/festivals/{name}/trend`
- ❌ UI: 차트 컴포넌트 필요 (Recharts)
- 기능: 연간 검색량, 이벤트 기간 트렌드

#### 7. 💬 감성 분석
- ✅ API: `GET /api/festivals/{name}/sentiment`
- ❌ UI: 대시보드 컴포넌트 필요
- 기능: 블로그 리뷰 분석, 긍정/부정 비율, 워드클라우드

#### 8. 🎨 AI 렌더링
- ✅ API: `POST /api/festivals/{name}/render`
- ❌ UI: 이미지 갤러리 필요
- 기능: 야경/계절별 AI 생성 이미지

#### 9. 🏆 AI 랭킹
- ✅ API: `POST /api/festivals/ranking`
- ❌ UI: 랭킹 결과 표시 컴포넌트 필요
- 기능: 트렌드 + 감성 분석 기반 순위

#### 10. 🗺️ 주변 추천
- ✅ API: `POST /api/nearby/search`
- ❌ UI: 지도 + 추천 리스트 필요
- 기능: 반경 내 시설/코스/축제 추천

#### 11. 👑 AI 에티켓 가이드
- ✅ API: `GET /api/festivals/{name}/precautions`
- ❌ UI: 마크다운 렌더링 필요
- 기능: 축제별 맞춤 주의사항

---

## 🔌 API 엔드포인트

### 설정 API

```http
GET /api/config/areas
GET /api/config/sigungus?area={area}
GET /api/config/categories
GET /api/config/categories/medium?main_cat={main_cat}
GET /api/config/categories/small?main_cat={main_cat}&medium_cat={medium_cat}
```

### 축제 API

```http
POST /api/festivals/search
{
  "area": "서울",
  "sigungu": "강남구",
  "main_cat": "전통과 역사",
  "medium_cat": "전통 축제",
  "small_cat": "명절 축제",
  "status": "진행중",
  "page": 1
}

GET /api/festivals/{festivalName}
GET /api/festivals/{festivalName}/trend
GET /api/festivals/{festivalName}/sentiment?num_reviews=10
GET /api/festivals/{festivalName}/precautions

POST /api/festivals/ranking
{
  "festivals": ["축제A", "축제B", "축제C"],
  "num_reviews": 10,
  "top_n": 3
}

POST /api/festivals/{festivalName}/render
```

### 코스 API

```http
POST /api/course/validate
{
  "course": [
    { "title": "축제A", "type": "festival", ... },
    { "title": "관광지B", "type": "facility", ... }
  ],
  "duration": "1박 2일"
}
```

### 주변 검색 API

```http
POST /api/nearby/search
{
  "latitude": 37.5665,
  "longitude": 126.9780,
  "radius": 5000,
  "current_festival_id": "optional"
}
```

### Assets API

```http
GET /api/assets/best_images_and_icons/icons/{filename}
GET /api/assets/best_images_and_icons/best_images/{filename}
```

---

## 💻 개발 가이드

### 백엔드 개발

#### 새로운 API 엔드포인트 추가

```python
# api_server.py에 추가

@app.get("/api/your-endpoint")
async def your_endpoint(param: str = Query(...)):
    """엔드포인트 설명"""
    try:
        # 비즈니스 로직 호출
        result = your_service.do_something(param)
        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### Request/Response 모델 정의

```python
from pydantic import BaseModel

class YourRequest(BaseModel):
    field1: str
    field2: int

class YourResponse(BaseModel):
    result: str
    success: bool
```

### 프론트엔드 개발

#### 1. 새로운 페이지 추가

```tsx
// src/pages/NewPage.tsx
import { motion } from 'framer-motion'

export default function NewPage() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <h1 className="section-title">새 페이지</h1>
      {/* 내용 */}
    </motion.div>
  )
}

// App.tsx에 라우트 추가
<Route path="/new" element={<NewPage />} />
```

#### 2. API 함수 추가

```typescript
// src/lib/api.ts
export const yourNewApi = async (param: string) => {
  const { data } = await api.get('/your-endpoint', { params: { param } })
  return data
}
```

#### 3. API 호출 (React Query)

```tsx
import { useQuery } from '@tanstack/react-query'
import { yourNewApi } from '@/lib/api'

function YourComponent() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['your-key', param],
    queryFn: () => yourNewApi(param),
  })

  if (isLoading) return <div>Loading...</div>
  if (error) return <div>Error!</div>

  return <div>{data}</div>
}
```

#### 4. 상태 관리 (Zustand)

```typescript
// src/store/useYourStore.ts
import { create } from 'zustand'

interface YourStore {
  value: string
  setValue: (value: string) => void
}

export const useYourStore = create<YourStore>((set) => ({
  value: '',
  setValue: (value) => set({ value }),
}))

// 컴포넌트에서 사용
const { value, setValue } = useYourStore()
```

#### 5. 커스텀 훅 만들기

```typescript
// src/hooks/useYourHook.ts
import { useState, useEffect } from 'react'

export function useYourHook() {
  const [state, setState] = useState()

  useEffect(() => {
    // 로직
  }, [])

  return { state, setState }
}
```

#### 6. 스타일링 가이드

```tsx
{/* Tailwind 유틸리티 클래스 */}
<div className="card p-6 space-y-4">
  <h2 className="text-2xl font-bold gradient-text">제목</h2>

  {/* 커스텀 버튼 */}
  <button className="btn-primary">
    클릭
  </button>

  {/* 글래스 효과 */}
  <div className="glass p-4 rounded-xl">
    내용
  </div>

  {/* 그리드 레이아웃 */}
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    {/* 아이템들 */}
  </div>
</div>
```

### 개발 명령어

#### 백엔드

```bash
# 개발 서버 (auto-reload)
python api_server.py

# uvicorn 직접 실행
uvicorn api_server:app --reload --port 8000

# 데이터베이스 재초기화
python -c "from src.infrastructure.persistence.database import init_db; init_db()"
```

#### 프론트엔드

```bash
cd frontend

# 개발 서버
npm run dev

# 타입 체크
npm run lint

# 프로덕션 빌드
npm run build

# 빌드 결과 미리보기
npm run preview

# 패키지 업데이트
npm update

# 캐시 클리어
rm -rf node_modules package-lock.json
npm install
```

---

## 🐛 트러블슈팅

### 백엔드 문제

#### 문제: 포트 8000이 이미 사용 중

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID번호> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

#### 문제: ModuleNotFoundError

```bash
# 가상환경 활성화 확인
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 패키지 재설치
pip install -r requirements.txt --upgrade
```

#### 문제: SQLite 데이터베이스 오류

```bash
# 데이터베이스 삭제 후 재생성
rm tour.db
python api_server.py
```

#### 문제: API 키 오류

1. `.env` 파일이 프로젝트 **루트**에 있는지 확인
2. API 키에 공백이나 따옴표가 없는지 확인
3. 백엔드 서버 재시작

### 프론트엔드 문제

#### 문제: npm install 실패

```bash
# Node 버전 확인 (18+ 필요)
node --version

# 캐시 클리어 후 재설치
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

#### 문제: 백엔드 연결 실패 (CORS, Proxy)

1. 백엔드 서버가 실행 중인지 확인: http://localhost:8000
2. `vite.config.ts` 프록시 설정 확인:
```typescript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

#### 문제: 타입 오류

```bash
# TypeScript 재컴파일
npm run build

# 타입 정의 재설치
npm install --save-dev @types/node @types/react @types/react-dom
```

#### 문제: Tailwind 스타일이 적용 안됨

1. `tailwind.config.js`에서 content 경로 확인
2. 개발 서버 재시작
```bash
npm run dev
```

#### 문제: 빌드 오류

```bash
# 타입 체크
npm run lint

# 빌드 로그 확인
npm run build

# 특정 경고 무시 (필요시)
// @ts-ignore
```

### 일반적인 문제

#### 문제: 페이지가 흰 화면

1. 브라우저 콘솔(F12) 확인
2. 네트워크 탭에서 API 응답 확인
3. 백엔드 로그 확인

#### 문제: 이미지가 안 보임

1. `best_images_and_icons` 폴더 확인
2. API 경로 확인: `/api/assets/...`
3. Fallback 이미지 확인

#### 문제: 느린 성능

```bash
# 프론트엔드 프로덕션 빌드
cd frontend
npm run build
npm run preview

# 백엔드 최적화 (필요시 Gunicorn 사용)
pip install gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 🚀 배포 가이드

### 프론트엔드 배포 (Vercel 권장)

```bash
cd frontend

# 빌드
npm run build

# dist 폴더가 생성됨
# Vercel/Netlify에 dist 폴더 배포
```

**vercel.json** 설정:
```json
{
  "rewrites": [
    { "source": "/api/:path*", "destination": "https://your-backend.com/api/:path*" },
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

### 백엔드 배포 (Railway/Render 권장)

```bash
# Dockerfile 생성
FROM python:3.11

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 환경 변수 설정

배포 플랫폼에서 환경 변수 설정:
- `GOOGLE_API_KEY`
- `NAVER_CLIENT_ID`
- `NAVER_CLIENT_SECRET`
- 등등...

---

## 💡 개발 팁

### 1. Hot Reload 활용

- 백엔드: `python api_server.py` (uvicorn auto-reload)
- 프론트엔드: `npm run dev` (Vite HMR)
- 코드 변경 시 자동 새로고침!

### 2. 개발자 도구

- **React DevTools**: Chrome 확장 설치
- **Swagger UI**: http://localhost:8000/docs
- **Network Tab**: API 요청/응답 확인

### 3. 디버깅

```tsx
// 프론트엔드 디버깅
console.log('Debug:', data)
console.table(array)

// 에러 바운더리 (추후 구현 권장)
import { ErrorBoundary } from 'react-error-boundary'
```

```python
# 백엔드 디버깅
import logging
logging.basicConfig(level=logging.DEBUG)

# FastAPI 자동 로깅
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response
```

### 4. 코드 품질

```bash
# 프론트엔드 린트
npm run lint

# 타입 체크
npx tsc --noEmit

# Prettier (권장)
npm install --save-dev prettier
npx prettier --write src/**/*.tsx
```

### 5. Git 커밋 메시지

```
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 수정
style: 코드 포맷팅
refactor: 코드 리팩토링
test: 테스트 추가
chore: 빌드 설정 변경
```

---

## 📚 참고 자료

### 공식 문서

- [React](https://react.dev/)
- [TypeScript](https://www.typescriptlang.org/)
- [Vite](https://vitejs.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React Query](https://tanstack.com/query/latest)
- [Zustand](https://github.com/pmndrs/zustand)
- [Framer Motion](https://www.framer.com/motion/)

### 커뮤니티

- [React Discord](https://discord.gg/react)
- [FastAPI Discord](https://discord.gg/fastapi)
- [Tailwind Discord](https://discord.gg/tailwindcss)

---

## 📊 브라우저 지원

| 브라우저 | 버전 | 지원 |
|---------|------|------|
| Chrome | 최신 | ✅ 완전 지원 (권장) |
| Firefox | 최신 | ✅ 완전 지원 |
| Safari | 최신 | ✅ 완전 지원 |
| Edge | 최신 | ✅ 완전 지원 |
| IE | 11 | ❌ 미지원 |

---

## 🔒 보안 고려사항

1. **API 키 보호**
   - ✅ 모든 API 키는 백엔드에서만 관리
   - ✅ `.env` 파일은 `.gitignore`에 포함
   - ❌ 프론트엔드에 절대 노출 금지

2. **CORS 설정**
   - ✅ 개발: localhost:3000 허용
   - ⚠️ 프로덕션: 실제 도메인만 허용

3. **입력 검증**
   - ✅ Pydantic 모델로 백엔드 검증
   - ✅ TypeScript로 프론트엔드 타입 체크

4. **Rate Limiting** (추후 구현 권장)
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.get("/api/endpoint")
@limiter.limit("10/minute")
async def endpoint():
    ...
```

---

## 🎓 학습 경로

### 초급 (프론트엔드)
1. React 기초
2. TypeScript 기초
3. Tailwind CSS
4. 라우팅 (React Router)

### 중급
5. 상태 관리 (Zustand)
6. 데이터 페칭 (React Query)
7. 애니메이션 (Framer Motion)
8. API 통신 (Axios)

### 고급
9. 성능 최적화
10. 에러 처리
11. 테스트 (Jest, React Testing Library)
12. 배포 및 CI/CD

---

## 📝 라이센스

© 2025 FestMoment. All rights reserved.

**Team FestMoment**
- 염정운
- 최가윤

---

## 🙋 FAQ

### Q: Gradio UI는 완전히 삭제해도 되나요?
A: `archive_gradio/` 폴더에 백업되어 있으니 필요시 복구 가능합니다.

### Q: 프로덕션 배포는 어떻게 하나요?
A: 프론트엔드는 Vercel, 백엔드는 Railway를 권장합니다. (배포 가이드 참조)

### Q: 모바일 앱으로 만들 수 있나요?
A: React Native로 포팅하거나 PWA로 만들 수 있습니다.

### Q: 더 많은 AI 기능을 추가하고 싶어요
A: `api_server.py`에 엔드포인트 추가 후 프론트엔드에서 호출하면 됩니다.

### Q: 데이터베이스를 PostgreSQL로 바꿀 수 있나요?
A: 네, `src/infrastructure/persistence/database.py`를 수정하면 됩니다.

---

**행복한 개발 되세요! 🎪✨**

문제가 있거나 질문이 있으시면 언제든 GitHub Issues에 올려주세요.
