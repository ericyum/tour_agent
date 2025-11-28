# 🚀 FestMoment Render 배포 가이드

이 가이드는 FestMoment 앱을 **Render**에 **무료**로 배포하는 방법을 설명합니다.

---

## 📋 목차

1. [사전 준비](#사전-준비)
2. [Render 계정 생성](#1-render-계정-생성)
3. [PostgreSQL 데이터베이스 생성](#2-postgresql-데이터베이스-생성)
4. [Redis 생성 (선택)](#3-redis-생성-선택)
5. [Backend API 배포](#4-backend-api-배포)
6. [Frontend 배포](#5-frontend-배포)
7. [도메인 연결](#6-도메인-연결-festmomentcokr)
8. [환경 변수 설정](#7-환경-변수-설정)
9. [데이터베이스 초기화](#8-데이터베이스-초기화)

---

## 사전 준비

### 필수 항목
- [x] GitHub 계정 (저장소: `ericyum/tour_agent`)
- [x] Google Gemini API Key
- [x] Naver API Keys (Search, DataLab)
- [x] 도메인 (`festmoment.co.kr` - 가비아)

### 비용
- ✅ **완전 무료!** (Render Free Tier)
- ⚠️ **제한사항**:
  - 15분 미사용 시 슬립 모드 (첫 요청 느림)
  - PostgreSQL: 90일 후 삭제 (백업 필요)
  - Redis: 25MB 제한

---

## 1. Render 계정 생성

### 1-1. Render 가입
1. [https://render.com](https://render.com) 접속
2. **Sign Up** 클릭
3. **GitHub**로 계정 연결 (권장)
4. GitHub 저장소 접근 권한 허용

### 1-2. GitHub 저장소 연결
1. Render 대시보드에서 **New +** 클릭
2. **Connect a repository** 선택
3. `ericyum/tour_agent` 저장소 선택

---

## 2. PostgreSQL 데이터베이스 생성

### 2-1. 데이터베이스 생성
1. Render 대시보드에서 **New +** → **PostgreSQL** 선택
2. 설정:
   ```
   Name: festmoment-db
   Database: festmoment
   User: festmoment
   Region: Singapore (가장 가까운 리전)
   PostgreSQL Version: 16
   Instance Type: Free
   ```
3. **Create Database** 클릭

### 2-2. Connection String 복사
생성 후 **Internal Database URL** 복사 (나중에 사용)
```
postgres://festmoment:xxxxx@xxxxx.render.com/festmoment
```

---

## 3. Redis 생성 (선택)

⚠️ **참고**: Render 무료 Redis는 25MB 제한. 캐싱 기능 제한적.

1. **New +** → **Redis** 선택
2. 설정:
   ```
   Name: festmoment-redis
   Region: Singapore
   Instance Type: Free (25MB)
   ```
3. **Create Redis** 클릭
4. **Internal Redis URL** 복사

---

## 4. Backend API 배포

### 4-1. Web Service 생성
1. **New +** → **Web Service** 선택
2. **Build and deploy from a Git repository** 선택
3. 저장소: `ericyum/tour_agent` 선택

### 4-2. 서비스 설정
```
Name: festmoment-api
Region: Singapore
Branch: main
Root Directory: (비워두기 - 루트)
Runtime: Docker
Dockerfile Path: ./Dockerfile
Instance Type: Free
```

### 4-3. 환경 변수 설정 (Environment Variables)

**Database:**
```
DATABASE_URL = <2단계에서 복사한 PostgreSQL Internal URL>
REDIS_URL = <3단계에서 복사한 Redis URL> (선택)
```

**API Keys:**
```
GOOGLE_API_KEY = your-google-gemini-api-key
GEMINI_MAPS_KEY = your-gemini-maps-key
NAVER_CLIENT_ID = your-naver-client-id
NAVER_CLIENT_SECRET = your-naver-client-secret
NAVER_TREND_CLIENT_ID = your-naver-trend-client-id
NAVER_TREND_CLIENT_SECRET = your-naver-trend-client-secret
```

**Authentication:**
```
JWT_SECRET_KEY = <강력한 랜덤 문자열 생성>
GOOGLE_CLIENT_ID = your-google-oauth-client-id
GOOGLE_CLIENT_SECRET = your-google-oauth-client-secret
```

**기타:**
```
FRONTEND_URL = https://festmoment.co.kr
PYTHONUNBUFFERED = 1
PORT = 8080
```

> 💡 **JWT_SECRET_KEY 생성 방법:**
> ```bash
> openssl rand -base64 32
> ```

### 4-4. 배포 시작
1. **Create Web Service** 클릭
2. 자동 빌드 시작 (약 5-10분 소요)
3. 배포 완료 후 **URL 복사** (예: `https://festmoment-api.onrender.com`)

---

## 5. Frontend 배포

### 5-1. Static Site 생성
1. **New +** → **Static Site** 선택
2. 저장소: `ericyum/tour_agent` 선택

### 5-2. 사이트 설정
```
Name: festmoment-frontend
Region: Singapore
Branch: main
Root Directory: frontend
Build Command: npm install && npm run build
Publish Directory: dist
```

### 5-3. 환경 변수 설정
```
VITE_API_URL = <4-4단계에서 복사한 Backend URL>
VITE_GOOGLE_CLIENT_ID = your-google-oauth-client-id
```

예시:
```
VITE_API_URL = https://festmoment-api.onrender.com
```

### 5-4. 배포 시작
1. **Create Static Site** 클릭
2. 빌드 시작 (약 3-5분 소요)
3. 배포 완료 후 URL 확인 (예: `https://festmoment.onrender.com`)

---

## 6. 도메인 연결 (`festmoment.co.kr`)

### 6-1. Render에서 Custom Domain 추가

**Frontend Static Site에 도메인 연결:**
1. Frontend 서비스 선택
2. **Settings** → **Custom Domains** 섹션
3. **Add Custom Domain** 클릭
4. 도메인 입력:
   - `festmoment.co.kr`
   - `www.festmoment.co.kr`
5. Render가 제공하는 **CNAME 레코드** 복사

예시:
```
festmoment.co.kr → CNAME → festmoment.onrender.com
www.festmoment.co.kr → CNAME → festmoment.onrender.com
```

### 6-2. 가비아 DNS 설정

1. [가비아](https://www.gabia.com/) 로그인
2. **My가비아** → **서비스 관리** → **도메인**
3. `festmoment.co.kr` → **DNS 정보** 클릭
4. **레코드 추가**:

| 타입 | 호스트 | 값/위치 | TTL |
|------|--------|---------|-----|
| CNAME | @ | festmoment.onrender.com | 3600 |
| CNAME | www | festmoment.onrender.com | 3600 |

> ⚠️ **주의**: 일부 DNS 제공자는 루트 도메인(@)에 CNAME을 지원 안 할 수 있음.
> 그 경우 A 레코드로 변경하거나, `www.festmoment.co.kr`만 사용.

5. 저장 후 **10분~1시간 대기** (DNS 전파)

### 6-3. SSL 인증서 자동 발급
- Render가 자동으로 Let's Encrypt SSL 인증서 발급
- DNS 전파 완료 후 자동으로 HTTPS 활성화

---

## 7. 환경 변수 설정

### Backend 환경 변수 최종 확인

Render Dashboard → `festmoment-api` → **Environment** 탭:

```bash
# Database
DATABASE_URL=postgres://festmoment:xxxxx@xxxxx.render.com/festmoment
REDIS_URL=redis://red-xxxxx:6379

# Google API
GOOGLE_API_KEY=AIzaSy...
GEMINI_MAPS_KEY=AIzaSy...

# Naver API
NAVER_CLIENT_ID=Z_vVg4...
NAVER_CLIENT_SECRET=MrvcRe...
NAVER_TREND_CLIENT_ID=BN0uoS...
NAVER_TREND_CLIENT_SECRET=Wo4weG...

# Authentication
JWT_SECRET_KEY=<openssl rand -base64 32로 생성한 값>
GOOGLE_CLIENT_ID=345400291877-...
GOOGLE_CLIENT_SECRET=GOCSPX-...

# Frontend URL
FRONTEND_URL=https://festmoment.co.kr

# Server
PYTHONUNBUFFERED=1
PORT=8080
```

### Frontend 환경 변수 최종 확인

Render Dashboard → `festmoment-frontend` → **Environment** 탭:

```bash
VITE_API_URL=https://festmoment-api.onrender.com
VITE_GOOGLE_CLIENT_ID=345400291877-...
```

---

## 8. 데이터베이스 초기화

### 8-1. 로컬에서 PostgreSQL 접속

```bash
# PostgreSQL 클라이언트 설치 (Windows)
# https://www.postgresql.org/download/windows/

# 또는 psql 명령어로 접속
psql "postgres://festmoment:xxxxx@xxxxx.render.com/festmoment"
```

### 8-2. 초기화 스크립트 실행

프로젝트 루트의 `init_postgres.sql` 파일 실행:

```sql
-- psql 프롬프트에서
\i C:/Users/SBA/github/tour_agent/init_postgres.sql
```

또는 Render Shell 사용:
1. `festmoment-api` 서비스 → **Shell** 탭
2. 아래 명령 실행:
```bash
psql $DATABASE_URL -f init_postgres.sql
```

---

## 9. 배포 완료 확인

### 9-1. Backend API 테스트
```bash
# Health Check
curl https://festmoment-api.onrender.com/

# 축제 목록 조회
curl https://festmoment-api.onrender.com/api/festivals
```

### 9-2. Frontend 접속
- **Render URL**: `https://festmoment.onrender.com`
- **커스텀 도메인**: `https://festmoment.co.kr`

### 9-3. CORS 설정 확인

`api_server.py`에서 CORS 설정 확인:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://festmoment.co.kr",
        "https://www.festmoment.co.kr",
        "https://festmoment.onrender.com",  # Frontend Render URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

수정 후 Git 커밋 & 푸시하면 자동 재배포됨.

---

## 🎉 배포 완료!

이제 `https://festmoment.co.kr`에서 앱이 실행됩니다!

---

## ⚠️ Render 무료 티어 제한사항

### Backend (Web Service)
- ✅ **무료**
- ⚠️ **15분 미사용 시 슬립 모드**
  - 첫 요청 시 **30초~1분** 웜업 시간 소요
  - 해결: 주기적으로 핑 (Uptime Robot, Cron-job.org)
- ⚠️ **월 750시간** 제한 (약 31일)

### Frontend (Static Site)
- ✅ **완전 무료**
- ✅ **슬립 모드 없음**
- ✅ **무제한 대역폭**

### PostgreSQL
- ✅ **무료**
- ⚠️ **90일 후 자동 삭제**
  - 해결: 주기적으로 백업하거나 유료 플랜($7/월)
- ⚠️ **1GB 저장 공간**

### Redis
- ✅ **무료**
- ⚠️ **25MB 제한**
- ⚠️ **30일 후 자동 삭제**

---

## 💡 비용 절감 팁

1. **Uptime Robot으로 슬립 방지**:
   - [https://uptimerobot.com](https://uptimerobot.com)
   - 5분마다 Backend URL을 핑하여 슬립 모드 방지

2. **PostgreSQL 백업**:
   ```bash
   # 로컬로 백업
   pg_dump "postgres://festmoment:xxxxx@xxxxx.render.com/festmoment" > backup.sql
   ```

3. **Redis 대신 in-memory 캐싱**:
   - Redis가 없어도 앱이 작동하도록 코드 수정 가능

---

## 🔧 문제 해결

### Backend가 실행 안 됨
1. Render Dashboard → `festmoment-api` → **Logs** 확인
2. 환경 변수 누락 확인
3. Dockerfile 빌드 에러 확인

### Frontend가 Backend와 통신 안 됨
1. `VITE_API_URL` 환경 변수 확인
2. `api_server.py`의 CORS 설정 확인
3. Backend가 슬립 상태인지 확인

### 도메인 연결 안 됨
1. DNS 전파 확인 (10분~1시간 소요)
2. `nslookup festmoment.co.kr` 명령으로 확인
3. Render Dashboard에서 도메인 상태 확인

---

## 📞 지원

문제가 발생하면:
- [Render Docs](https://render.com/docs)
- [Render Community](https://community.render.com/)

---

**🎊 성공적인 배포를 축하합니다!**
