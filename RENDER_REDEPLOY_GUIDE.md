# 🔄 FestMoment Render 재배포 가이드

기존 배포를 삭제하고 처음부터 다시 배포하는 가이드입니다.

---

## 📋 재배포가 필요한 이유

기존 배포에서 발견된 문제들:
1. ❌ **검색 기능 실패** - 데이터베이스 초기화 누락
2. ❌ **Google OAuth 실패** - 하드코딩된 localhost URL 문제 + OAuth 승인 도메인 미설정

✅ **해결 완료**: 프론트엔드 코드 수정 완료 (하드코딩된 URL → 환경변수)

---

## 🗑️ Step 1: 기존 배포 삭제

### 1-1. Render Dashboard 접속
1. [https://dashboard.render.com](https://dashboard.render.com) 로그인
2. 왼쪽 메뉴에서 **All Services** 클릭

### 1-2. 기존 서비스 삭제
아래 서비스들을 하나씩 삭제합니다:

**Backend API 삭제:**
1. `festmoment-api` 서비스 클릭
2. **Settings** 탭으로 이동
3. 맨 아래 **Delete Web Service** 클릭
4. 서비스 이름 입력 후 **Delete** 확인

**Frontend 삭제:**
1. `festmoment-frontend` 서비스 클릭
2. **Settings** 탭 → **Delete Static Site**

**PostgreSQL 삭제:**
1. `festmoment-db` 데이터베이스 클릭
2. **Settings** 탭 → **Delete Database**

**Redis 삭제 (있는 경우):**
1. `festmoment-redis` 클릭
2. **Settings** 탭 → **Delete Redis Instance**

⚠️ **주의**: 모든 데이터가 영구 삭제됩니다!

---

## 🛠️ Step 2: Google Cloud Console 설정

### 2-1. OAuth 승인 도메인 추가
1. [Google Cloud Console](https://console.cloud.google.com/apis/credentials) 접속
2. 프로젝트 선택
3. **OAuth 2.0 클라이언트 ID** 클릭 (Client ID: `345400291877-...`)
4. 다음 항목들을 추가:

**승인된 JavaScript 원본:**
```
https://festmoment.co.kr
https://www.festmoment.co.kr
https://festmoment.onrender.com
https://festmoment-frontend.onrender.com
```

**승인된 리디렉션 URI:**
```
https://festmoment.co.kr
https://www.festmoment.co.kr
https://festmoment.onrender.com
https://festmoment-frontend.onrender.com
```

5. **저장** 클릭
6. ⏰ 변경사항 적용까지 최대 5분 소요

---

## 🚀 Step 3: Render 재배포

### 3-1. PostgreSQL 생성
1. **New +** → **PostgreSQL**
2. 설정:
   ```
   Name: festmoment-db
   Database: festmoment
   User: festmoment
   Region: Singapore
   PostgreSQL Version: 16
   Instance Type: Free
   ```
3. **Create Database** 클릭
4. **Internal Database URL** 복사 (나중에 사용)
   ```
   postgres://festmoment:xxxxx@xxxxx.render.com/festmoment
   ```

### 3-2. Backend API 배포
1. **New +** → **Web Service**
2. **Build and deploy from a Git repository**
3. 저장소: `ericyum/FestMoment` (또는 `ericyum/tour_agent`)

**서비스 설정:**
```
Name: festmoment-api
Region: Singapore
Branch: main
Root Directory: (비워두기)
Runtime: Docker
Dockerfile Path: ./Dockerfile
Instance Type: Free
```

**환경 변수 설정:**
```bash
# Database
DATABASE_URL=<Step 3-1에서 복사한 PostgreSQL URL>

# Google API
GOOGLE_API_KEY=AIzaSyDgCWDdLjZZ14PtRyk6Mag8l2p3ig0cS8E
GEMINI_MAPS_KEY=AIzaSyAdgOb1CUMalwFMoE8Qzr7S3F-GWs7ITAM

# Naver API
NAVER_CLIENT_ID=Z_vVg4YrrgpvqG8M3No5
NAVER_CLIENT_SECRET=MrvcReRaHd
NAVER_TREND_CLIENT_ID=BN0uoSnoytAMyULvVMqk
NAVER_TREND_CLIENT_SECRET=Wo4weGsjTP

# Authentication
JWT_SECRET_KEY=<강력한 랜덤 문자열 - 아래 명령어로 생성>
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Frontend URL
FRONTEND_URL=https://festmoment.co.kr

# Server
PYTHONUNBUFFERED=1
PORT=8080
```

**JWT_SECRET_KEY 생성 방법:**
```bash
# 로컬 터미널에서 실행
openssl rand -base64 32
```

4. **Create Web Service** 클릭
5. 빌드 완료 대기 (약 5-10분)
6. **Backend URL 복사** (예: `https://festmoment-api.onrender.com`)

### 3-3. Frontend 배포
1. **New +** → **Static Site**
2. 저장소: `ericyum/FestMoment` 선택

**사이트 설정:**
```
Name: festmoment-frontend
Region: Singapore
Branch: main
Root Directory: frontend
Build Command: npm install && npm run build
Publish Directory: dist
```

**환경 변수 설정:**
```bash
VITE_API_URL=<Step 3-2에서 복사한 Backend URL>
VITE_GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
```

예시:
```bash
VITE_API_URL=https://festmoment-api.onrender.com
```

3. **Create Static Site** 클릭
4. 빌드 완료 대기 (약 3-5분)

---

## 🗄️ Step 4: 데이터베이스 초기화

### 4-1. Render Shell 사용 (권장)
1. Render Dashboard → `festmoment-api` 서비스
2. **Shell** 탭 클릭
3. 아래 명령어 실행:
```bash
psql $DATABASE_URL -f init_postgres.sql
```

### 4-2. 로컬에서 초기화 (대안)
```bash
# PostgreSQL 클라이언트 설치 필요
psql "postgres://festmoment:xxxxx@xxxxx.render.com/festmoment" -f init_postgres.sql
```

**⚠️ 중요**: `init_postgres.sql` 파일에 축제 데이터가 포함되어 있는지 확인!

---

## 🌐 Step 5: 커스텀 도메인 연결 (festmoment.co.kr)

### 5-1. Render에서 도메인 추가
1. Frontend 서비스 (`festmoment-frontend`) 선택
2. **Settings** → **Custom Domains**
3. **Add Custom Domain** 클릭
4. 도메인 입력:
   - `festmoment.co.kr`
   - `www.festmoment.co.kr`
5. Render가 제공하는 **CNAME 레코드** 복사

### 5-2. 가비아 DNS 설정
1. [가비아](https://www.gabia.com/) 로그인
2. **My가비아** → **서비스 관리** → **도메인**
3. `festmoment.co.kr` → **DNS 정보**
4. **레코드 추가**:

| 타입 | 호스트 | 값/위치 | TTL |
|------|--------|---------|-----|
| CNAME | @ | festmoment.onrender.com | 3600 |
| CNAME | www | festmoment.onrender.com | 3600 |

5. 저장 후 **10분~1시간 대기** (DNS 전파)

---

## ✅ Step 6: 배포 확인

### 6-1. Backend API 테스트
```bash
# Health Check
curl https://festmoment-api.onrender.com/

# 축제 검색 테스트
curl https://festmoment-api.onrender.com/api/festivals/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"area":"전체","page":1}'
```

**예상 결과**:
- Health Check: `{"status":"ok","service":"FestMoment API",...}`
- 축제 검색: `{"festivals":[...],"total":123,...}`

### 6-2. Frontend 접속
- **Render URL**: `https://festmoment.onrender.com`
- **커스텀 도메인**: `https://festmoment.co.kr`

### 6-3. 기능 테스트 체크리스트
- [ ] 축제 검색 기능 작동
- [ ] 축제 상세 정보 표시
- [ ] Google 로그인 작동
- [ ] 일반 로그인/회원가입 작동
- [ ] 감성 분석, 트렌드 등 AI 기능 작동

---

## 🔧 문제 해결

### Backend가 실행 안 됨
1. **Logs 확인**: Render Dashboard → `festmoment-api` → **Logs**
2. **환경 변수 확인**: 모든 필수 변수가 설정되었는지 확인
3. **Dockerfile 확인**: 빌드 에러가 없는지 확인

### Frontend가 Backend와 통신 안 됨
1. **환경 변수 확인**:
   - `VITE_API_URL`이 정확한 Backend URL인지 확인
2. **CORS 설정 확인**:
   - `api_server.py:97-110`에 Frontend URL이 포함되어 있는지 확인
3. **Backend 상태 확인**:
   - Backend가 슬립 모드인지 확인 (첫 요청 시 30초 소요)

### 검색 결과가 없습니다
1. **데이터베이스 확인**:
   ```bash
   # Render Shell에서 실행
   psql $DATABASE_URL -c "SELECT COUNT(*) FROM festivals;"
   ```
2. 결과가 0이면 **Step 4** 데이터베이스 초기화 재실행

### Google 로그인 실패
1. **OAuth 도메인 확인**: Google Cloud Console에서 도메인이 정확히 추가되었는지 확인
2. **5분 대기**: OAuth 설정 변경 후 최대 5분 소요
3. **브라우저 캐시 삭제**: 시크릿 모드에서 재시도
4. **환경 변수 확인**:
   - Backend: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`
   - Frontend: `VITE_GOOGLE_CLIENT_ID`
   - 모두 동일한 Client ID인지 확인

### 도메인 연결 안 됨
1. **DNS 전파 확인** (10분~1시간 소요):
   ```bash
   nslookup festmoment.co.kr
   ```
2. **Render Dashboard 확인**: Custom Domain 상태가 "Verified"인지 확인
3. **SSL 인증서 확인**: Let's Encrypt 자동 발급까지 시간 소요 가능

---

## 📊 재배포 후 체크리스트

### 필수 확인 사항
- [ ] Backend API Health Check 통과
- [ ] 축제 검색 결과 표시됨
- [ ] Google OAuth 로그인 성공
- [ ] 커스텀 도메인 접속 가능
- [ ] HTTPS 인증서 발급 완료

### 선택 확인 사항
- [ ] 감성 분석 기능 작동
- [ ] 워드클라우드 생성 기능 작동
- [ ] AI 렌더링 기능 작동
- [ ] Q&A 게시판 기능 작동

---

## 🎉 완료!

이제 `https://festmoment.co.kr`에서 완전히 작동하는 앱을 확인할 수 있습니다!

**주요 변경사항:**
✅ 프론트엔드 하드코딩된 localhost URL 제거
✅ 환경변수 기반 API URL 설정
✅ Google OAuth 승인 도메인 설정
✅ 데이터베이스 초기화 가이드 명확화

---

## 💡 추가 팁

### Render 무료 티어 슬립 방지
1. [Uptime Robot](https://uptimerobot.com) 가입
2. 모니터 추가: `https://festmoment-api.onrender.com/`
3. 체크 간격: 5분마다
4. 효과: Backend 슬립 모드 방지

### 데이터베이스 백업 (90일 제한)
```bash
# 로컬로 백업
pg_dump "postgres://festmoment:xxxxx@xxxxx.render.com/festmoment" > backup_$(date +%Y%m%d).sql
```

### 로그 모니터링
- Render Dashboard → 각 서비스 → **Logs** 탭
- 실시간 로그 확인 가능
