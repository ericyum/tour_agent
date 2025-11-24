# FestMoment GCP 배포 가이드

이 문서는 FestMoment 앱을 Google Cloud Platform (GCP)에 배포하는 방법을 설명합니다.

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Google Cloud Platform                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │   Frontend   │    │  API Server  │    │Celery Worker │           │
│  │  Cloud Run   │───▶│  Cloud Run   │◀───│  Cloud Run   │           │
│  │   (React)    │    │  (FastAPI)   │    │  (Celery)    │           │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘           │
│                              │                    │                   │
│                    ┌─────────┴────────────┬──────┘                   │
│                    │                      │                           │
│              ┌─────┴─────┐         ┌──────┴──────┐                   │
│              │ Cloud SQL │         │ Memorystore │                   │
│              │ PostgreSQL│         │   (Redis)   │                   │
│              └───────────┘         └─────────────┘                   │
│                                                                       │
│              ┌─────────────────────────────────┐                     │
│              │       Secret Manager            │                     │
│              │ (API Keys, DB Credentials, etc) │                     │
│              └─────────────────────────────────┘                     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## 사전 요구사항

### 1. Google Cloud SDK 설치

**Windows:**
```powershell
# Google Cloud SDK 다운로드 및 설치
# https://cloud.google.com/sdk/docs/install-sdk

# 또는 Chocolatey 사용
choco install gcloudsdk
```

**macOS:**
```bash
brew install google-cloud-sdk
```

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### 2. gcloud 초기 설정

```bash
# 로그인
gcloud auth login

# 프로젝트 생성 (또는 기존 프로젝트 사용)
gcloud projects create YOUR_PROJECT_ID --name="FestMoment"

# 프로젝트 설정
gcloud config set project YOUR_PROJECT_ID

# 결제 계정 연결 (필수)
# GCP Console에서 결제 설정: https://console.cloud.google.com/billing
```

## 배포 단계

### Step 1: GCP 인프라 설정

자동화 스크립트를 사용하여 필요한 인프라를 생성합니다:

```bash
# 프로젝트 ID 설정
export PROJECT_ID=your-project-id
export REGION=asia-northeast3

# 스크립트 실행
chmod +x deploy/setup-gcp.sh
./deploy/setup-gcp.sh
```

이 스크립트는 다음을 생성합니다:
- Cloud SQL (PostgreSQL) 인스턴스
- Memorystore (Redis) 인스턴스
- VPC Connector
- Secret Manager 시크릿

### Step 2: API 키 시크릿 생성

다음 시크릿을 수동으로 생성해야 합니다:

```bash
# Google API Key
echo -n 'your-google-api-key' | gcloud secrets create GOOGLE_API_KEY --data-file=-

# Naver API Keys
echo -n 'your-naver-client-id' | gcloud secrets create NAVER_CLIENT_ID --data-file=-
echo -n 'your-naver-client-secret' | gcloud secrets create NAVER_CLIENT_SECRET --data-file=-

# Naver Trend API Keys (옵션)
echo -n 'your-naver-trend-client-id' | gcloud secrets create NAVER_TREND_CLIENT_ID --data-file=-
echo -n 'your-naver-trend-client-secret' | gcloud secrets create NAVER_TREND_CLIENT_SECRET --data-file=-

# JWT Secret Key (강력한 랜덤 값 사용)
echo -n "$(openssl rand -base64 64)" | gcloud secrets create JWT_SECRET_KEY --data-file=-

# Google OAuth Client ID
echo -n 'your-google-oauth-client-id' | gcloud secrets create GOOGLE_CLIENT_ID --data-file=-
```

### Step 3: 데이터베이스 초기화

Cloud SQL에 접속하여 데이터베이스를 초기화합니다:

```bash
# Cloud SQL Proxy 사용 (권장)
# https://cloud.google.com/sql/docs/postgres/connect-auth-proxy

# 또는 직접 연결
gcloud sql connect festmoment-postgres --user=festmoment --database=festmoment

# psql 프롬프트에서 init_postgres.sql 실행
\i init_postgres.sql
```

### Step 4: 애플리케이션 배포

Cloud Build를 사용하여 배포합니다:

```bash
# .gcp-config 파일에서 설정 로드
source .gcp-config

# Cloud Build 실행
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_REGION=$REGION,_CLOUD_SQL_CONNECTION=$CLOUD_SQL_CONNECTION,_VPC_CONNECTOR=$VPC_CONNECTOR,_GOOGLE_CLIENT_ID=your-google-client-id \
  --project=$PROJECT_ID
```

### Step 5: CORS 및 OAuth 설정 업데이트

배포 후 Cloud Run URL을 확인하고 다음을 업데이트합니다:

1. **api_server.py의 CORS 설정:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://festmoment-frontend-xxxxx-an.a.run.app",  # Frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. **Google OAuth 승인된 리디렉션 URI:**
   - GCP Console > APIs & Services > Credentials
   - Frontend Cloud Run URL 추가

## 수동 배포 (개별 서비스)

### API 서버만 배포

```bash
# 이미지 빌드
gcloud builds submit --tag gcr.io/$PROJECT_ID/festmoment-api

# Cloud Run 배포
gcloud run deploy festmoment-api \
  --image gcr.io/$PROJECT_ID/festmoment-api \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-cloudsql-instances $CLOUD_SQL_CONNECTION \
  --vpc-connector $VPC_CONNECTOR \
  --set-secrets "DATABASE_URL=DATABASE_URL:latest,REDIS_URL=REDIS_URL:latest,GOOGLE_API_KEY=GOOGLE_API_KEY:latest"
```

### Frontend만 배포

```bash
# Frontend 디렉토리에서
cd frontend

# 이미지 빌드
gcloud builds submit --tag gcr.io/$PROJECT_ID/festmoment-frontend \
  --build-arg VITE_API_URL=https://festmoment-api-xxxxx-an.a.run.app

# Cloud Run 배포
gcloud run deploy festmoment-frontend \
  --image gcr.io/$PROJECT_ID/festmoment-frontend \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 256Mi
```

## 비용 최적화

### 예상 월간 비용 (최소 사용 기준)

| 서비스 | 구성 | 예상 비용 |
|--------|------|----------|
| Cloud Run (API) | 2 vCPU, 2GB RAM, min=0 | ~$10-30 |
| Cloud Run (Frontend) | 1 vCPU, 256MB RAM | ~$0-5 |
| Cloud Run (Celery) | 2 vCPU, 2GB RAM, min=1 | ~$30-50 |
| Cloud SQL | db-f1-micro | ~$10 |
| Memorystore | 1GB Basic | ~$35 |
| **총계** | | **~$85-130/월** |

### 비용 절감 팁

1. **Cloud Run min-instances=0**: 트래픽이 없을 때 0으로 스케일 다운
2. **Cloud SQL**: 개발 시 `db-f1-micro` 사용, 프로덕션에서만 업그레이드
3. **Memorystore 대안**: Redis Cloud 무료 티어 사용 가능
4. **Celery 대안**: Cloud Tasks + Cloud Functions 사용

## 모니터링 및 로그

### 로그 확인

```bash
# API 서버 로그
gcloud run services logs read festmoment-api --region=$REGION

# 실시간 로그 스트리밍
gcloud run services logs tail festmoment-api --region=$REGION
```

### Cloud Console에서 모니터링

- **Cloud Run**: https://console.cloud.google.com/run
- **Cloud SQL**: https://console.cloud.google.com/sql
- **Cloud Logging**: https://console.cloud.google.com/logs

## 자동 캐싱 설정 (Cloud Scheduler)

배포 후 자동으로 축제 데이터를 캐싱하려면 Cloud Scheduler를 설정합니다.

### Step 1: PRECACHE_API_KEY 시크릿 생성

```bash
# 강력한 랜덤 키 생성 및 저장
PRECACHE_KEY=$(openssl rand -base64 32)
echo -n "$PRECACHE_KEY" | gcloud secrets create PRECACHE_API_KEY --data-file=-

# 키 값 확인 (Cloud Scheduler 설정에 필요)
echo "PRECACHE_API_KEY: $PRECACHE_KEY"
```

### Step 2: Cloud Scheduler 작업 생성

```bash
# Cloud Scheduler API 활성화
gcloud services enable cloudscheduler.googleapis.com

# 매일 오전 3시 활성 축제 캐싱
gcloud scheduler jobs create http daily-precache-active \
  --location=$REGION \
  --schedule="0 3 * * *" \
  --uri="https://festmoment-api-$PROJECT_ID.$REGION.run.app/api/admin/trigger-precache" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{"active_only": true, "num_reviews": 50, "api_key": "YOUR_PRECACHE_API_KEY"}' \
  --time-zone="Asia/Seoul"

# 매주 일요일 오전 2시 전체 축제 캐싱
gcloud scheduler jobs create http weekly-precache-all \
  --location=$REGION \
  --schedule="0 2 * * 0" \
  --uri="https://festmoment-api-$PROJECT_ID.$REGION.run.app/api/admin/trigger-precache" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{"active_only": false, "num_reviews": 50, "api_key": "YOUR_PRECACHE_API_KEY"}' \
  --time-zone="Asia/Seoul"
```

> **중요**: `YOUR_PRECACHE_API_KEY`를 Step 1에서 생성한 실제 키 값으로 교체하세요.

### Step 3: 수동 테스트

```bash
# 스케줄러 작업 즉시 실행 (테스트)
gcloud scheduler jobs run daily-precache-active --location=$REGION

# 로그 확인
gcloud run services logs read festmoment-api --region=$REGION --limit=50
```

### Cloud Scheduler 관리

```bash
# 작업 목록 확인
gcloud scheduler jobs list --location=$REGION

# 작업 일시 중지
gcloud scheduler jobs pause daily-precache-active --location=$REGION

# 작업 재개
gcloud scheduler jobs resume daily-precache-active --location=$REGION

# 작업 삭제
gcloud scheduler jobs delete daily-precache-active --location=$REGION
```

## 커스텀 도메인 설정

### 1. 도메인 매핑

```bash
# Frontend
gcloud run domain-mappings create \
  --service festmoment-frontend \
  --domain www.festmoment.com \
  --region $REGION

# API
gcloud run domain-mappings create \
  --service festmoment-api \
  --domain api.festmoment.com \
  --region $REGION
```

### 2. DNS 설정

Cloud Run에서 제공하는 CNAME 레코드를 도메인 DNS에 추가합니다.

## 문제 해결

### 일반적인 문제

**1. Cloud SQL 연결 실패**
```bash
# VPC Connector 상태 확인
gcloud compute networks vpc-access connectors describe $VPC_CONNECTOR --region=$REGION

# Cloud SQL 인스턴스 상태 확인
gcloud sql instances describe festmoment-postgres
```

**2. Secret Manager 권한 오류**
```bash
# 서비스 계정에 권한 부여
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

**3. Redis 연결 실패**
- VPC Connector가 올바르게 설정되었는지 확인
- Redis 인스턴스가 같은 리전에 있는지 확인

### 지원

문제가 지속되면 GCP 문서를 참조하세요:
- [Cloud Run 문서](https://cloud.google.com/run/docs)
- [Cloud SQL 문서](https://cloud.google.com/sql/docs)
- [Memorystore 문서](https://cloud.google.com/memorystore/docs)
