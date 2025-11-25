# 🚀 FestMoment GCP 배포 가이드

## 📋 목차
1. [사전 준비](#사전-준비)
2. [GCP VM 생성](#gcp-vm-생성)
3. [자동 배포](#자동-배포)
4. [접속 및 확인](#접속-및-확인)
5. [문제 해결](#문제-해결)

---

## 사전 준비

### 필요한 것
- Google Cloud Platform 계정
- GCP 프로젝트 생성
- gcloud CLI 설치 (로컬)
- API 키:
  - Google Gemini API Key
  - Naver Search API (Client ID/Secret)
  - Naver DataLab API (Client ID/Secret)

### 권장 VM 스펙
- **머신 유형**: e2-standard-2 (vCPU 2개, 메모리 8GB)
- **부팅 디스크**: Ubuntu 22.04 LTS, 50GB
- **지역**: asia-northeast3 (서울)
- **예상 비용**: 월 $50~70

---

## GCP VM 생성

### 1. gcloud CLI로 VM 생성

```bash
# GCP 로그인
gcloud auth login

# 프로젝트 설정
gcloud config set project YOUR_PROJECT_ID

# VM 생성
gcloud compute instances create festmoment-vm \
  --machine-type=e2-standard-2 \
  --zone=asia-northeast3-a \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-standard \
  --tags=http-server,https-server

# 방화벽 규칙 (HTTP 포트 80)
gcloud compute firewall-rules create allow-http \
  --allow tcp:80 \
  --target-tags=http-server \
  --description="Allow HTTP traffic"

# VM 외부 IP 확인
gcloud compute instances describe festmoment-vm \
  --zone=asia-northeast3-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

### 2. GCP 콘솔로 VM 생성 (GUI)

1. GCP Console → Compute Engine → VM 인스턴스
2. "인스턴스 만들기" 클릭
3. 설정:
   - 이름: `festmoment-vm`
   - 리전: `asia-northeast3 (서울)`
   - 영역: `asia-northeast3-a`
   - 머신 유형: `e2-standard-2`
   - 부팅 디스크: Ubuntu 22.04 LTS, 50GB
   - 방화벽: ✅ HTTP 트래픽 허용
4. "만들기" 클릭

---

## 자동 배포

### 1. VM에 SSH 접속

```bash
# gcloud CLI
gcloud compute ssh festmoment-vm --zone=asia-northeast3-a

# 또는 GCP 콘솔에서 "SSH" 버튼 클릭
```

### 2. 프로젝트 클론

```bash
# Git 설치
sudo apt-get update
sudo apt-get install -y git

# 프로젝트 클론
git clone https://github.com/YOUR_USERNAME/tour_agent.git
cd tour_agent
```

### 3. 환경 변수 설정

```bash
# .env 파일 생성
nano .env
```

**.env 내용**:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret
NAVER_TREND_CLIENT_ID=your_naver_trend_client_id
NAVER_TREND_CLIENT_SECRET=your_naver_trend_client_secret
REDIS_URL=redis://redis:6379/0
DATABASE_URL=postgresql://festmoment:festmoment_password@postgres:5432/festmoment
```

저장: `Ctrl + O` → `Enter` → `Ctrl + X`

### 4. 자동 배포 스크립트 실행

```bash
# 실행 권한 부여
chmod +x deploy/deploy_gcp.sh

# 배포 실행
bash deploy/deploy_gcp.sh
```

스크립트가 자동으로:
- Docker 및 Docker Compose 설치
- Basic Auth 계정 생성 (입력 필요)
- 전체 스택 빌드 및 실행

---

## 접속 및 확인

### 외부 IP 확인

```bash
# VM에서 실행
curl ifconfig.me

# 또는 gcloud CLI
gcloud compute instances describe festmoment-vm \
  --zone=asia-northeast3-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

### 서비스 접속

| 서비스 | URL | 인증 |
|--------|-----|------|
| **Swagger UI** | `http://YOUR_IP/docs` | ✅ Basic Auth |
| **Flower** | `http://YOUR_IP/flower/` | ✅ Basic Auth |
| **API** | `http://YOUR_IP/api/...` | ❌ 없음 |

**Basic Auth 계정**: 배포 시 설정한 사용자명/비밀번호

### 서비스 상태 확인

```bash
# 모든 컨테이너 상태
sudo docker-compose ps

# 로그 확인
sudo docker-compose logs -f api-server
sudo docker-compose logs -f celery-worker
sudo docker-compose logs -f celery-beat
sudo docker-compose logs -f celery-flower

# 캐시 통계 확인
sudo docker-compose exec celery-worker python -c "
import sys; sys.path.insert(0, '/app')
from src.infrastructure.cache_manager import get_cache_stats
print(get_cache_stats())
"
```

### Celery 자동 캐싱 확인

배포 후 자동으로:
- **매일 오전 3시**: 활성 축제 캐싱
- **매주 일요일 오전 2시**: 전체 축제 캐싱

**Flower에서 확인**:
1. `http://YOUR_IP/flower/` 접속
2. Tasks 탭에서 스케줄된 작업 확인

---

## 문제 해결

### 1. 포트 80 접속 불가

**증상**: 브라우저에서 연결 거부

**해결**:
```bash
# 방화벽 규칙 확인
gcloud compute firewall-rules list | grep allow-http

# 없으면 생성
gcloud compute firewall-rules create allow-http \
  --allow tcp:80 \
  --target-tags=http-server

# VM에 태그 추가
gcloud compute instances add-tags festmoment-vm \
  --tags=http-server \
  --zone=asia-northeast3-a
```

### 2. 컨테이너 재시작 반복

**증상**: `docker-compose ps`에서 컨테이너가 계속 재시작

**해결**:
```bash
# 로그 확인
sudo docker-compose logs api-server

# 일반적인 원인:
# - .env 파일 누락 → 확인 후 재시작
# - 메모리 부족 → VM 스펙 업그레이드
# - 포트 충돌 → 다른 서비스 중지
```

### 3. Basic Auth 비밀번호 변경

```bash
# 새 비밀번호 설정
sudo apt-get install -y apache2-utils
htpasswd -c deploy/.htpasswd admin

# Nginx 재시작
sudo docker-compose restart nginx
```

### 4. SSL 인증서 추가 (HTTPS)

**도메인이 있는 경우**:

```bash
# Certbot 설치
sudo apt-get install -y certbot

# 인증서 발급
sudo certbot certonly --standalone \
  -d your-domain.com \
  --email your-email@example.com

# Nginx 설정 수정 필요 (443 포트 추가)
```

### 5. 전체 재배포

```bash
cd tour_agent

# 모든 컨테이너 중지 및 삭제
sudo docker-compose down -v

# 캐시 제거
sudo docker-compose exec celery-worker python -c "
import sys; sys.path.insert(0, '/app')
from src.infrastructure.cache_manager import clear_cache
clear_cache()
"

# 재배포
bash deploy/deploy_gcp.sh
```

---

## 비용 절감 팁

### 1. 개발 시에만 VM 실행

```bash
# VM 중지 (과금 중지)
gcloud compute instances stop festmoment-vm --zone=asia-northeast3-a

# VM 재시작
gcloud compute instances start festmoment-vm --zone=asia-northeast3-a
```

### 2. 스케줄러 조정

매일 캐싱이 불필요한 경우, `src/celery/app.py` 수정:

```python
beat_schedule={
    # 매주 1회로 변경
    "weekly-all-festivals-precache": {
        "task": "weekly_precache_all_festivals",
        "schedule": crontab(hour=2, minute=0, day_of_week=0),
    },
}
```

---

## 유용한 명령어

```bash
# 전체 재시작
sudo docker-compose restart

# 특정 서비스만 재시작
sudo docker-compose restart api-server

# 실시간 로그
sudo docker-compose logs -f --tail=100

# 디스크 용량 확인
df -h

# 메모리 사용량 확인
free -h

# Docker 디스크 정리
sudo docker system prune -a
```

---

## 📞 지원

문제가 발생하면:
1. 로그 확인: `sudo docker-compose logs`
2. GitHub Issues 등록
3. 이메일: ericyum9196@gmail.com

---

**배포 완료! 🎉**
