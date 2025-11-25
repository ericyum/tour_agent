#!/bin/bash
# GCP 배포 스크립트
# 실행: bash deploy/deploy_gcp.sh

set -e  # 오류 발생 시 중단

echo "=================================="
echo "FestMoment GCP 배포 시작"
echo "=================================="

# 1. 시스템 패키지 업데이트
echo "[1/8] 시스템 업데이트..."
sudo apt-get update -qq

# 2. Docker 설치 확인 및 설치
echo "[2/8] Docker 확인 및 설치..."
if ! command -v docker &> /dev/null; then
    echo "Docker 설치 중..."
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER
    echo "✅ Docker 설치 완료"
else
    echo "✅ Docker 이미 설치됨"
fi

# 3. Docker Compose 설치 확인 및 설치
echo "[3/8] Docker Compose 확인 및 설치..."
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose 설치 중..."
    sudo apt-get install -y docker-compose
    echo "✅ Docker Compose 설치 완료"
else
    echo "✅ Docker Compose 이미 설치됨"
fi

# 4. Apache Utils 설치 (htpasswd 명령어)
echo "[4/8] Apache Utils 설치..."
sudo apt-get install -y apache2-utils

# 5. Basic Auth 파일 생성
echo "[5/8] Basic Auth 파일 생성..."
mkdir -p deploy
if [ -f "deploy/.htpasswd" ]; then
    echo "기존 .htpasswd 파일이 있습니다. 덮어쓰시겠습니까? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm deploy/.htpasswd
    fi
fi

if [ ! -f "deploy/.htpasswd" ]; then
    echo "관리자 계정 생성 (기본값: admin)"
    read -p "사용자명 [admin]: " USERNAME
    USERNAME=${USERNAME:-admin}

    htpasswd -c deploy/.htpasswd "$USERNAME"
    echo "✅ Basic Auth 파일 생성 완료"
else
    echo "✅ 기존 .htpasswd 사용"
fi

# 6. 환경 변수 파일 확인
echo "[6/8] 환경 변수 확인..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env 파일이 없습니다!"
    echo "다음 내용으로 .env 파일을 생성하세요:"
    echo ""
    echo "GOOGLE_API_KEY=your_google_api_key"
    echo "NAVER_CLIENT_ID=your_naver_client_id"
    echo "NAVER_CLIENT_SECRET=your_naver_client_secret"
    echo "NAVER_TREND_CLIENT_ID=your_trend_client_id"
    echo "NAVER_TREND_CLIENT_SECRET=your_trend_client_secret"
    echo "REDIS_URL=redis://redis:6379/0"
    echo "DATABASE_URL=postgresql://festmoment:festmoment_password@postgres:5432/festmoment"
    echo ""
    exit 1
else
    echo "✅ .env 파일 존재"
fi

# 7. Docker Compose로 전체 스택 실행
echo "[7/8] Docker Compose 실행..."
sudo docker-compose down 2>/dev/null || true
sudo docker-compose up -d --build

# 8. 서비스 상태 확인
echo "[8/8] 서비스 상태 확인..."
sleep 5
sudo docker-compose ps

echo ""
echo "=================================="
echo "✅ FestMoment 배포 완료!"
echo "=================================="
echo ""
echo "접속 정보:"
echo "  - Swagger UI: http://$(curl -s ifconfig.me)/docs"
echo "  - Flower:     http://$(curl -s ifconfig.me)/flower/"
echo "  - API:        http://$(curl -s ifconfig.me)/api/..."
echo ""
echo "Basic Auth 계정: $USERNAME / (입력한 비밀번호)"
echo ""
echo "로그 확인:"
echo "  sudo docker-compose logs -f api-server"
echo "  sudo docker-compose logs -f celery-worker"
echo "  sudo docker-compose logs -f celery-beat"
echo ""
