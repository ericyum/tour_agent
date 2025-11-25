#!/bin/bash
# SSL 인증서 설치 스크립트 (Let's Encrypt)
# 실행: bash deploy/setup_ssl.sh your-domain.com

set -e

DOMAIN=$1
EMAIL=${2:-admin@$DOMAIN}

if [ -z "$DOMAIN" ]; then
    echo "사용법: bash deploy/setup_ssl.sh festmoment.com [email@example.com]"
    exit 1
fi

echo "=================================="
echo "SSL 인증서 설치: $DOMAIN"
echo "=================================="

# 1. Certbot 설치
echo "[1/5] Certbot 설치..."
sudo apt-get update -qq
sudo apt-get install -y certbot

# 2. 기존 Nginx 컨테이너 중지 (포트 80 비우기)
echo "[2/5] Nginx 일시 중지..."
sudo docker-compose stop nginx

# 3. Standalone 모드로 인증서 발급
echo "[3/5] Let's Encrypt 인증서 발급..."
sudo certbot certonly --standalone \
  -d $DOMAIN \
  -d www.$DOMAIN \
  --email $EMAIL \
  --agree-tos \
  --non-interactive \
  --preferred-challenges http

# 4. Nginx SSL 설정 생성
echo "[4/5] Nginx SSL 설정 생성..."
cat > deploy/nginx-ssl.conf << 'EOF'
# HTTP → HTTPS 리다이렉트
server {
    listen 80;
    server_name DOMAIN_PLACEHOLDER www.DOMAIN_PLACEHOLDER;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS
server {
    listen 443 ssl http2;
    server_name DOMAIN_PLACEHOLDER www.DOMAIN_PLACEHOLDER;

    # SSL 인증서
    ssl_certificate /etc/letsencrypt/live/DOMAIN_PLACEHOLDER/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/DOMAIN_PLACEHOLDER/privkey.pem;

    # SSL 설정 (Mozilla Intermediate)
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_prefer_server_ciphers off;

    # 클라이언트 최대 업로드 크기
    client_max_body_size 50M;

    # Swagger UI (FastAPI 문서)
    location /docs {
        auth_basic "FestMoment API Documentation";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://api-server:8000/docs;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /openapi.json {
        auth_basic "FestMoment API Documentation";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://api-server:8000/openapi.json;
        proxy_set_header Host $host;
    }

    location /redoc {
        auth_basic "FestMoment API Documentation";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://api-server:8000/redoc;
        proxy_set_header Host $host;
    }

    # Flower (Celery 모니터링)
    location /flower/ {
        auth_basic "FestMoment Flower Dashboard";
        auth_basic_user_file /etc/nginx/.htpasswd;
        rewrite ^/flower/(.*)$ /$1 break;
        proxy_pass http://celery-flower:5555;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API 엔드포인트
    location /api/ {
        proxy_pass http://api-server:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # CORS
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization' always;

        if ($request_method = 'OPTIONS') {
            return 204;
        }
    }

    # 프론트엔드
    location / {
        proxy_pass http://api-server:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

# 도메인 교체
sed -i "s/DOMAIN_PLACEHOLDER/$DOMAIN/g" deploy/nginx-ssl.conf

# 5. docker-compose.yml 수정 (SSL 볼륨 추가)
echo "[5/5] docker-compose.yml 업데이트..."
cat >> docker-compose.yml << 'COMPOSE_EOF'

  # Nginx with SSL
  nginx-ssl:
    image: nginx:alpine
    container_name: festmoment-nginx-ssl
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deploy/nginx-ssl.conf:/etc/nginx/conf.d/default.conf:ro
      - ./deploy/.htpasswd:/etc/nginx/.htpasswd:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - api-server
      - celery-flower
    restart: unless-stopped
COMPOSE_EOF

echo ""
echo "=================================="
echo "✅ SSL 인증서 설치 완료!"
echo "=================================="
echo ""
echo "다음 단계:"
echo "  1. Nginx 재시작:"
echo "     sudo docker-compose up -d nginx-ssl"
echo ""
echo "  2. 접속 테스트:"
echo "     https://$DOMAIN/docs"
echo "     https://$DOMAIN/flower/"
echo ""
echo "인증서 자동 갱신 (90일마다):"
echo "  sudo certbot renew --nginx"
echo ""
