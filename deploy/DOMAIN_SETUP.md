# 🌐 도메인 설정 가이드

## 개요

IP 주소 대신 `festmoment.com` 같은 도메인으로 접속하려면 다음 단계를 따르세요.

**변경 전**: `http://34.64.123.45/docs`
**변경 후**: `https://festmoment.com/docs`

---

## 📋 전체 프로세스

```
1. 도메인 구매 (festmoment.com)
   ↓
2. GCP 고정 IP 할당 (34.64.123.45)
   ↓
3. DNS A 레코드 설정 (festmoment.com → 34.64.123.45)
   ↓
4. DNS 전파 대기 (1~24시간)
   ↓
5. SSL 인증서 설치 (Let's Encrypt)
   ↓
6. HTTPS 접속 (https://festmoment.com)
```

---

## 1️⃣ 도메인 구매

### 추천 업체

| 업체 | 장점 | 가격 (.com) |
|------|------|------------|
| **Cloudflare** | CDN 포함, 최저가 | $10/년 |
| **Namecheap** | 저렴, 프라이버시 보호 무료 | $13/년 |
| **가비아** | 한국어, 국내 결제 | 15,000원/년 |
| **Google Domains** | GCP 연동 편리 | $12/년 |

### 구매 과정 (Cloudflare 예시)

1. https://www.cloudflare.com/products/registrar/ 접속
2. 도메인 검색: `festmoment.com`
3. 구매 및 결제
4. Cloudflare 계정에 도메인 추가

---

## 2️⃣ GCP 고정 IP 할당

**이유**: VM 재시작 시 IP가 바뀌는 것 방지

```bash
# SSH로 GCP VM 접속
gcloud compute ssh festmoment-vm --zone=asia-northeast3-a

# 현재 IP 확인
CURRENT_IP=$(curl -s ifconfig.me)
echo "현재 IP: $CURRENT_IP"

# 고정 IP로 예약 (로컬에서 실행)
gcloud compute addresses create festmoment-ip \
  --addresses $CURRENT_IP \
  --region=asia-northeast3

# 확인
gcloud compute addresses describe festmoment-ip \
  --region=asia-northeast3 \
  --format='get(address)'

# 출력: 34.64.123.45 (이 IP를 기억하세요!)
```

---

## 3️⃣ DNS A 레코드 설정

### Cloudflare

1. Cloudflare 대시보드 → DNS → Records
2. "Add record" 클릭
3. 설정:
   ```
   Type: A
   Name: @
   IPv4 address: 34.64.123.45
   Proxy status: ✅ Proxied (주황색 구름)
   TTL: Auto
   ```
4. "Save" 클릭
5. `www` 서브도메인도 추가:
   ```
   Type: A
   Name: www
   IPv4 address: 34.64.123.45
   Proxy status: ✅ Proxied
   ```

### 가비아

1. My가비아 → 도메인 관리
2. 도메인 선택 → DNS 정보 → 설정
3. "레코드 추가" 클릭
4. 설정:
   ```
   타입: A
   호스트: @
   값/위치: 34.64.123.45
   TTL: 3600
   ```
5. `www` 레코드도 추가

### Namecheap

1. Domain List → Manage → Advanced DNS
2. "ADD NEW RECORD" 클릭
3. 설정:
   ```
   Type: A Record
   Host: @
   Value: 34.64.123.45
   TTL: Automatic
   ```

---

## 4️⃣ DNS 전파 확인

**대기 시간**: 보통 1~2시간, 최대 24시간

### 확인 방법

```bash
# 명령어로 확인
nslookup festmoment.com

# 출력 예시:
# Server:  8.8.8.8
# Address: 8.8.8.8#53
#
# Non-authoritative answer:
# Name:    festmoment.com
# Address: 34.64.123.45  ← 이게 나와야 함!
```

### 온라인 도구

- https://dnschecker.org
- 도메인 입력: `festmoment.com`
- Type: `A`
- ✅ 전 세계에서 IP가 보이면 완료

---

## 5️⃣ SSL 인증서 설치

### 자동 설치 스크립트 사용

```bash
# GCP VM에서 실행
cd tour_agent

# 실행 권한 부여
chmod +x deploy/setup_ssl.sh

# SSL 설치 (이메일은 인증서 만료 알림용)
bash deploy/setup_ssl.sh festmoment.com admin@festmoment.com
```

스크립트가 자동으로:
1. ✅ Certbot 설치
2. ✅ Let's Encrypt 인증서 발급 (무료, 90일 유효)
3. ✅ Nginx SSL 설정 생성
4. ✅ HTTP → HTTPS 리다이렉트 설정

### 수동 설치 (스크립트 없이)

```bash
# 1. Certbot 설치
sudo apt-get update
sudo apt-get install -y certbot

# 2. Nginx 일시 중지 (포트 80 비우기)
sudo docker-compose stop nginx

# 3. 인증서 발급
sudo certbot certonly --standalone \
  -d festmoment.com \
  -d www.festmoment.com \
  --email admin@festmoment.com \
  --agree-tos

# 4. 인증서 확인
sudo ls -la /etc/letsencrypt/live/festmoment.com/
# fullchain.pem, privkey.pem 파일이 있어야 함

# 5. Nginx 설정 수정
sudo nano deploy/nginx.conf
# server_name을 festmoment.com으로 변경

# 6. SSL 볼륨 마운트 추가 (docker-compose.yml)
# nginx 서비스에 추가:
#   volumes:
#     - /etc/letsencrypt:/etc/letsencrypt:ro

# 7. Nginx 재시작
sudo docker-compose up -d nginx
```

---

## 6️⃣ 방화벽 설정 (HTTPS 포트 443)

```bash
# GCP 방화벽 규칙 생성
gcloud compute firewall-rules create allow-https \
  --allow tcp:443 \
  --target-tags=http-server \
  --description="Allow HTTPS traffic"
```

---

## 7️⃣ 접속 테스트

### HTTP → HTTPS 자동 리다이렉트

```bash
# HTTP로 접속해도 자동으로 HTTPS로 이동
curl -I http://festmoment.com
# Location: https://festmoment.com/

# HTTPS 접속
curl -I https://festmoment.com
# HTTP/2 200 OK
```

### 브라우저 접속

```
✅ https://festmoment.com/docs       # Swagger UI
✅ https://festmoment.com/flower/    # Flower
✅ https://festmoment.com/api/...    # API
```

**자물쇠 아이콘** 🔒이 표시되면 SSL 성공!

---

## 🔄 SSL 인증서 자동 갱신

Let's Encrypt 인증서는 **90일**마다 갱신 필요.

### Cron으로 자동 갱신 설정

```bash
# Crontab 편집
sudo crontab -e

# 매일 오전 3시에 갱신 시도 (실제로는 만료 30일 전부터만 갱신됨)
0 3 * * * certbot renew --quiet --post-hook "docker-compose restart nginx"
```

### 수동 갱신 테스트

```bash
# Dry run (실제 갱신 안 함, 테스트만)
sudo certbot renew --dry-run

# 실제 갱신
sudo certbot renew

# Nginx 재시작
sudo docker-compose restart nginx
```

---

## 🎯 전체 설정 완료 확인

### ✅ 체크리스트

- [ ] 도메인 구매 완료
- [ ] GCP 고정 IP 할당
- [ ] DNS A 레코드 설정
- [ ] DNS 전파 확인 (`nslookup`)
- [ ] SSL 인증서 발급
- [ ] HTTPS 포트 (443) 방화벽 개방
- [ ] HTTP → HTTPS 리다이렉트 작동
- [ ] 브라우저에서 자물쇠 아이콘 표시
- [ ] SSL 자동 갱신 Cron 설정

### 최종 접속 URL

```
Swagger UI:  https://festmoment.com/docs
Flower:      https://festmoment.com/flower/
API:         https://festmoment.com/api/festivals
```

---

## 🚨 문제 해결

### DNS 전파가 안 됨

```bash
# DNS 캐시 초기화 (Windows)
ipconfig /flushdns

# DNS 캐시 초기화 (macOS)
sudo dscacheutil -flushcache

# DNS 캐시 초기화 (Linux)
sudo systemd-resolve --flush-caches
```

### SSL 인증서 발급 실패

**원인**: 포트 80이 이미 사용 중

```bash
# Nginx 중지 후 재시도
sudo docker-compose stop nginx
sudo certbot certonly --standalone -d festmoment.com
```

### "Your connection is not private" 경고

**원인**: 인증서가 제대로 설치되지 않음

```bash
# 인증서 확인
sudo certbot certificates

# Nginx 로그 확인
sudo docker-compose logs nginx
```

---

## 💰 비용

| 항목 | 비용 |
|------|------|
| 도메인 (.com) | $10~15/년 |
| SSL 인증서 | **무료** (Let's Encrypt) |
| GCP 고정 IP | $3.65/월 (~$44/년) |
| **합계** | **~$60/년** |

---

## 📞 지원

도메인 설정 중 문제가 있으면:
- GitHub Issues
- Email: admin@festmoment.com

---

**도메인 설정 완료! 🎉**

이제 `https://festmoment.com`으로 접속 가능합니다!
