#!/bin/bash
# setup-gcp.sh
# FestMoment GCP Infrastructure Setup Script

set -e

# ============================================
# Configuration - MODIFY THESE VALUES
# ============================================
PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-asia-northeast3}"
ZONE="${ZONE:-asia-northeast3-a}"

# Service names
API_SERVICE="festmoment-api"
FRONTEND_SERVICE="festmoment-frontend"
CELERY_SERVICE="festmoment-celery"

# Database configuration
DB_INSTANCE="festmoment-postgres"
DB_NAME="festmoment"
DB_USER="festmoment"
DB_PASSWORD="${DB_PASSWORD:-$(openssl rand -base64 32)}"

# Redis configuration
REDIS_INSTANCE="festmoment-redis"

# VPC configuration
VPC_CONNECTOR="festmoment-vpc-connector"

echo "============================================"
echo "FestMoment GCP Setup"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "============================================"

# ============================================
# Step 1: Enable Required APIs
# ============================================
echo ""
echo "[1/8] Enabling required GCP APIs..."

gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    sqladmin.googleapis.com \
    redis.googleapis.com \
    secretmanager.googleapis.com \
    vpcaccess.googleapis.com \
    containerregistry.googleapis.com \
    --project=$PROJECT_ID

echo "APIs enabled successfully."

# ============================================
# Step 2: Create VPC Connector (for Redis access)
# ============================================
echo ""
echo "[2/8] Creating VPC Connector..."

gcloud compute networks vpc-access connectors create $VPC_CONNECTOR \
    --region=$REGION \
    --range=10.8.0.0/28 \
    --project=$PROJECT_ID \
    2>/dev/null || echo "VPC Connector already exists or created."

echo "VPC Connector created."

# ============================================
# Step 3: Create Cloud SQL Instance (PostgreSQL)
# ============================================
echo ""
echo "[3/8] Creating Cloud SQL PostgreSQL instance..."

gcloud sql instances create $DB_INSTANCE \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=$REGION \
    --storage-type=SSD \
    --storage-size=10GB \
    --storage-auto-increase \
    --availability-type=zonal \
    --project=$PROJECT_ID \
    2>/dev/null || echo "Cloud SQL instance already exists."

# Wait for instance to be ready
echo "Waiting for Cloud SQL instance to be ready..."
gcloud sql instances describe $DB_INSTANCE --project=$PROJECT_ID --format="value(state)" | grep -q "RUNNABLE" || sleep 60

# Create database and user
echo "Creating database and user..."

gcloud sql databases create $DB_NAME \
    --instance=$DB_INSTANCE \
    --project=$PROJECT_ID \
    2>/dev/null || echo "Database already exists."

gcloud sql users create $DB_USER \
    --instance=$DB_INSTANCE \
    --password=$DB_PASSWORD \
    --project=$PROJECT_ID \
    2>/dev/null || echo "User already exists."

# Get connection name
CLOUD_SQL_CONNECTION=$(gcloud sql instances describe $DB_INSTANCE --project=$PROJECT_ID --format="value(connectionName)")
echo "Cloud SQL Connection: $CLOUD_SQL_CONNECTION"

# ============================================
# Step 4: Create Memorystore (Redis) Instance
# ============================================
echo ""
echo "[4/8] Creating Memorystore Redis instance..."

gcloud redis instances create $REDIS_INSTANCE \
    --size=1 \
    --region=$REGION \
    --tier=basic \
    --project=$PROJECT_ID \
    2>/dev/null || echo "Redis instance already exists."

# Wait for Redis instance to be ready
echo "Waiting for Redis instance to be ready..."
sleep 30

# Get Redis host
REDIS_HOST=$(gcloud redis instances describe $REDIS_INSTANCE --region=$REGION --project=$PROJECT_ID --format="value(host)" 2>/dev/null || echo "")
REDIS_PORT=$(gcloud redis instances describe $REDIS_INSTANCE --region=$REGION --project=$PROJECT_ID --format="value(port)" 2>/dev/null || echo "6379")

if [ -n "$REDIS_HOST" ]; then
    echo "Redis Host: $REDIS_HOST:$REDIS_PORT"
fi

# ============================================
# Step 5: Create Secrets in Secret Manager
# ============================================
echo ""
echo "[5/8] Creating secrets in Secret Manager..."

# Function to create or update secret
create_secret() {
    local secret_name=$1
    local secret_value=$2

    echo "$secret_value" | gcloud secrets create $secret_name \
        --data-file=- \
        --replication-policy=automatic \
        --project=$PROJECT_ID \
        2>/dev/null || \
    echo "$secret_value" | gcloud secrets versions add $secret_name \
        --data-file=- \
        --project=$PROJECT_ID
}

# Create DATABASE_URL secret
DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@/$DB_NAME?host=/cloudsql/$CLOUD_SQL_CONNECTION"
create_secret "DATABASE_URL" "$DATABASE_URL"

# Create REDIS_URL secret (will be updated after Redis is ready)
if [ -n "$REDIS_HOST" ]; then
    REDIS_URL="redis://$REDIS_HOST:$REDIS_PORT/0"
    create_secret "REDIS_URL" "$REDIS_URL"
fi

echo ""
echo "============================================"
echo "IMPORTANT: Create these secrets manually:"
echo "============================================"
echo ""
echo "Run these commands with your actual values:"
echo ""
echo "gcloud secrets create GOOGLE_API_KEY --data-file=- --project=$PROJECT_ID <<< 'your-google-api-key'"
echo "gcloud secrets create NAVER_CLIENT_ID --data-file=- --project=$PROJECT_ID <<< 'your-naver-client-id'"
echo "gcloud secrets create NAVER_CLIENT_SECRET --data-file=- --project=$PROJECT_ID <<< 'your-naver-client-secret'"
echo "gcloud secrets create JWT_SECRET_KEY --data-file=- --project=$PROJECT_ID <<< 'your-jwt-secret-key'"
echo "gcloud secrets create GOOGLE_CLIENT_ID --data-file=- --project=$PROJECT_ID <<< 'your-google-client-id'"
echo ""

# ============================================
# Step 6: Grant Permissions
# ============================================
echo ""
echo "[6/8] Granting IAM permissions..."

# Get project number
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Cloud Run service account needs access to secrets
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" \
    --project=$PROJECT_ID

# Cloud Build service account needs Cloud Run admin
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/run.admin" \
    --project=$PROJECT_ID

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser" \
    --project=$PROJECT_ID

echo "Permissions granted."

# ============================================
# Step 7: Initialize Database
# ============================================
echo ""
echo "[7/8] Database initialization instructions..."
echo ""
echo "To initialize the database, run:"
echo ""
echo "gcloud sql connect $DB_INSTANCE --user=$DB_USER --database=$DB_NAME --project=$PROJECT_ID"
echo ""
echo "Then run the SQL from init_postgres.sql"
echo ""

# ============================================
# Step 8: Print Summary
# ============================================
echo ""
echo "[8/8] Setup Complete!"
echo ""
echo "============================================"
echo "DEPLOYMENT SUMMARY"
echo "============================================"
echo ""
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo ""
echo "Cloud SQL:"
echo "  Instance: $DB_INSTANCE"
echo "  Connection: $CLOUD_SQL_CONNECTION"
echo "  Database: $DB_NAME"
echo "  User: $DB_USER"
echo ""
echo "Redis:"
echo "  Instance: $REDIS_INSTANCE"
echo "  Host: $REDIS_HOST:$REDIS_PORT"
echo ""
echo "VPC Connector: $VPC_CONNECTOR"
echo ""
echo "============================================"
echo "NEXT STEPS"
echo "============================================"
echo ""
echo "1. Create API key secrets (see commands above)"
echo "2. Initialize database with init_postgres.sql"
echo "3. Deploy using Cloud Build:"
echo ""
echo "   gcloud builds submit --config=cloudbuild.yaml \\"
echo "     --substitutions=_REGION=$REGION,_CLOUD_SQL_CONNECTION=$CLOUD_SQL_CONNECTION,_VPC_CONNECTOR=$VPC_CONNECTOR \\"
echo "     --project=$PROJECT_ID"
echo ""
echo "============================================"

# Save configuration to file
cat > .gcp-config <<EOF
PROJECT_ID=$PROJECT_ID
REGION=$REGION
CLOUD_SQL_CONNECTION=$CLOUD_SQL_CONNECTION
VPC_CONNECTOR=$VPC_CONNECTOR
REDIS_HOST=$REDIS_HOST
REDIS_PORT=$REDIS_PORT
DB_INSTANCE=$DB_INSTANCE
DB_NAME=$DB_NAME
DB_USER=$DB_USER
EOF

echo "Configuration saved to .gcp-config"
