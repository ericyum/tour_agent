# Guest User Tracking Implementation

## Overview
This document describes the guest user tracking system implemented for the FestMoment application.

## Features

### 1. Guest User Identification
- **Guest users** (non-authenticated visitors) are assigned unique identifiers (e.g., "Guest1", "Guest2", "Guest3456")
- Guest IDs are stored in `localStorage` and persist across sessions
- Guest IDs are generated randomly with a number between 1-10000

### 2. User Behavior Tracking
All user events are tracked with the following information:
- **Authenticated users**: `user_id`, `username`
- **Guest users**: `guest_id` (e.g., "Guest123")

### 3. Feature Access Control

#### Accessible to ALL Users (including guests):
- ✅ Festival search and browsing
- ✅ AI sentiment analysis
- ✅ WordCloud generation
- ✅ Travel course planning
- ✅ Viewing Q&A posts
- ✅ All navigation and viewing features

#### Restricted to Authenticated Users Only:
- ❌ Submitting feedback (thumbs up/down)
- ❌ Posting Q&A questions
- ❌ Answering Q&A questions
- ❌ Editing/deleting Q&A content

### 4. Guest Experience
When a guest user tries to submit feedback or post to Q&A:
1. A **login prompt** appears
2. The prompt explains that feedback and Q&A require login
3. Options to **Login** or **Register** are provided
4. Guest can continue browsing without logging in

## Technical Implementation

### Frontend (`frontend/src/lib/analytics.ts`)
```typescript
// Get guest user ID (e.g., "Guest123")
export const getGuestUserId = (): string => {
  let guestId = localStorage.getItem('guest_user_id');
  if (!guestId) {
    const guestNumber = Math.floor(Math.random() * 10000) + 1;
    guestId = `Guest${guestNumber}`;
    localStorage.setItem('guest_user_id', guestId);
  }
  return guestId;
};

// Get user identifier (authenticated or guest)
export const getUserIdentifier = () => {
  const authState = useAuthStore.getState();

  if (authState.isAuthenticated && authState.user) {
    return {
      userId: authState.user.id,
      guestId: null,
      username: authState.user.username,
    };
  }

  return {
    userId: null,
    guestId: getGuestUserId(),
    username: null,
  };
};
```

### Backend (`api_server.py`)
```python
class UserEventSubmission(BaseModel):
    event_category: str
    event_action: str
    event_label: Optional[str] = None
    session_id: Optional[str] = None
    page_url: Optional[str] = None
    user_id: Optional[int] = None      # Authenticated user ID
    guest_id: Optional[str] = None     # Guest identifier (e.g., "Guest1")
    username: Optional[str] = None     # Username for authenticated users

@app.post("/api/analytics/event")
async def track_event(event: UserEventSubmission):
    """Track user behavior events with guest/user identification"""
    # Stores user_id, guest_id, and username in database
```

### Database Schema (`user_events` table)
```sql
CREATE TABLE user_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_category TEXT NOT NULL,
    event_action TEXT NOT NULL,
    event_label TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    page_url TEXT,
    user_id INTEGER,        -- NULL for guests
    guest_id TEXT,          -- NULL for authenticated users
    username TEXT           -- NULL for guests
);

CREATE INDEX idx_user_events_user_id ON user_events(user_id);
CREATE INDEX idx_user_events_guest_id ON user_events(guest_id);
```

## Benefits

1. **User Learning**: Track which features guests use most before signing up
2. **Conversion Funnel**: Analyze guest → user conversion patterns
3. **Privacy-Friendly**: No personal data collected for guests
4. **MVP Feedback**: Collect usage data to improve product (Build-Measure-Learn cycle)

## Example Analytics Queries

### Count events by user type
```sql
SELECT
    CASE
        WHEN user_id IS NOT NULL THEN 'Authenticated'
        ELSE 'Guest'
    END as user_type,
    COUNT(*) as event_count
FROM user_events
GROUP BY user_type;
```

### Top guest users by activity
```sql
SELECT
    guest_id,
    COUNT(*) as event_count,
    MIN(timestamp) as first_seen,
    MAX(timestamp) as last_seen
FROM user_events
WHERE guest_id IS NOT NULL
GROUP BY guest_id
ORDER BY event_count DESC
LIMIT 10;
```

### Most popular features for guests
```sql
SELECT
    event_category,
    event_action,
    COUNT(*) as usage_count
FROM user_events
WHERE guest_id IS NOT NULL
GROUP BY event_category, event_action
ORDER BY usage_count DESC;
```

## Migration
Run the migration script to add guest tracking columns:
```bash
python add_guest_tracking.py
```

This adds:
- `user_id` column to `user_events` table
- `guest_id` column to `user_events` table
- `username` column to `user_events` table
- Appropriate indexes for performance
