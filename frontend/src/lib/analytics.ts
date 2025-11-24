// User behavior tracking utility for MVP analytics
import { useAuthStore } from '@/store/useAuthStore';

// API URL configuration for different environments
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Get or create guest user identifier (Guest1, Guest2, etc.)
export const getGuestUserId = (): string => {
  let guestId = localStorage.getItem('guest_user_id');
  if (!guestId) {
    // Fetch the next guest number from server or generate locally
    const guestNumber = Math.floor(Math.random() * 10000) + 1;
    guestId = `Guest${guestNumber}`;
    localStorage.setItem('guest_user_id', guestId);
  }
  return guestId;
};

// Get or create session ID
export const getSessionId = (): string => {
  let sessionId = localStorage.getItem('session_id');
  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('session_id', sessionId);
  }
  return sessionId;
};

// Get user identifier (authenticated user ID or guest ID)
export const getUserIdentifier = (): { userId: number | null; guestId: string | null; username: string | null } => {
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

// Track a user event
export const trackEvent = async (
  category: string,
  action: string,
  label?: string
): Promise<void> => {
  try {
    const userIdentifier = getUserIdentifier();

    await fetch(`${API_BASE_URL}/api/analytics/event`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        event_category: category,
        event_action: action,
        event_label: label,
        session_id: getSessionId(),
        page_url: window.location.pathname,
        user_id: userIdentifier.userId,
        guest_id: userIdentifier.guestId,
        username: userIdentifier.username,
      }),
    });
  } catch (error) {
    console.error('Failed to track event:', error);
  }
};

// Predefined tracking functions for common events
export const analytics = {
  // Festival events
  festivalSearched: (query: string) => trackEvent('Festival', 'Searched', query),
  festivalViewed: (festivalName: string) => trackEvent('Festival', 'Viewed', festivalName),
  festivalFiltered: (filterType: string) => trackEvent('Festival', 'Filtered', filterType),

  // AI feature events
  sentimentAnalysisClicked: (festivalName: string) =>
    trackEvent('AI', 'SentimentAnalysisClicked', festivalName),
  sentimentAnalysisCompleted: (festivalName: string) =>
    trackEvent('AI', 'SentimentAnalysisCompleted', festivalName),

  wordcloudGenerated: (festivalName: string) =>
    trackEvent('AI', 'WordcloudGenerated', festivalName),

  trendAnalysisViewed: (festivalName: string) =>
    trackEvent('AI', 'TrendAnalysisViewed', festivalName),

  aiRenderingRequested: (festivalName: string) =>
    trackEvent('AI', 'RenderingRequested', festivalName),
  aiRenderingCompleted: (festivalName: string) =>
    trackEvent('AI', 'RenderingCompleted', festivalName),

  rankingRequested: (festivalCount: number) =>
    trackEvent('AI', 'RankingRequested', `${festivalCount} festivals`),

  // Course events
  courseItemAdded: (itemType: string, itemName: string) =>
    trackEvent('Course', 'ItemAdded', `${itemType}: ${itemName}`),
  courseItemRemoved: (itemType: string, itemName: string) =>
    trackEvent('Course', 'ItemRemoved', `${itemType}: ${itemName}`),
  courseValidated: (itemCount: number) =>
    trackEvent('Course', 'Validated', `${itemCount} items`),

  // Navigation events
  pageViewed: (pageName: string) => trackEvent('Navigation', 'PageViewed', pageName),
  tabChanged: (tabName: string) => trackEvent('Navigation', 'TabChanged', tabName),

  // Search events
  categorySelected: (category: string) => trackEvent('Search', 'CategorySelected', category),
  areaSelected: (area: string) => trackEvent('Search', 'AreaSelected', area),
  statusFilterApplied: (status: string) => trackEvent('Search', 'StatusFilterApplied', status),

  // Nearby search events
  nearbySearchPerformed: (festivalName: string, radius: number) =>
    trackEvent('Nearby', 'SearchPerformed', `${festivalName} (${radius}km)`),

  // Image events
  blogImagesScraped: (festivalName: string, imageCount: number) =>
    trackEvent('Images', 'BlogImagesScraped', `${festivalName}: ${imageCount} images`),

  // Precaution events
  precautionsViewed: (festivalName: string) =>
    trackEvent('Precautions', 'Viewed', festivalName),
};

// Hook to track page views on component mount
export const usePageTracking = (pageName: string) => {
  React.useEffect(() => {
    analytics.pageViewed(pageName);
  }, [pageName]);
};

// For compatibility with React
import React from 'react';
