import axios from 'axios'
import type {
  SearchFilters,
  SearchResponse,
  FestivalDetails,
  TrendData,
  SentimentAnalysis,
  RankingResult,
  ValidationResult,
  NearbyRecommendations,
  ScrapedImages,
  WordCloud,
  ReviewSummary,
} from '@/types'

const api = axios.create({
  baseURL: '/api',
  timeout: 0, // No timeout - wait until completion or actual error
})

// Configuration APIs
export const getAreas = async () => {
  const { data } = await api.get('/config/areas')
  return data.areas as string[]
}

export const getSigungus = async (area: string) => {
  const { data } = await api.get('/config/sigungus', { params: { area } })
  return data.sigungus as string[]
}

export const getMainCategories = async () => {
  const { data } = await api.get('/config/categories')
  return data.main_categories as string[]
}

export const getMediumCategories = async (main_cat: string) => {
  const { data } = await api.get('/config/categories/medium', { params: { main_cat } })
  return data.medium_categories as string[]
}

export const getSmallCategories = async (main_cat: string, medium_cat: string) => {
  const { data } = await api.get('/config/categories/small', { params: { main_cat, medium_cat } })
  return data.small_categories as string[]
}

// Festival APIs
export const searchFestivals = async (filters: SearchFilters): Promise<SearchResponse> => {
  const { data } = await api.post('/festivals/search', filters)
  return data
}

export const getFestivalDetails = async (festivalName: string) => {
  const { data } = await api.get(`/festivals/${encodeURIComponent(festivalName)}`)
  return data as { details: FestivalDetails; icon_path: string | null; best_image_path: string | null }
}

export const getFestivalTrend = async (festivalName: string): Promise<TrendData> => {
  const { data } = await api.get(`/festivals/${encodeURIComponent(festivalName)}/trend`)
  return data
}

export const getFestivalSentiment = async (
  festivalName: string,
  numReviews: number = 10
): Promise<SentimentAnalysis> => {
  const { data } = await api.get(`/festivals/${encodeURIComponent(festivalName)}/sentiment`, {
    params: { num_reviews: numReviews },
  })
  return data
}

export const getFestivalImages = async (
  festivalName: string,
  numBlogs: number = 5
): Promise<ScrapedImages> => {
  const { data } = await api.get(`/festivals/${encodeURIComponent(festivalName)}/images`, {
    params: { num_blogs: numBlogs },
  })
  return data
}

export const getFestivalWordCloud = async (
  festivalName: string,
  numReviews: number = 20
): Promise<WordCloud> => {
  const { data } = await api.get(`/festivals/${encodeURIComponent(festivalName)}/wordcloud`, {
    params: { num_reviews: numReviews },
  })
  return data
}

export const getFestivalReviewSummary = async (
  festivalName: string,
  numReviews: number = 5
): Promise<ReviewSummary> => {
  const { data } = await api.get(`/festivals/${encodeURIComponent(festivalName)}/review-summary`, {
    params: { num_reviews: numReviews },
  })
  return data
}

export const getFestivalPrecautions = async (festivalName: string) => {
  const { data } = await api.get(`/festivals/${encodeURIComponent(festivalName)}/precautions`)
  return data.precautions as string
}

export const rankFestivals = async (
  festivals: string[],
  numReviews: number = 10,
  topN: number = 3
): Promise<RankingResult> => {
  const { data } = await api.post('/festivals/ranking', {
    festivals,
    num_reviews: numReviews,
    top_n: topN,
  })
  return data
}

export const renderFestivalImage = async (festivalName: string) => {
  const { data } = await api.post(`/festivals/${encodeURIComponent(festivalName)}/render`)
  return data
}

export const getCourseDetails = async (courseTitle: string) => {
  const { data } = await api.get(`/courses/${encodeURIComponent(courseTitle)}`)
  return data as { details: any } // Adjust type as needed
}

export const getFacilityDetails = async (facilityTitle: string) => {
  const { data } = await api.get(`/facilities/${encodeURIComponent(facilityTitle)}`)
  return data as { details: any } // Adjust type as needed
}

// Course APIs
export const validateCourse = async (
  course: any[],
  duration: string
): Promise<ValidationResult> => {
  const { data } = await api.post('/course/validate', { course, duration })
  return data
}

// Nearby search API
export const searchNearby = async (
  latitude: number,
  longitude: number,
  radius: number,
  currentFestivalId?: string
): Promise<NearbyRecommendations> => {
  const { data } = await api.post('/nearby/search', {
    latitude,
    longitude,
    radius,
    current_festival_id: currentFestivalId,
  })
  return data
}

export default api
