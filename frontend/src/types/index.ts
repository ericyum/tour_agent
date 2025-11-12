export interface Festival {
  title: string
  image: string
  start_date?: string
  end_date?: string
}

export interface SearchFilters {
  area: string
  sigungu: string
  main_cat: string
  medium_cat: string
  small_cat: string
  status: string
  page: number
}

export interface SearchResponse {
  festivals: Festival[]
  total: number
  page: number
  total_pages: number
}

export interface FestivalDetails {
  [key: string]: any
  title?: string
  addr1?: string
  overview?: string
  eventstartdate?: string
  eventenddate?: string
  firstimage?: string
  tel?: string
  homepage?: string
}

export interface TrendData {
  yearly_trend: string | null
  event_trend: string | null
  message: string
}

export interface SentimentAnalysis {
  summary: string
  positive_count: number
  negative_count: number
  neutral_count: number
  charts: {
    donut_chart?: string
    satisfaction_chart?: string
    absolute_chart?: string
    outlier_chart?: string
    wordcloud_positive?: string
    wordcloud_negative?: string
    // Chart data for Recharts rendering
    donut_data?: {
      positive: number
      negative: number
    }
    satisfaction_data?: {
      labels: string[]
      counts: number[]
    }
    absolute_data?: {
      labels: string[]
      counts: number[]
    }
    outlier_data?: {
      min: number
      q1: number
      median: number
      q3: number
      max: number
      lower_bound: number
      upper_bound: number
      outliers: number[]
    }
  }
  blog_results: Array<{
    '블로그 제목': string
    '링크': string
    '감성 빈도': number
    '감성 점수': string
    '긍정 문장 수': number
    '부정 문장 수': number
    '긍정 비율 (%)': string
    '부정 비율 (%)': string
    '긍/부정 문장 요약': string
    '만족도 점수': string
  }>
  blog_list_csv_path?: string
  positive_keywords?: string
  negative_summary?: string
  outlier_description?: string
  total_score_count?: number
  outlier_count?: number
  overall_summary_text?: string
}

export interface RankingResult {
  report: string
  top_festivals: Array<{
    name: string
    score: number
    rank: number
  }>
}

export interface CourseItem {
  title: string
  type: 'festival' | 'facility' | 'course'
  [key: string]: any
}

export interface ValidationResult {
  validation_result: string
}

export interface NearbyRecommendations {
  facilities: any[]
  courses: any[]
  festivals: any[]
}

export interface ScrapedImages {
  image_urls: string[]
}

export interface WordCloud {
  wordcloud: string // base64 image string
  message: string
}

export interface ReviewSummary {
  summary: string
}
