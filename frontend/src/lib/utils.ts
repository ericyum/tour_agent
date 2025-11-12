import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'
import { format, parse } from 'date-fns'
import { ko } from 'date-fns/locale'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(dateString: string | undefined): string {
  if (!dateString) return '-'

  try {
    // Parse YYYYMMDD format
    const parsed = parse(dateString, 'yyyyMMdd', new Date())
    return format(parsed, 'yyyy년 M월 d일', { locale: ko })
  } catch {
    return dateString
  }
}

export function formatDateRange(startDate: string | undefined, endDate: string | undefined): string {
  if (!startDate && !endDate) return '날짜 미정'
  if (!startDate) return `~ ${formatDate(endDate)}`
  if (!endDate) return `${formatDate(startDate)} ~`

  return `${formatDate(startDate)} ~ ${formatDate(endDate)}`
}

export function getFestivalStatus(startDate: string | undefined, endDate: string | undefined): {
  status: '진행중' | '진행 예정' | '종료' | '미정'
  color: string
} {
  if (!startDate || !endDate) {
    return { status: '미정', color: 'text-gray-500 bg-gray-100' }
  }

  const today = new Date()
  const start = parse(startDate, 'yyyyMMdd', new Date())
  const end = parse(endDate, 'yyyyMMdd', new Date())

  if (today >= start && today <= end) {
    return { status: '진행중', color: 'text-green-700 bg-green-100' }
  } else if (today < start) {
    return { status: '진행 예정', color: 'text-blue-700 bg-blue-100' }
  } else {
    return { status: '종료', color: 'text-gray-700 bg-gray-100' }
  }
}

export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + '...'
}

export function removeHtmlTags(html: string): string {
  return html.replace(/<[^>]*>/g, '')
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null
      func(...args)
    }

    if (timeout) {
      clearTimeout(timeout)
    }
    timeout = setTimeout(later, wait)
  }
}

export function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}
