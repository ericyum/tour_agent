import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { FaSearch, FaSpinner } from 'react-icons/fa'
import {
  getAreas,
  getSigungus,
  getMainCategories,
  getMediumCategories,
  getSmallCategories,
} from '@/lib/api'
import type { SearchFilters as SearchFiltersType } from '@/types'

interface SearchFiltersProps {
  filters: SearchFiltersType
  onFiltersChange: (filters: SearchFiltersType) => void
  onSearch: () => void
  isSearching?: boolean
}

export default function SearchFilters({
  filters,
  onFiltersChange,
  onSearch,
  isSearching,
}: SearchFiltersProps) {
  const { data: areas = ['전체'] } = useQuery({
    queryKey: ['areas'],
    queryFn: getAreas,
  })

  const { data: sigungus = ['전체'] } = useQuery({
    queryKey: ['sigungus', filters.area],
    queryFn: () => getSigungus(filters.area),
    enabled: filters.area !== '전체',
  })

  const { data: mainCategories = ['전체'] } = useQuery({
    queryKey: ['mainCategories'],
    queryFn: getMainCategories,
  })

  const { data: mediumCategories = ['전체'] } = useQuery({
    queryKey: ['mediumCategories', filters.main_cat],
    queryFn: () => getMediumCategories(filters.main_cat),
    enabled: filters.main_cat !== '전체',
  })

  const { data: smallCategories = ['전체'] } = useQuery({
    queryKey: ['smallCategories', filters.main_cat, filters.medium_cat],
    queryFn: () => getSmallCategories(filters.main_cat, filters.medium_cat),
    enabled: filters.main_cat !== '전체' && filters.medium_cat !== '전체',
  })

  const updateFilter = (key: keyof SearchFiltersType, value: string) => {
    const newFilters = { ...filters, [key]: value, page: 1 }

    // Reset dependent filters
    if (key === 'area') {
      newFilters.sigungu = '전체'
    }
    if (key === 'main_cat') {
      newFilters.medium_cat = '전체'
      newFilters.small_cat = '전체'
    }
    if (key === 'medium_cat') {
      newFilters.small_cat = '전체'
    }

    onFiltersChange(newFilters)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card p-6 mb-8"
    >
      <h2 className="text-2xl font-bold mb-6 flex items-center space-x-2">
        <FaSearch className="text-primary-600" />
        <span>축제 검색</span>
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {/* Area */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            시/도
          </label>
          <select
            value={filters.area}
            onChange={(e) => updateFilter('area', e.target.value)}
            className="w-full px-4 py-3 rounded-lg border-2 border-slate-200 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 transition-all outline-none"
          >
            {areas.map((area) => (
              <option key={area} value={area}>
                {area}
              </option>
            ))}
          </select>
        </div>

        {/* Sigungu */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            시/군/구
          </label>
          <select
            value={filters.sigungu}
            onChange={(e) => updateFilter('sigungu', e.target.value)}
            className="w-full px-4 py-3 rounded-lg border-2 border-slate-200 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 transition-all outline-none"
            disabled={filters.area === '전체'}
          >
            {sigungus.map((sigungu) => (
              <option key={sigungu} value={sigungu}>
                {sigungu}
              </option>
            ))}
          </select>
        </div>

        {/* Main Category */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            대분류
          </label>
          <select
            value={filters.main_cat}
            onChange={(e) => updateFilter('main_cat', e.target.value)}
            className="w-full px-4 py-3 rounded-lg border-2 border-slate-200 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 transition-all outline-none"
          >
            {mainCategories.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
        </div>

        {/* Medium Category */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            중분류
          </label>
          <select
            value={filters.medium_cat}
            onChange={(e) => updateFilter('medium_cat', e.target.value)}
            className="w-full px-4 py-3 rounded-lg border-2 border-slate-200 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 transition-all outline-none"
            disabled={filters.main_cat === '전체'}
          >
            {mediumCategories.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
        </div>

        {/* Small Category */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            소분류
          </label>
          <select
            value={filters.small_cat}
            onChange={(e) => updateFilter('small_cat', e.target.value)}
            className="w-full px-4 py-3 rounded-lg border-2 border-slate-200 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 transition-all outline-none"
            disabled={filters.medium_cat === '전체'}
          >
            {smallCategories.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
        </div>

        {/* Status */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            진행 상태
          </label>
          <select
            value={filters.status}
            onChange={(e) => updateFilter('status', e.target.value)}
            className="w-full px-4 py-3 rounded-lg border-2 border-slate-200 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 transition-all outline-none"
          >
            <option value="전체">전체</option>
            <option value="축제 진행중">진행중</option>
            <option value="진행 예정">진행 예정</option>
            <option value="종료된 축제">종료</option>
          </select>
        </div>
      </div>

      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={onSearch}
        disabled={isSearching}
        className="w-full btn-primary flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isSearching ? (
          <>
            <FaSpinner className="animate-spin" />
            <span>검색 중...</span>
          </>
        ) : (
          <>
            <FaSearch />
            <span>검색하기</span>
          </>
        )}
      </motion.button>
    </motion.div>
  )
}
