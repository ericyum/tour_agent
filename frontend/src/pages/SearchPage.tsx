import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { FaSpinner, FaChevronLeft, FaChevronRight, FaTrophy, FaCheckSquare, FaSquare } from 'react-icons/fa'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import SearchFilters from '@/components/search/SearchFilters'
import FestivalCard from '@/components/festival/FestivalCard'
import { searchFestivals, rankFestivals } from '@/lib/api'
import type { SearchFilters as SearchFiltersType } from '@/types'

export default function SearchPage() {
  const [filters, setFilters] = useState<SearchFiltersType>({
    area: '전체',
    sigungu: '전체',
    main_cat: '전체',
    medium_cat: '전체',
    small_cat: '전체',
    status: '전체',
    page: 1,
  })

  const [hasSearched, setHasSearched] = useState(false)
  const [selectedFestivals, setSelectedFestivals] = useState<string[]>([])
  const [rankingResult, setRankingResult] = useState<any>(null)
  const [numReviews, setNumReviews] = useState(5)
  const [topN, setTopN] = useState(3)

  const {
    data,
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ['festivals', filters],
    queryFn: () => searchFestivals(filters),
    enabled: hasSearched,
  })

  const rankingMutation = useMutation({
    mutationFn: () => rankFestivals(selectedFestivals, numReviews, topN),
    onSuccess: (data) => {
      console.log('Ranking result:', data)
      setRankingResult(data)
    },
    onError: (error: any) => {
      console.error('Ranking error:', error)
      console.error('Error response:', error.response?.data)
      console.error('Error status:', error.response?.status)
      alert(`랭킹 분석 중 오류가 발생했습니다.\n${error.response?.data?.detail || error.message}`)
    }
  })

  const handleSearch = () => {
    setHasSearched(true)
    setFilters((prev) => ({ ...prev, page: 1 }))
    setSelectedFestivals([])
    setRankingResult(null)
    refetch()
  }

  const handlePageChange = (newPage: number) => {
    setFilters((prev) => ({ ...prev, page: newPage }))
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const toggleFestivalSelection = (festivalName: string) => {
    setSelectedFestivals((prev) => {
      const newSelection = prev.includes(festivalName)
        ? prev.filter((name) => name !== festivalName)
        : [...prev, festivalName]

      // Auto-adjust topN to not exceed selected festivals count
      if (newSelection.length > 0 && topN > newSelection.length) {
        setTopN(newSelection.length)
      }

      return newSelection
    })
  }

  const handleRanking = () => {
    if (selectedFestivals.length < 2) {
      alert('최소 2개 이상의 축제를 선택해주세요')
      return
    }
    rankingMutation.mutate()
  }

  const renderPagination = () => {
    if (!data || data.total_pages <= 1) return null

    const pages = []
    const maxButtons = 5
    let startPage = Math.max(1, filters.page - Math.floor(maxButtons / 2))
    let endPage = Math.min(data.total_pages, startPage + maxButtons - 1)

    if (endPage - startPage + 1 < maxButtons) {
      startPage = Math.max(1, endPage - maxButtons + 1)
    }

    for (let i = startPage; i <= endPage; i++) {
      pages.push(i)
    }

    return (
      <div className="flex justify-center items-center space-x-2 mt-12">
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => handlePageChange(filters.page - 1)}
          disabled={filters.page === 1}
          className="p-3 rounded-lg bg-white shadow-md hover:bg-primary-50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          <FaChevronLeft />
        </motion.button>

        {pages.map((page) => (
          <motion.button
            key={page}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => handlePageChange(page)}
            className={`
              px-4 py-2 rounded-lg font-medium transition-all
              ${page === filters.page
                ? 'bg-primary-600 text-white shadow-lg'
                : 'bg-white text-slate-700 hover:bg-primary-50 shadow-md'
              }
            `}
          >
            {page}
          </motion.button>
        ))}

        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => handlePageChange(filters.page + 1)}
          disabled={filters.page === data.total_pages}
          className="p-3 rounded-lg bg-white shadow-md hover:bg-primary-50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          <FaChevronRight />
        </motion.button>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-8"
    >
      <div className="text-center mb-8">
        <h1 className="section-title">축제 검색</h1>
        <p className="text-slate-600 text-lg">
          원하는 조건으로 전국의 축제를 찾아보세요
        </p>
      </div>

      <SearchFilters
        filters={filters}
        onFiltersChange={setFilters}
        onSearch={handleSearch}
        isSearching={isFetching}
      />

      {/* Results */}
      <AnimatePresence mode="wait">
        {isLoading ? (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center py-20 space-y-4"
          >
            <FaSpinner className="text-6xl text-primary-600 animate-spin" />
            <p className="text-xl text-slate-600">축제를 검색하는 중...</p>
          </motion.div>
        ) : hasSearched && data ? (
          <motion.div
            key="results"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Results Header with Ranking Button */}
            <div className="mb-6 space-y-4">
              <div className="flex flex-col sm:flex-row gap-4 justify-between items-center">
                <div>
                  <p className="text-lg text-slate-700">
                    총 <span className="font-bold text-primary-600">{data.total}</span>개의 축제를 찾았습니다
                  </p>
                  <p className="text-sm text-slate-500">
                    {filters.page} / {data.total_pages} 페이지
                  </p>
                </div>
                {selectedFestivals.length > 0 && (
                  <div className="flex flex-col items-end gap-2">
                    <motion.button
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={handleRanking}
                      disabled={rankingMutation.isPending}
                      className="btn-primary flex items-center space-x-2"
                    >
                      {rankingMutation.isPending ? (
                        <>
                          <FaSpinner className="animate-spin" />
                          <span>AI 분석 중...</span>
                        </>
                      ) : (
                        <>
                          <FaTrophy />
                          <span>선택한 축제 랭킹 ({selectedFestivals.length}개)</span>
                        </>
                      )}
                    </motion.button>
                    {rankingMutation.isPending && (
                      <p className="text-xs text-slate-500 animate-pulse">
                        ⏱️ AI가 블로그 후기를 분석 중입니다... (최대 5분 소요)
                      </p>
                    )}
                  </div>
                )}
              </div>

              {/* Ranking Settings */}
              <motion.div
                animate={selectedFestivals.length > 0 ? 'open' : 'closed'}
                initial="closed"
                variants={{
                  open: {
                    gridTemplateRows: '1fr',
                    opacity: 1,
                    marginTop: 0
                  },
                  closed: {
                    gridTemplateRows: '0fr',
                    opacity: 0,
                    marginTop: -16
                  }
                }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
                className="grid overflow-hidden"
              >
                <div className="min-h-0">
                  <div className="card p-4 bg-gradient-to-r from-primary-50 to-accent-50">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">
                          후기 분석 개수: <span className="font-bold text-primary-600">{numReviews}개</span>
                        </label>
                        <input
                          type="range"
                          min="1"
                          max="10"
                          step="1"
                          value={numReviews}
                          onChange={(e) => setNumReviews(Number(e.target.value))}
                          className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
                        />
                        <div className="flex justify-between text-xs text-slate-500 mt-1">
                          <span>1개</span>
                          <span>5개</span>
                          <span>10개</span>
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">
                          순위 표시: <span className="font-bold text-primary-600">Top {topN}</span>
                          <span className="text-xs text-slate-500 ml-2">(선택: {selectedFestivals.length}개)</span>
                        </label>
                        <input
                          type="range"
                          min="1"
                          max={Math.min(5, selectedFestivals.length)}
                          step="1"
                          value={topN}
                          onChange={(e) => setTopN(Number(e.target.value))}
                          className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-accent-600"
                        />
                        <div className="relative flex justify-between text-xs text-slate-500 mt-1">
                          {Array.from({ length: Math.min(5, selectedFestivals.length) }, (_, i) => i + 1).map((num) => (
                            <span key={num} className="flex-1 text-center first:text-left last:text-right">
                              Top {num}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Festival Grid */}
            {data.festivals.length > 0 ? (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {data.festivals.map((festival, index) => (
                    <div key={festival.title} className="relative">
                      {/* Selection Checkbox */}
                      <motion.button
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                        onClick={() => toggleFestivalSelection(festival.title)}
                        className="absolute top-4 right-4 z-10 p-2 bg-white rounded-full shadow-lg hover:shadow-xl transition-all"
                      >
                        {selectedFestivals.includes(festival.title) ? (
                          <FaCheckSquare className="text-2xl text-primary-600" />
                        ) : (
                          <FaSquare className="text-2xl text-slate-300" />
                        )}
                      </motion.button>
                      <FestivalCard
                        item={{ ...festival, type: 'festival', image: festival.image || 'https://placehold.co/400x300?text=No+Image' }}
                        index={index}
                      />
                    </div>
                  ))}
                </div>

                {/* Pagination */}
                {renderPagination()}
              </>
            ) : (
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="card p-12 text-center"
              >
                <div className="text-6xl mb-4">🔍</div>
                <h3 className="text-2xl font-bold mb-2">검색 결과가 없습니다</h3>
                <p className="text-slate-600">
                  다른 조건으로 다시 검색해보세요
                </p>
              </motion.div>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="card p-12 text-center"
          >
            <div className="text-6xl mb-4">🎪</div>
            <h3 className="text-2xl font-bold mb-2">검색을 시작하세요</h3>
            <p className="text-slate-600">
              위의 필터를 선택하고 검색 버튼을 눌러주세요
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Ranking Result */}
      {rankingResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card p-6 mt-8"
        >
          <h2 className="text-2xl font-bold mb-4 flex items-center space-x-2">
            <FaTrophy className="text-yellow-500" />
            <span>AI 축제 랭킹</span>
          </h2>

          {rankingResult.ranked_festivals && rankingResult.ranked_festivals.length > 0 && (
            <div className="space-y-4 mb-6">
              {rankingResult.ranked_festivals.slice(0, topN).map((festival: any, idx: number) => (
                <motion.div
                  key={festival.title}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className={`p-4 rounded-lg flex items-center space-x-4 ${
                    idx === 0 ? 'bg-gradient-to-r from-yellow-100 to-yellow-50 border-2 border-yellow-300' :
                    idx === 1 ? 'bg-gradient-to-r from-gray-100 to-gray-50 border-2 border-gray-300' :
                    idx === 2 ? 'bg-gradient-to-r from-orange-100 to-orange-50 border-2 border-orange-300' :
                    'bg-slate-50'
                  }`}
                >
                  <div className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center text-2xl font-bold ${
                    idx === 0 ? 'bg-yellow-500 text-white' :
                    idx === 1 ? 'bg-gray-400 text-white' :
                    idx === 2 ? 'bg-orange-500 text-white' :
                    'bg-slate-300 text-white'
                  }`}>
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <h3 className="font-bold text-lg">{festival.title}</h3>
                    <p className="text-sm text-slate-600">점수: {festival.ranking_score?.toFixed(2) || 'N/A'}</p>
                  </div>
                  {idx === 0 && <div className="text-4xl">🏆</div>}
                  {idx === 1 && <div className="text-4xl">🥈</div>}
                  {idx === 2 && <div className="text-4xl">🥉</div>}
                </motion.div>
              ))}
            </div>
          )}

          {rankingResult.analysis && (
            <div className="bg-primary-50 border-l-4 border-primary-500 p-6 rounded-r-lg">
              <h3 className="font-bold mb-2">AI 분석 결과</h3>
              <article className="prose prose-slate max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {rankingResult.analysis}
                </ReactMarkdown>
              </article>
            </div>
          )}
        </motion.div>
      )}
    </motion.div>
  )
}
