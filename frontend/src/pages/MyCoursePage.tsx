import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FaTrash, FaRoute, FaSpinner, FaMapMarkerAlt } from 'react-icons/fa'
import { useMutation } from '@tanstack/react-query'
import { useCourseStore } from '@/store/useCourseStore'
import { validateCourse, searchNearby } from '@/lib/api'
import { formatDateRange } from '@/lib/utils'

export default function MyCoursePage() {
  const { courseItems, removeItem, clearCourse } = useCourseStore()
  const [duration, setDuration] = useState('1박 2일')
  const [validationResult, setValidationResult] = useState<string | null>(null)
  const [nearbyResult, setNearbyResult] = useState<any>(null)
  const [nearbyRadius, setNearbyRadius] = useState(5)

  const validateMutation = useMutation({
    mutationFn: () => validateCourse(courseItems, duration),
    onSuccess: (data) => {
      setValidationResult(data.validation_result)
    },
  })

  const nearbyMutation = useMutation({
    mutationFn: () => {
      const firstItem = courseItems[0]
      if (!firstItem.mapx || !firstItem.mapy) {
        throw new Error('위치 정보가 없습니다')
      }
      return searchNearby(
        parseFloat(firstItem.mapy),
        parseFloat(firstItem.mapx),
        nearbyRadius * 1000,
        firstItem.contentid
      )
    },
    onSuccess: (data) => {
      setNearbyResult(data)
    },
  })

  const handleValidate = () => {
    if (courseItems.length === 0) {
      alert('코스에 최소 1개 이상의 항목을 추가해주세요')
      return
    }
    setValidationResult(null)
    validateMutation.mutate()
  }

  const handleNearbySearch = () => {
    if (courseItems.length === 0) {
      alert('코스에 최소 1개 이상의 항목을 추가해주세요')
      return
    }
    if (!courseItems[0].mapx || !courseItems[0].mapy) {
      alert('첫 번째 항목에 위치 정보가 없습니다')
      return
    }
    setNearbyResult(null)
    nearbyMutation.mutate()
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-8"
    >
      <div className="text-center mb-8">
        <h1 className="section-title">내 여행 코스</h1>
        <p className="text-slate-600 text-lg">
          AI가 최적의 동선과 일정을 제안해드립니다
        </p>
      </div>

      {courseItems.length === 0 ? (
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="card p-12 text-center"
        >
          <div className="text-6xl mb-4">📍</div>
          <h3 className="text-2xl font-bold mb-2">코스가 비어있습니다</h3>
          <p className="text-slate-600 mb-6">
            축제나 관광지를 검색하여 '내 코스에 추가' 버튼을 눌러주세요
          </p>
        </motion.div>
      ) : (
        <>
          {/* Course Settings */}
          <div className="card p-6">
            <div className="flex flex-col gap-4">
              {/* Duration and Actions Row */}
              <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
                <div className="flex items-center space-x-4">
                  <label className="font-medium text-slate-700">여행 기간:</label>
                  <select
                    value={duration}
                    onChange={(e) => setDuration(e.target.value)}
                    className="px-4 py-2 rounded-lg border-2 border-slate-200 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 transition-all outline-none"
                  >
                    <option>당일치기</option>
                    <option>1박 2일</option>
                    <option>2박 3일</option>
                    <option>3박 4일</option>
                    <option>4박 5일</option>
                  </select>
                </div>

                <div className="flex gap-2">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleValidate}
                    disabled={validateMutation.isPending}
                    className="btn-primary flex items-center space-x-2"
                  >
                    {validateMutation.isPending ? (
                      <>
                        <FaSpinner className="animate-spin" />
                        <span>검증 중...</span>
                      </>
                    ) : (
                      <>
                        <FaRoute />
                        <span>AI 코스 검증</span>
                      </>
                    )}
                  </motion.button>

                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={clearCourse}
                    className="px-6 py-3 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 transition-colors"
                  >
                    전체 삭제
                  </motion.button>
                </div>
              </div>

              {/* Nearby Search Row */}
              <div className="flex flex-col sm:flex-row gap-4 items-center justify-between border-t pt-4">
                <div className="flex items-center space-x-4">
                  <label className="font-medium text-slate-700">주변 검색 반경:</label>
                  <select
                    value={nearbyRadius}
                    onChange={(e) => setNearbyRadius(Number(e.target.value))}
                    className="px-4 py-2 rounded-lg border-2 border-slate-200 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 transition-all outline-none"
                  >
                    <option value={1}>1km</option>
                    <option value={3}>3km</option>
                    <option value={5}>5km</option>
                    <option value={10}>10km</option>
                    <option value={20}>20km</option>
                  </select>
                </div>

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleNearbySearch}
                  disabled={nearbyMutation.isPending}
                  className="btn-secondary flex items-center space-x-2"
                >
                  {nearbyMutation.isPending ? (
                    <>
                      <FaSpinner className="animate-spin" />
                      <span>검색 중...</span>
                    </>
                  ) : (
                    <>
                      <FaMapMarkerAlt />
                      <span>주변 추천받기</span>
                    </>
                  )}
                </motion.button>
              </div>
            </div>
          </div>

          {/* Course Items List */}
          <div className="card p-6">
            <h2 className="text-2xl font-bold mb-6 flex items-center space-x-2">
              <span>📍</span>
              <span>코스 목록 ({courseItems.length})</span>
            </h2>

            <div className="space-y-4">
              <AnimatePresence>
                {courseItems.map((item, index) => (
                  <motion.div
                    key={item.title}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className="flex items-center gap-4 p-4 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors"
                  >
                    <div className="flex-shrink-0 w-10 h-10 bg-primary-600 text-white rounded-full flex items-center justify-center font-bold">
                      {index + 1}
                    </div>

                    <div className="flex-1">
                      <h3 className="font-bold text-lg">{item.title}</h3>
                      {item.eventstartdate && (
                        <p className="text-sm text-slate-600">
                          {formatDateRange(item.eventstartdate, item.eventenddate)}
                        </p>
                      )}
                      {item.addr1 && (
                        <p className="text-sm text-slate-500">{item.addr1}</p>
                      )}
                    </div>

                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => removeItem(item.title)}
                      className="p-3 rounded-full bg-red-100 text-red-600 hover:bg-red-200 transition-colors"
                    >
                      <FaTrash />
                    </motion.button>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>

          {/* Validation Result */}
          {validationResult && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="card p-6"
            >
              <h2 className="text-2xl font-bold mb-4 flex items-center space-x-2">
                <span>🤖</span>
                <span>AI 코스 검증 결과</span>
              </h2>
              <div className="prose max-w-none text-slate-700 leading-relaxed whitespace-pre-wrap">
                {validationResult}
              </div>
            </motion.div>
          )}

          {validateMutation.isError && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="card p-6 bg-red-50 border-2 border-red-200"
            >
              <p className="text-red-700">
                코스 검증 중 오류가 발생했습니다. 다시 시도해주세요.
              </p>
            </motion.div>
          )}

          {/* Nearby Recommendations Result */}
          {nearbyResult && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="card p-6"
            >
              <h2 className="text-2xl font-bold mb-4 flex items-center space-x-2">
                <FaMapMarkerAlt className="text-primary-600" />
                <span>주변 추천 장소</span>
              </h2>

              {nearbyResult.summary && (
                <div className="bg-primary-50 border-l-4 border-primary-500 p-6 rounded-r-lg mb-6">
                  <h3 className="font-bold mb-2">AI 추천 요약</h3>
                  <p className="text-slate-700 leading-relaxed whitespace-pre-wrap">
                    {nearbyResult.summary}
                  </p>
                </div>
              )}

              {nearbyResult.nearby_items && nearbyResult.nearby_items.length > 0 && (
                <div>
                  <h3 className="text-lg font-bold mb-3">추천 장소 목록</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {nearbyResult.nearby_items.map((item: any, idx: number) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.05 }}
                        className="bg-slate-50 p-4 rounded-lg hover:bg-slate-100 transition-colors"
                      >
                        <h4 className="font-bold text-lg mb-2">{item.title}</h4>
                        {item.addr1 && (
                          <p className="text-sm text-slate-600 mb-1 flex items-center space-x-1">
                            <FaMapMarkerAlt className="text-xs" />
                            <span>{item.addr1}</span>
                          </p>
                        )}
                        {item.dist && (
                          <p className="text-sm text-primary-600 font-medium">
                            거리: {(item.dist / 1000).toFixed(2)}km
                          </p>
                        )}
                        {item.cat3 && (
                          <span className="inline-block mt-2 px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-xs">
                            {item.cat3}
                          </span>
                        )}
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {(!nearbyResult.nearby_items || nearbyResult.nearby_items.length === 0) && (
                <p className="text-center text-slate-500 py-8">
                  주변에 추천할 장소가 없습니다.
                </p>
              )}
            </motion.div>
          )}

          {nearbyMutation.isError && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="card p-6 bg-red-50 border-2 border-red-200"
            >
              <p className="text-red-700">
                주변 검색 중 오류가 발생했습니다. 첫 번째 항목에 위치 정보가 있는지 확인해주세요.
              </p>
            </motion.div>
          )}
        </>
      )}
    </motion.div>
  )
}
