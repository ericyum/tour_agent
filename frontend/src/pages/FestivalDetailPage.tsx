import { useParams } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  FaSpinner, FaMapMarkerAlt, FaPhone, FaGlobe, FaCalendarAlt, FaPlus,
  FaChartLine, FaHeart, FaExclamationTriangle, FaImage, FaCamera, FaCloud, FaCommentDots, FaStreetView
} from 'react-icons/fa'
import {
  getFestivalDetails, getFestivalTrend, getFestivalSentiment, getFestivalPrecautions, renderFestivalImage,
  getFestivalImages, getFestivalWordCloud, getFestivalReviewSummary, searchNearby
} from '@/lib/api'
import { formatDateRange, getFestivalStatus, removeHtmlTags } from '@/lib/utils'
import { useCourseStore } from '@/store/useCourseStore'
import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import FestivalCard from '@/components/festival/FestivalCard'
import SentimentDonutChart from '@/components/charts/SentimentDonutChart'
import SatisfactionBarChart from '@/components/charts/SatisfactionBarChart'
import AbsoluteScoreLineChart from '@/components/charts/AbsoluteScoreLineChart'
import OutlierBoxPlot from '@/components/charts/OutlierBoxPlot'

type TabType = 'images' | 'trend' | 'wordcloud' | 'sentiment' | 'summary' | 'nearby' | 'precautions' | 'rendering'

const LoadingSpinner = ({ text }: { text: string }) => (
  <div className="flex flex-col items-center justify-center py-12">
    <FaSpinner className="text-4xl text-primary-600 animate-spin mb-4" />
    <p className="text-slate-600">{text}</p>
  </div>
)

const ErrorMessage = ({ text }: { text: string }) => (
  <div className="text-center py-12 text-red-600">
    <p>{text}</p>
  </div>
)

export default function FestivalDetailPage() {
  const { festivalName } = useParams<{ festivalName: string }>()
  const addItem = useCourseStore((state) => state.addItem)
  const courseItems = useCourseStore((state) => state.courseItems)
  const [activeTab, setActiveTab] = useState<TabType>('images')

  // State for sliders
  const [numBlogsForImages, setNumBlogsForImages] = useState(5)
  const [numReviewsForSentiment, setNumReviewsForSentiment] = useState(10)
  const [numReviewsForWordCloud, setNumReviewsForWordCloud] = useState(20)
  const [numReviewsForSummary, setNumReviewsForSummary] = useState(5)
  const [nearbyRadius, setNearbyRadius] = useState(5000) // 5km
  const [blogTablePage, setBlogTablePage] = useState(1)
  const [selectedBlogIndex, setSelectedBlogIndex] = useState<number | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['festival-details', festivalName],
    queryFn: () => getFestivalDetails(festivalName!),
    enabled: !!festivalName,
  })

  // --- Mutations for user-triggered analysis ---
  const trendMutation = useMutation({ mutationFn: () => getFestivalTrend(festivalName!) })
  const sentimentMutation = useMutation({ mutationFn: () => getFestivalSentiment(festivalName!, numReviewsForSentiment) })
  const precautionsMutation = useMutation({ mutationFn: () => getFestivalPrecautions(festivalName!) })
  const renderingMutation = useMutation({ mutationFn: () => renderFestivalImage(festivalName!) })
  const imagesMutation = useMutation({ mutationFn: () => getFestivalImages(festivalName!, numBlogsForImages) })
  const wordCloudMutation = useMutation({ mutationFn: () => getFestivalWordCloud(festivalName!, numReviewsForWordCloud) })
  const summaryMutation = useMutation({ mutationFn: () => getFestivalReviewSummary(festivalName!, numReviewsForSummary) })
  const nearbyMutation = useMutation({
    mutationFn: () => searchNearby(data!.details.mapy, data!.details.mapx, nearbyRadius, data!.details.contentid),
  })

  if (isLoading) {
    return <LoadingSpinner text="축제 정보를 불러오는 중..." />
  }

  if (error || !data) {
    return <ErrorMessage text="축제 정보를 찾을 수 없습니다" />
  }

  const { details } = data
  const status = getFestivalStatus(details.eventstartdate, details.eventenddate)
  const isInCourse = courseItems.some((item) => item.title === details.title)

  const handleAddToCourse = () => {
    addItem({ ...details, type: 'festival', title: details.title || '' })
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'trend':
        return (
          <div>
            <button onClick={() => trendMutation.mutate()} disabled={trendMutation.isPending} className="btn-primary mb-4">
              {trendMutation.isPending ? '분석 중...' : '트렌드 분석 실행'}
            </button>
            {trendMutation.isPending && <LoadingSpinner text="트렌드 데이터를 분석 중..." />}
            {trendMutation.isError && <ErrorMessage text="트렌드 데이터를 불러올 수 없습니다" />}
            {trendMutation.data && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-xl font-bold mb-4">연간 검색량 추이</h3>
                  {trendMutation.data.yearly_trend ? <img src={trendMutation.data.yearly_trend} alt="Yearly Trend" className="w-full h-auto rounded-lg shadow-md" /> : <p>데이터 없음</p>}
                </div>
                <div>
                  <h3 className="text-xl font-bold mb-4">축제 기간 중심 추이</h3>
                  {trendMutation.data.event_trend ? <img src={trendMutation.data.event_trend} alt="Event Trend" className="w-full h-auto rounded-lg shadow-md" /> : <p>데이터 없음</p>}
                </div>
              </div>
            )}
          </div>
        )
      case 'sentiment': {
        const itemsPerPage = 5
        const paginatedBlogResults = sentimentMutation.data?.blog_results.slice(
          (blogTablePage - 1) * itemsPerPage,
          blogTablePage * itemsPerPage
        )
        const totalBlogPages = sentimentMutation.data ? Math.ceil(sentimentMutation.data.blog_results.length / itemsPerPage) : 0

        return (
          <div>
            <div className="mb-4 space-y-2">
              <label className="block text-sm font-medium text-slate-700">분석할 후기 수: <span className="font-bold text-primary-600">{numReviewsForSentiment}개</span></label>
              <input type="range" min="1" max="10" step="1" value={numReviewsForSentiment} onChange={(e) => setNumReviewsForSentiment(Number(e.target.value))} className="w-full accent-primary-600" />
              <button onClick={() => { setBlogTablePage(1); sentimentMutation.mutate(); }} disabled={sentimentMutation.isPending} className="btn-primary">
                {sentimentMutation.isPending ? '분석 중...' : '감성 분석 실행'}
              </button>
            </div>
            {sentimentMutation.isPending && <LoadingSpinner text="감성 분석 중..." />}
            {sentimentMutation.isError && <ErrorMessage text="감성 분석 데이터를 불러올 수 없습니다" />}
            {sentimentMutation.data && (
              <div className="space-y-12">
                <div>
                  <h3 className="text-2xl font-bold mb-4">종합 분석 결과</h3>
                  {sentimentMutation.data.overall_summary_text && (
                    <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded mb-4">
                      <div className="prose prose-slate max-w-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{sentimentMutation.data.overall_summary_text}</ReactMarkdown>
                      </div>
                    </div>
                  )}
                  <div className="prose prose-slate max-w-none bg-slate-50 p-4 rounded-lg">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{sentimentMutation.data.summary}</ReactMarkdown>
                  </div>
                </div>

                {/* Positive/Negative Summaries */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {sentimentMutation.data.positive_keywords && (
                    <div className="card p-6">
                      <h3 className="text-xl font-bold mb-4">👍 긍정적 피드백</h3>
                      <div dangerouslySetInnerHTML={{ __html: sentimentMutation.data.positive_keywords }} />
                    </div>
                  )}
                  {sentimentMutation.data.negative_summary && (
                    <div className="card p-6">
                      <h3 className="text-xl font-bold mb-4">👎 부정적 피드백 요약</h3>
                      <p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{sentimentMutation.data.negative_summary}</p>
                    </div>
                  )}
                </div>

                {/* Charts */}
                <div className="space-y-4">
                  <h3 className="text-2xl font-bold mb-4">리뷰 감성 분포 차트</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {sentimentMutation.data.charts.donut_data && (
                      <div className="card p-6">
                        <h4 className="text-lg font-semibold mb-4 text-center">전체 감성 분포</h4>
                        <SentimentDonutChart data={sentimentMutation.data.charts.donut_data} />
                      </div>
                    )}
                    {sentimentMutation.data.charts.satisfaction_data && (
                      <div className="card p-6">
                        <h4 className="text-lg font-semibold mb-4 text-center">상대적 만족도 분포</h4>
                        <SatisfactionBarChart data={sentimentMutation.data.charts.satisfaction_data} />
                      </div>
                    )}
                    {sentimentMutation.data.charts.absolute_data && (
                      <div className="card p-6">
                        <h4 className="text-lg font-semibold mb-4 text-center">절대 감성 점수 분포</h4>
                        <AbsoluteScoreLineChart data={sentimentMutation.data.charts.absolute_data} />
                      </div>
                    )}
                    {sentimentMutation.data.charts.outlier_data && (
                      <div className="card p-6">
                        <h4 className="text-lg font-semibold mb-4 text-center">감성 점수 이상치 분석</h4>
                        <OutlierBoxPlot data={sentimentMutation.data.charts.outlier_data} />
                      </div>
                    )}
                  </div>
                  {sentimentMutation.data.outlier_description && (
                    <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded mt-4">
                      <div className="prose prose-slate max-w-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{sentimentMutation.data.outlier_description}</ReactMarkdown>
                      </div>
                    </div>
                  )}
                </div>

                {/* Keyword Word Clouds */}
                <div className="space-y-4">
                  <h3 className="text-2xl font-bold mb-4">긍정/부정 키워드 (워드클라우드)</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {sentimentMutation.data.charts.wordcloud_positive && <img src={`data:image/png;base64,${sentimentMutation.data.charts.wordcloud_positive}`} alt="긍정 키워드" className="w-full h-auto rounded-lg shadow-md" />}
                    {sentimentMutation.data.charts.wordcloud_negative && <img src={`data:image/png;base64,${sentimentMutation.data.charts.wordcloud_negative}`} alt="부정 키워드" className="w-full h-auto rounded-lg shadow-md" />}
                  </div>
                </div>

                {/* Individual Blog Results Table */}
                {sentimentMutation.data.blog_results && sentimentMutation.data.blog_results.length > 0 && (
                  <div className="space-y-4">
                    <h3 className="text-2xl font-bold mb-4">개별 블로그 분석 결과</h3>
                    <div className="card overflow-x-auto">
                      <table className="w-full text-sm text-left text-slate-500">
                        <thead className="text-xs text-slate-700 uppercase bg-slate-50">
                          <tr>
                            <th scope="col" className="px-4 py-3">블로그 제목</th>
                            <th scope="col" className="px-4 py-3">링크</th>
                            <th scope="col" className="px-4 py-3">감성 빈도</th>
                            <th scope="col" className="px-4 py-3">감성 점수</th>
                            <th scope="col" className="px-4 py-3">긍정 문장 수</th>
                            <th scope="col" className="px-4 py-3">부정 문장 수</th>
                            <th scope="col" className="px-4 py-3">긍정 비율 (%)</th>
                            <th scope="col" className="px-4 py-3">부정 비율 (%)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {paginatedBlogResults?.map((blog, index) => {
                            const globalIndex = (blogTablePage - 1) * itemsPerPage + index
                            const isSelected = selectedBlogIndex === globalIndex
                            return (
                              <>
                                <tr
                                  key={index}
                                  onClick={() => setSelectedBlogIndex(isSelected ? null : globalIndex)}
                                  className={`bg-white border-b hover:bg-slate-100 cursor-pointer transition-colors ${isSelected ? 'bg-blue-50' : ''}`}
                                >
                                  <td className="px-4 py-4 font-medium text-slate-900">{blog['블로그 제목']}</td>
                                  <td className="px-4 py-4">
                                    <a
                                      href={blog['링크']}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="font-medium text-primary-600 hover:underline"
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      바로가기
                                    </a>
                                  </td>
                                  <td className="px-4 py-4">{blog['감성 빈도']}</td>
                                  <td className="px-4 py-4">{blog['감성 점수']}</td>
                                  <td className="px-4 py-4 text-green-600">{blog['긍정 문장 수']}</td>
                                  <td className="px-4 py-4 text-red-600">{blog['부정 문장 수']}</td>
                                  <td className="px-4 py-4">{blog['긍정 비율 (%)']}</td>
                                  <td className="px-4 py-4">{blog['부정 비율 (%)']}</td>
                                </tr>
                                {isSelected && (
                                  <tr key={`detail-${index}`}>
                                    <td colSpan={8} className="px-4 py-4 bg-slate-50">
                                      <div className="space-y-4">
                                        <h4 className="font-bold text-lg text-slate-800">개별 블로그 상세 분석</h4>
                                        {blog['긍/부정 문장 요약'] && (
                                          <div className="bg-white p-4 rounded-lg shadow-sm">
                                            <h5 className="font-semibold mb-2 text-slate-700">긍정/부정 문장 요약</h5>
                                            <div
                                              className="text-sm text-slate-600 prose prose-slate max-w-none"
                                              dangerouslySetInnerHTML={{
                                                __html: blog['긍/부정 문장 요약'].replace(/<br>---<br>/g, '<hr class="my-2 border-slate-200" />')
                                              }}
                                            />
                                          </div>
                                        )}
                                      </div>
                                    </td>
                                  </tr>
                                )}
                              </>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                    {/* Pagination Controls */}
                    <div className="flex justify-between items-center mt-4">
                      <button
                        onClick={() => setBlogTablePage(p => Math.max(1, p - 1))}
                        disabled={blogTablePage === 1}
                        className="btn-secondary"
                      >
                        이전
                      </button>
                      <span className="text-sm text-slate-700">
                        페이지 {blogTablePage} / {totalBlogPages}
                      </span>
                      <button
                        onClick={() => setBlogTablePage(p => Math.min(totalBlogPages, p + 1))}
                        disabled={blogTablePage === totalBlogPages}
                        className="btn-secondary"
                      >
                        다음
                      </button>
                    </div>
                    {sentimentMutation.data.blog_list_csv_path && (
                       <a href={`/${sentimentMutation.data.blog_list_csv_path}`} download className="btn-outline mt-4 inline-block">
                         전체 목록 CSV 다운로드
                       </a>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )
      }
      case 'images':
        return (
          <div>
            <div className="mb-4 space-y-2">
              <label className="block text-sm font-medium text-slate-700">이미지 수집 대상 블로그 수: <span className="font-bold text-primary-600">{numBlogsForImages}개</span></label>
              <input type="range" min="1" max="10" value={numBlogsForImages} onChange={(e) => setNumBlogsForImages(Number(e.target.value))} className="w-full accent-primary-600" />
              <button onClick={() => imagesMutation.mutate()} disabled={imagesMutation.isPending} className="btn-primary">
                {imagesMutation.isPending ? '수집 중...' : '이미지 수집 실행'}
              </button>
            </div>
            {imagesMutation.isPending && <LoadingSpinner text="블로그에서 이미지를 수집 중..." />}
            {imagesMutation.isError && <ErrorMessage text="이미지를 불러올 수 없습니다" />}
            {imagesMutation.data && (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {imagesMutation.data.image_urls.map((url, index) => (
                  <motion.div key={index} whileHover={{ scale: 1.05 }} className="card overflow-hidden shadow-lg">
                    <img src={url} alt={`Scraped festival image ${index + 1}`} className="w-full h-48 object-cover" />
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        )
      case 'wordcloud':
        return (
          <div>
            <div className="mb-4 space-y-2">
              <label className="block text-sm font-medium text-slate-700">워드클라우드 분석용 후기 수: <span className="font-bold text-primary-600">{numReviewsForWordCloud}개</span></label>
              <input type="range" min="1" max="10" step="1" value={numReviewsForWordCloud} onChange={(e) => setNumReviewsForWordCloud(Number(e.target.value))} className="w-full accent-primary-600" />
              <button onClick={() => wordCloudMutation.mutate()} disabled={wordCloudMutation.isPending} className="btn-primary">
                {wordCloudMutation.isPending ? '생성 중...' : '워드클라우드 생성 실행'}
              </button>
            </div>
            {wordCloudMutation.isPending && <LoadingSpinner text="워드클라우드를 생성 중..." />}
            {wordCloudMutation.isError && <ErrorMessage text="워드클라우드를 생성할 수 없습니다" />}
            {wordCloudMutation.data && (
              <div className="flex justify-center">
                <img src={wordCloudMutation.data.wordcloud} alt="Festival Word Cloud" className="w-1/2 h-auto rounded-lg shadow-md" />
              </div>
            )}
          </div>
        )
      case 'summary':
        return (
          <div>
            <div className="mb-4 space-y-2">
              <label className="block text-sm font-medium text-slate-700">후기 요약용 후기 수: <span className="font-bold text-primary-600">{numReviewsForSummary}개</span></label>
              <input type="range" min="1" max="10" value={numReviewsForSummary} onChange={(e) => setNumReviewsForSummary(Number(e.target.value))} className="w-full accent-primary-600" />
              <button onClick={() => summaryMutation.mutate()} disabled={summaryMutation.isPending} className="btn-primary">
                {summaryMutation.isPending ? '요약 중...' : '후기 요약 실행'}
              </button>
            </div>
            {summaryMutation.isPending && <LoadingSpinner text="AI가 후기를 요약 중..." />}
            {summaryMutation.isError && <ErrorMessage text="후기 요약을 불러올 수 없습니다" />}
            {summaryMutation.data && (
              <div className="prose prose-slate max-w-none bg-slate-50 p-6 rounded-lg">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{summaryMutation.data.summary}</ReactMarkdown>
              </div>
            )}
          </div>
        )
      case 'nearby':
        return (
          <div>
            <div className="mb-4 space-y-2">
              <label className="block text-sm font-medium text-slate-700">주변 검색 반경: <span className="font-bold text-primary-600">{(nearbyRadius / 1000).toFixed(1)}km</span></label>
              <input type="range" min="1000" max="20000" step="1000" value={nearbyRadius} onChange={(e) => setNearbyRadius(Number(e.target.value))} className="w-full accent-primary-600" />
              <button onClick={() => nearbyMutation.mutate()} disabled={nearbyMutation.isPending} className="btn-primary">
                {nearbyMutation.isPending ? '검색 중...' : '주변 추천 검색'}
              </button>
            </div>
            {nearbyMutation.isPending && <LoadingSpinner text="주변 추천 장소를 검색 중..." />}
            {nearbyMutation.isError && <ErrorMessage text="주변 추천 정보를 가져올 수 없습니다." />}
            {nearbyMutation.data && (
              <div className="space-y-8">
                {(['facilities', 'courses', 'festivals'] as const).map(type => (
                  nearbyMutation.data[type] && nearbyMutation.data[type].length > 0 && (
                    <div key={type}>
                      <h3 className="text-2xl font-bold mb-4 capitalize">{type === 'facilities' ? '주변 관광지' : type === 'courses' ? '주변 추천코스' : '주변 축제'}</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {nearbyMutation.data[type].map((item: any, index: number) => {
                          const itemType = type === 'facilities' ? 'facility' : type.slice(0, -1) as 'course' | 'festival';
                          const cardItem = {
                            ...item,
                            type: itemType,
                            image: item.firstimage || 'https://placehold.co/400x300?text=No+Image'
                          };
                          return <FestivalCard key={item.title || index} item={cardItem} index={index} />
                        })}
                      </div>
                    </div>
                  )
                ))}
              </div>
            )}
          </div>
        )
      case 'precautions':
        return (
          <div>
            <button onClick={() => precautionsMutation.mutate()} disabled={precautionsMutation.isPending} className="btn-primary mb-4">
              {precautionsMutation.isPending ? '분석 중...' : 'AI 주의사항 생성'}
            </button>
            {precautionsMutation.isPending && <LoadingSpinner text="AI가 주의사항을 분석 중..." />}
            {precautionsMutation.isError && <ErrorMessage text="주의사항을 불러올 수 없습니다" />}
            {precautionsMutation.data && (
              <div className="prose max-w-none">
                <div className="bg-amber-50 border-l-4 border-amber-500 p-6 rounded-r-lg">
                  <div className="flex items-start space-x-3">
                    <FaExclamationTriangle className="text-amber-600 text-xl mt-1 flex-shrink-0" />
                    <div className="whitespace-pre-wrap text-slate-700 leading-relaxed">{precautionsMutation.data}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )
      case 'rendering':
        return (
          <div>
            <button onClick={() => renderingMutation.mutate()} disabled={renderingMutation.isPending} className="btn-primary mb-4">
              {renderingMutation.isPending ? '생성 중...' : 'AI 렌더링 실행'}
            </button>
            {renderingMutation.isPending && <LoadingSpinner text="AI가 이미지를 생성 중... (최대 2분 소요)" />}
            {renderingMutation.isError && <ErrorMessage text="이미지를 생성할 수 없습니다" />}
            {renderingMutation.data && (
              <div className="space-y-8">
                {/* Representative Image */}
                {renderingMutation.data.representative_image && (
                  <div className="space-y-4">
                    <h3 className="text-2xl font-bold text-primary-700">대표 렌더링</h3>
                    <div className="card p-4">
                      <img src={`data:image/png;base64,${renderingMutation.data.representative_image.image_base64}`} alt="AI Representative Rendering" className="w-full h-auto rounded-lg shadow-md" />
                      {renderingMutation.data.representative_image.prompt && <div className="mt-2 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg"><strong>프롬프트:</strong> {renderingMutation.data.representative_image.prompt}</div>}
                    </div>
                  </div>
                )}

                {/* Conditional Images */}
                {renderingMutation.data.conditional_images && renderingMutation.data.conditional_images.length > 0 && (
                  <div className="space-y-4">
                    <h3 className="text-2xl font-bold text-primary-700">조건부 렌더링</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {renderingMutation.data.conditional_images.map((img, index) => (
                        <div key={index} className="card p-4">
                          <img src={`data:image/png;base64,${img.image_base64}`} alt={`AI Conditional Rendering ${index + 1}`} className="w-full h-auto rounded-lg shadow-md" />
                          {img.prompt && <div className="mt-2 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg"><strong>프롬프트:</strong> {img.prompt}</div>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )
      default:
        return null
    }
  }

  const tabs: { id: TabType; icon: React.ReactNode; label: string }[] = [
    { id: 'images', icon: <FaCamera />, label: '이미지' },
    { id: 'trend', icon: <FaChartLine />, label: '트렌드 분석' },
    { id: 'wordcloud', icon: <FaCloud />, label: '워드클라우드' },
    { id: 'sentiment', icon: <FaHeart />, label: '감성 분석' },
    { id: 'summary', icon: <FaCommentDots />, label: '후기 요약' },
    { id: 'nearby', icon: <FaStreetView />, label: '주변 추천' },
    { id: 'precautions', icon: <FaExclamationTriangle />, label: 'AI 주의사항' },
    { id: 'rendering', icon: <FaImage />, label: 'AI 렌더링' },
  ]

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
      {/* Hero Section */}
      <div className="card overflow-hidden">
        <div className="relative h-96">
          <img src={details.firstimage || 'https://placehold.co/1200x400?text=Festival'} alt={details.title} className="w-full h-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent" />
          <div className="absolute bottom-0 left-0 right-0 p-8 text-white">
            <div className="container mx-auto">
              <div className={`inline-block px-4 py-2 rounded-full text-sm font-medium mb-4 ${status.color}`}>{status.status}</div>
              <h1 className="text-4xl md:text-5xl font-bold mb-4">{details.title}</h1>
              <div className="flex flex-wrap items-center gap-4 text-lg">
                {details.eventstartdate && <div className="flex items-center space-x-2"><FaCalendarAlt /><span>{formatDateRange(details.eventstartdate, details.eventenddate)}</span></div>}
                {details.addr1 && <div className="flex items-center space-x-2"><FaMapMarkerAlt /><span>{details.addr1}</span></div>}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-4">
        <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={handleAddToCourse} disabled={isInCourse} className={`btn-primary flex items-center space-x-2 ${isInCourse ? 'opacity-50 cursor-not-allowed' : ''}`}>
          <FaPlus />
          <span>{isInCourse ? '이미 추가됨' : '내 코스에 추가'}</span>
        </motion.button>
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          {details.overview && <div className="card p-6"><h2 className="text-2xl font-bold mb-4">축제 소개</h2><p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{removeHtmlTags(details.overview)}</p></div>}
          {details.program && <div className="card p-6"><h2 className="text-2xl font-bold mb-4">프로그램</h2><p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{removeHtmlTags(details.program)}</p></div>}
        </div>
        <div className="space-y-6">
          { (details.addr1 || details.mapx || details.mapy || details.tel || details.homepage || (details.eventstartdate && details.eventenddate) || details.eventplace || details.usetimefestival || details.playtime || details.agelimit || details.spendtimefestival || details.festivalgrade || details.eventhomepage) && (
            <div className="card p-6">
              <h3 className="text-xl font-bold mb-4">기본 정보</h3>
              <dl className="space-y-3">
              {details.addr1 && <div><dt className="text-sm text-slate-500 mb-1">주소</dt><dd className="text-slate-900">{details.addr1}</dd></div>}
              {details.mapx && details.mapy && <div><dt className="text-sm text-slate-500 mb-1">좌표</dt><dd className="text-slate-900">{details.mapy}, {details.mapx}</dd></div>}
              {details.tel && <div><dt className="text-sm text-slate-500 mb-1">전화번호</dt><dd className="flex items-center space-x-2"><FaPhone className="text-primary-600" /><a href={`tel:${details.tel}`} className="text-slate-900 hover:text-primary-600">{details.tel}</a></dd></div>}
              {details.homepage && <div><dt className="text-sm text-slate-500 mb-1">홈페이지</dt><dd className="flex items-center space-x-2"><FaGlobe className="text-primary-600" /><a href={removeHtmlTags(details.homepage).match(/href="([^"]+)"/)?.[1] || '#'} target="_blank" rel="noopener noreferrer" className="text-slate-900 hover:text-primary-600 truncate">바로가기</a></dd></div>}
              {details.eventstartdate && details.eventenddate && <div><dt className="text-sm text-slate-500 mb-1">행사 기간</dt><dd className="text-slate-900">{formatDateRange(details.eventstartdate, details.eventenddate)}</dd></div>}
              {details.eventplace && <div><dt className="text-sm text-slate-500 mb-1">행사 장소</dt><dd className="text-slate-900">{details.eventplace}</dd></div>}
              {details.usetimefestival && <div><dt className="text-sm text-slate-500 mb-1">이용 시간</dt><dd className="text-slate-900">{removeHtmlTags(details.usetimefestival)}</dd></div>}
              {details.playtime && <div><dt className="text-sm text-slate-500 mb-1">공연 시간</dt><dd className="text-slate-900">{removeHtmlTags(details.playtime)}</dd></div>}
              {details.agelimit && <div><dt className="text-sm text-slate-500 mb-1">연령 제한</dt><dd className="text-slate-900">{removeHtmlTags(details.agelimit)}</dd></div>}
              {details.spendtimefestival && <div><dt className="text-sm text-slate-500 mb-1">소요 시간</dt><dd className="text-slate-900">{removeHtmlTags(details.spendtimefestival)}</dd></div>}
              {details.festivalgrade && <div><dt className="text-sm text-slate-500 mb-1">축제 등급</dt><dd className="text-slate-900">{removeHtmlTags(details.festivalgrade)}</dd></div>}
              {details.eventhomepage && <div><dt className="text-sm text-slate-500 mb-1">행사 홈페이지</dt><dd className="text-slate-900"><a href={removeHtmlTags(details.eventhomepage).match(/href="([^"]+)"/)?.[1] || '#'} target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline">바로가기</a></dd></div>}
            </dl>
          </div>
          )} {/* End of Basic Information Section conditional */}
                    <div className="card p-6 bg-gradient-to-br from-primary-50 to-accent-50">
                      <h3 className="text-lg font-bold mb-3 flex items-center space-x-2"><span>💡</span><span>알아두세요</span></h3>
                      <p className="text-sm text-slate-700 leading-relaxed">축제 일정과 프로그램은 현장 사정에 따라 변경될 수 있습니다. 방문 전 공식 홈페이지나 전화로 확인하시는 것을 권장합니다.</p>
                    </div>
                    {/* Additional Information Section */}
                    { (details.sponsor1 || details.sponsor1tel || details.sponsor2 || details.sponsor2tel || details.subevent || details.discountinfofestival || details.placeinfo || details.bookingplace) && (
                      <div className="card p-6">
                        <h3 className="text-xl font-bold mb-4">추가 정보</h3>
                        <dl className="space-y-3">
                        {details.sponsor1 && <div><dt className="text-sm text-slate-500 mb-1">주최</dt><dd className="text-slate-900">{details.sponsor1}</dd></div>}
                        {details.sponsor1tel && <div><dt className="text-sm text-slate-500 mb-1">주최 전화번호</dt><dd className="text-slate-900">{details.sponsor1tel}</dd></div>}
                        {details.sponsor2 && <div><dt className="text-sm text-slate-500 mb-1">주관</dt><dd className="text-slate-900">{details.sponsor2}</dd></div>}
                        {details.sponsor2tel && <div><dt className="text-sm text-slate-500 mb-1">주관 전화번호</dt><dd className="text-slate-900">{details.sponsor2tel}</dd></div>}
                        {details.subevent && <div><dt className="text-sm text-slate-500 mb-1">부대 행사</dt><dd className="text-slate-900">{removeHtmlTags(details.subevent)}</dd></div>}
                        {details.discountinfofestival && <div><dt className="text-sm text-slate-500 mb-1">할인 정보</dt><dd className="text-slate-900">{removeHtmlTags(details.discountinfofestival)}</dd></div>}
                        {details.placeinfo && <div><dt className="text-sm text-slate-500 mb-1">장소 정보</dt><dd className="text-slate-900">{removeHtmlTags(details.placeinfo)}</dd></div>}
                        {details.bookingplace && <div><dt className="text-sm text-slate-500 mb-1">예약처</dt><dd className="text-slate-900">{removeHtmlTags(details.bookingplace)}</dd></div>}
                      </dl>
                    </div>
                    )} {/* End of Additional Information Section conditional */}
                  </div>
                </div>
                    {/* Tabs Section */}
      <div className="card overflow-hidden">
        <div className="flex border-b border-slate-200 overflow-x-auto">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-6 py-4 font-medium transition-colors whitespace-nowrap ${activeTab === tab.id ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50' : 'text-slate-600 hover:text-primary-600 hover:bg-slate-50'}`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
        <div className="p-6">
          {renderTabContent()}
        </div>
      </div>
    </motion.div>
  )
}
