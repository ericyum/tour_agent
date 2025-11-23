import { useState } from 'react'
import { motion } from 'framer-motion'
import { MessageSquare, Search, Plus, Eye, MessageCircle, User, Clock } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getQuestions } from '@/lib/api'
import { useAuthStore } from '@/store/useAuthStore'

export default function QnAPage() {
  const [searchTerm, setSearchTerm] = useState('')
  const { isAuthenticated } = useAuthStore()

  const { data, isLoading, isError } = useQuery({
    queryKey: ['questions'],
    queryFn: () => getQuestions(50, 0),
  })

  const filteredQuestions = data?.questions.filter(
    (q) =>
      q.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      q.content.toLowerCase().includes(searchTerm.toLowerCase())
  ) || []

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return ''
    const date = new Date(dateStr)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(diff / 3600000)
    const days = Math.floor(diff / 86400000)

    if (minutes < 1) return '방금 전'
    if (minutes < 60) return `${minutes}분 전`
    if (hours < 24) return `${hours}시간 전`
    if (days < 7) return `${days}일 전`
    return date.toLocaleDateString('ko-KR')
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-8"
      >
        {/* Header */}
        <div className="text-center">
          <div className="inline-block p-4 bg-gradient-to-r from-blue-100 to-indigo-100 rounded-full mb-4">
            <MessageSquare size={40} className="text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Q&A 게시판</h1>
          <p className="text-gray-600">궁금한 점이나 건의사항을 남겨주세요</p>
        </div>

        {/* Search and Actions */}
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="text-gray-400" size={20} />
            </div>
            <input
              type="text"
              placeholder="질문 검색..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {isAuthenticated ? (
            <Link to="/qna/new">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-medium rounded-lg hover:from-blue-700 hover:to-indigo-700 shadow-lg whitespace-nowrap"
              >
                <Plus size={20} />
                질문하기
              </motion.button>
            </Link>
          ) : (
            <Link to="/login">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="flex items-center gap-2 px-6 py-3 bg-gray-500 text-white font-medium rounded-lg hover:bg-gray-600 shadow-lg whitespace-nowrap"
              >
                <Plus size={20} />
                로그인 후 질문하기
              </motion.button>
            </Link>
          )}
        </div>

        {/* Questions List */}
        {isLoading ? (
          <div className="bg-white rounded-xl shadow-lg p-12 text-center border border-gray-200">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">질문 목록을 불러오는 중...</p>
          </div>
        ) : isError ? (
          <div className="bg-white rounded-xl shadow-lg p-12 text-center border border-red-200">
            <p className="text-red-600">질문 목록을 불러오는데 실패했습니다.</p>
          </div>
        ) : filteredQuestions.length === 0 ? (
          <div className="bg-white rounded-xl shadow-lg p-12 text-center border border-gray-200">
            <MessageSquare size={64} className="text-gray-300 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              {searchTerm ? '검색 결과가 없습니다' : '아직 질문이 없습니다'}
            </h2>
            <p className="text-gray-600 mb-4">
              {searchTerm
                ? '다른 키워드로 검색해보세요'
                : '첫 번째 질문을 남겨보세요!'}
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredQuestions.map((question, index) => (
              <motion.div
                key={question.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Link to={`/qna/${question.id}`}>
                  <div className="bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow p-6 border border-gray-200 hover:border-blue-300">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2 truncate">
                          {question.title}
                        </h3>
                        <p className="text-gray-600 text-sm line-clamp-2 mb-3">
                          {question.content}
                        </p>
                        <div className="flex items-center gap-4 text-sm text-gray-500">
                          <span className="flex items-center gap-1">
                            <User size={14} />
                            {question.author.username}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock size={14} />
                            {formatDate(question.created_at)}
                          </span>
                          <span className="flex items-center gap-1">
                            <Eye size={14} />
                            {question.views}
                          </span>
                        </div>
                      </div>
                      <div className="flex-shrink-0 flex flex-col items-center">
                        <div
                          className={`px-3 py-2 rounded-lg ${
                            question.answer_count > 0
                              ? 'bg-green-100 text-green-700'
                              : 'bg-gray-100 text-gray-500'
                          }`}
                        >
                          <MessageCircle size={20} className="mx-auto mb-1" />
                          <span className="text-sm font-medium">{question.answer_count}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        )}

        {/* Stats */}
        {data && data.total > 0 && (
          <div className="text-center text-gray-500 text-sm">
            총 {data.total}개의 질문
          </div>
        )}
      </motion.div>
    </div>
  )
}
