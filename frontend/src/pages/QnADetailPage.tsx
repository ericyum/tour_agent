import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  MessageSquare,
  ArrowLeft,
  User,
  Clock,
  Eye,
  Check,
  Edit2,
  Trash2,
  Shield,
  Send,
} from 'lucide-react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getQuestion,
  createAnswer,
  deleteQuestion,
  deleteAnswer,
  acceptAnswer,
  type Answer,
} from '@/lib/api'
import { useAuthStore } from '@/store/useAuthStore'

export default function QnADetailPage() {
  const { questionId } = useParams<{ questionId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { user, accessToken, isAuthenticated, isAdmin } = useAuthStore()
  const [newAnswer, setNewAnswer] = useState('')

  const { data: question, isLoading, isError } = useQuery({
    queryKey: ['question', questionId],
    queryFn: () => getQuestion(Number(questionId)),
    enabled: !!questionId,
  })

  const answerMutation = useMutation({
    mutationFn: () => createAnswer(Number(questionId), newAnswer, accessToken!),
    onSuccess: () => {
      setNewAnswer('')
      queryClient.invalidateQueries({ queryKey: ['question', questionId] })
    },
    onError: (error: any) => {
      alert(error.response?.data?.detail || '답변 등록에 실패했습니다.')
    },
  })

  const deleteQuestionMutation = useMutation({
    mutationFn: () => deleteQuestion(Number(questionId), accessToken!),
    onSuccess: () => {
      alert('질문이 삭제되었습니다.')
      navigate('/qna')
    },
    onError: (error: any) => {
      alert(error.response?.data?.detail || '삭제에 실패했습니다.')
    },
  })

  const deleteAnswerMutation = useMutation({
    mutationFn: (answerId: number) => deleteAnswer(answerId, accessToken!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['question', questionId] })
    },
    onError: (error: any) => {
      alert(error.response?.data?.detail || '삭제에 실패했습니다.')
    },
  })

  const acceptMutation = useMutation({
    mutationFn: (answerId: number) => acceptAnswer(answerId, accessToken!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['question', questionId] })
    },
    onError: (error: any) => {
      alert(error.response?.data?.detail || '채택에 실패했습니다.')
    },
  })

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return ''
    return new Date(dateStr).toLocaleString('ko-KR')
  }

  const handleSubmitAnswer = (e: React.FormEvent) => {
    e.preventDefault()
    if (!newAnswer.trim()) {
      alert('답변 내용을 입력해주세요.')
      return
    }
    answerMutation.mutate()
  }

  const handleDeleteQuestion = () => {
    if (confirm('정말로 이 질문을 삭제하시겠습니까?')) {
      deleteQuestionMutation.mutate()
    }
  }

  const handleDeleteAnswer = (answerId: number) => {
    if (confirm('정말로 이 답변을 삭제하시겠습니까?')) {
      deleteAnswerMutation.mutate(answerId)
    }
  }

  const handleAcceptAnswer = (answerId: number) => {
    if (confirm('이 답변을 채택하시겠습니까?')) {
      acceptMutation.mutate(answerId)
    }
  }

  const canEditQuestion = question && user && (question.user_id === user.id || isAdmin)
  const canDeleteQuestion = canEditQuestion

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="bg-white rounded-xl shadow-lg p-12 text-center border border-gray-200">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">질문을 불러오는 중...</p>
        </div>
      </div>
    )
  }

  if (isError || !question) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="bg-white rounded-xl shadow-lg p-12 text-center border border-red-200">
          <MessageSquare size={64} className="text-gray-300 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">질문을 찾을 수 없습니다</h2>
          <Link to="/qna">
            <button className="mt-4 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
              목록으로 돌아가기
            </button>
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* Back Button */}
        <Link to="/qna">
          <button className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors">
            <ArrowLeft size={20} />
            목록으로 돌아가기
          </button>
        </Link>

        {/* Question */}
        <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
          <div className="flex items-start justify-between mb-4">
            <h1 className="text-2xl font-bold text-gray-900">{question.title}</h1>
            {canDeleteQuestion && (
              <div className="flex gap-2">
                <button
                  onClick={handleDeleteQuestion}
                  className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                  title="삭제"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            )}
          </div>

          <div className="flex items-center gap-4 text-sm text-gray-500 mb-6">
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
              조회 {question.views}
            </span>
          </div>

          <div className="prose max-w-none text-gray-700 whitespace-pre-wrap">
            {question.content}
          </div>
        </div>

        {/* Answers */}
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <MessageSquare size={24} />
            답변 {question.answers.length}개
          </h2>

          {question.answers.length === 0 ? (
            <div className="bg-gray-50 rounded-xl p-8 text-center border border-gray-200">
              <p className="text-gray-500">아직 답변이 없습니다. 첫 번째로 답변해주세요!</p>
            </div>
          ) : (
            question.answers.map((answer: Answer, index: number) => (
              <motion.div
                key={answer.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`bg-white rounded-xl shadow-md p-6 border-2 ${
                  answer.is_accepted ? 'border-green-500' : 'border-gray-200'
                }`}
              >
                {answer.is_accepted && (
                  <div className="flex items-center gap-2 text-green-600 mb-3">
                    <Check size={20} />
                    <span className="font-medium">채택된 답변</span>
                  </div>
                )}

                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
                      {answer.author.role === 'admin' ? (
                        <Shield size={20} className="text-blue-600" />
                      ) : (
                        <User size={20} className="text-blue-600" />
                      )}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-gray-900">
                          {answer.author.username}
                        </span>
                        {answer.author.role === 'admin' && (
                          <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full">
                            관리자
                          </span>
                        )}
                      </div>
                      <span className="text-sm text-gray-500">
                        {formatDate(answer.created_at)}
                      </span>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    {/* Accept button - only for question author */}
                    {!answer.is_accepted &&
                      user &&
                      question.user_id === user.id && (
                        <button
                          onClick={() => handleAcceptAnswer(answer.id)}
                          className="p-2 text-green-500 hover:bg-green-50 rounded-lg transition-colors"
                          title="답변 채택"
                        >
                          <Check size={18} />
                        </button>
                      )}
                    {/* Delete button - for answer author or admin */}
                    {user && (answer.user_id === user.id || isAdmin) && (
                      <button
                        onClick={() => handleDeleteAnswer(answer.id)}
                        className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                        title="삭제"
                      >
                        <Trash2 size={18} />
                      </button>
                    )}
                  </div>
                </div>

                <div className="prose max-w-none text-gray-700 whitespace-pre-wrap">
                  {answer.content}
                </div>
              </motion.div>
            ))
          )}
        </div>

        {/* Answer Form */}
        {isAuthenticated ? (
          <form
            onSubmit={handleSubmitAnswer}
            className="bg-white rounded-xl shadow-lg p-6 border border-gray-200"
          >
            <h3 className="text-lg font-bold text-gray-900 mb-4">답변 작성</h3>
            <textarea
              value={newAnswer}
              onChange={(e) => setNewAnswer(e.target.value)}
              placeholder="답변을 작성해주세요"
              rows={5}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none mb-4"
              maxLength={5000}
            />
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">{newAnswer.length}/5000</span>
              <motion.button
                type="submit"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                disabled={answerMutation.isPending}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-medium rounded-lg hover:from-blue-700 hover:to-indigo-700 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send size={18} />
                {answerMutation.isPending ? '등록 중...' : '답변 등록'}
              </motion.button>
            </div>
          </form>
        ) : (
          <div className="bg-gray-50 rounded-xl p-6 text-center border border-gray-200">
            <p className="text-gray-600 mb-4">답변을 작성하려면 로그인이 필요합니다.</p>
            <Link to="/login">
              <button className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                로그인하기
              </button>
            </Link>
          </div>
        )}
      </motion.div>
    </div>
  )
}
