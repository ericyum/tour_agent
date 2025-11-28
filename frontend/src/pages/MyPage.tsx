import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { useAuthStore } from '@/store/useAuthStore';
import { qnaService, authService } from '@/lib/authApi';
import { User, MessageSquare, CheckCircle, Calendar, Trash2, AlertTriangle, X } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';

export default function MyPage() {
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();
  const [activeTab, setActiveTab] = useState<'questions' | 'answers'>('questions');
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const { data: myQuestions } = useQuery({
    queryKey: ['my-questions'],
    queryFn: () => qnaService.getMyQuestions(),
  });

  const { data: myAnswers } = useQuery({
    queryKey: ['my-answers'],
    queryFn: () => qnaService.getMyAnswers(),
  });

  const handleDeleteAccount = async () => {
    setIsDeleting(true);
    try {
      // TODO: Implement deleteAccount API
      // await authService.deleteAccount();
      alert('계정 삭제 기능은 현재 개발 중입니다.');
      logout();
      navigate('/');
    } catch (error) {
      console.error('Failed to delete account:', error);
      alert('계정 삭제에 실패했습니다. 다시 시도해주세요.');
    } finally {
      setIsDeleting(false);
      setShowDeleteModal(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-8"
      >
        {/* Profile Card */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl p-8 text-white shadow-xl">
          <div className="flex items-center space-x-4">
            <div className="h-20 w-20 bg-white rounded-full flex items-center justify-center">
              <User size={40} className="text-blue-600" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">{user?.full_name || user?.username}</h1>
              <p className="text-blue-100 mt-1">{user?.email}</p>
              <p className="text-sm text-blue-200 mt-2">
                가입일: {user?.created_at ? new Date(user.created_at).toLocaleDateString('ko-KR') : '-'}
              </p>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-xl p-6 shadow-lg border border-gray-200"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">내가 작성한 질문</p>
                <p className="text-3xl font-bold text-blue-600 mt-2">
                  {myQuestions?.questions?.length || 0}개
                </p>
              </div>
              <MessageSquare size={40} className="text-blue-600" />
            </div>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-xl p-6 shadow-lg border border-gray-200"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">내가 작성한 답변</p>
                <p className="text-3xl font-bold text-green-600 mt-2">
                  {myAnswers?.answers?.length || 0}개
                </p>
              </div>
              <CheckCircle size={40} className="text-green-600" />
            </div>
          </motion.div>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="flex border-b">
            <button
              onClick={() => setActiveTab('questions')}
              className={`flex-1 px-6 py-4 font-medium transition-colors ${
                activeTab === 'questions'
                  ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              내 질문 ({myQuestions?.questions?.length || 0})
            </button>
            <button
              onClick={() => setActiveTab('answers')}
              className={`flex-1 px-6 py-4 font-medium transition-colors ${
                activeTab === 'answers'
                  ? 'bg-green-50 text-green-600 border-b-2 border-green-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              내 답변 ({myAnswers?.answers?.length || 0})
            </button>
          </div>

          <div className="p-6">
            {activeTab === 'questions' && (
              <div className="space-y-4">
                {myQuestions?.questions && myQuestions.questions.length > 0 ? (
                  myQuestions.questions.map((question: any) => (
                    <Link
                      key={question.id}
                      to={`/qna?question=${question.id}`}
                      className="block p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-900 mb-1">{question.title}</h3>
                          <p className="text-sm text-gray-600 mb-2 line-clamp-2">{question.content}</p>
                          <div className="flex items-center gap-4 text-xs text-gray-500">
                            <span className="flex items-center gap-1">
                              <Calendar size={14} />
                              {new Date(question.created_at).toLocaleDateString('ko-KR')}
                            </span>
                            <span>답변 {question.answer_count}개</span>
                            <span>조회 {question.views}회</span>
                          </div>
                        </div>
                        <span className="ml-4 px-3 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">
                          {question.festival_name}
                        </span>
                      </div>
                    </Link>
                  ))
                ) : (
                  <p className="text-center text-gray-500 py-12">작성한 질문이 없습니다.</p>
                )}
              </div>
            )}

            {activeTab === 'answers' && (
              <div className="space-y-4">
                {myAnswers?.answers && myAnswers.answers.length > 0 ? (
                  myAnswers.answers.map((answer: any) => (
                    <Link
                      key={answer.id}
                      to={`/qna?question=${answer.question_id}`}
                      className="block p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-medium text-gray-700 text-sm">Q. {answer.question_title}</h4>
                        {answer.is_accepted && (
                          <span className="ml-2 px-2 py-0.5 bg-green-100 text-green-700 text-xs font-medium rounded-full flex items-center gap-1">
                            <CheckCircle size={12} />
                            채택됨
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-900 mb-2 line-clamp-2">{answer.content}</p>
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span className="flex items-center gap-1">
                          <Calendar size={14} />
                          {new Date(answer.created_at).toLocaleDateString('ko-KR')}
                        </span>
                        <span className="px-2 py-0.5 bg-gray-100 text-gray-700 rounded">
                          {answer.festival_name}
                        </span>
                      </div>
                    </Link>
                  ))
                ) : (
                  <p className="text-center text-gray-500 py-12">작성한 답변이 없습니다.</p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Account Deletion Section */}
        <div className="bg-red-50 border border-red-200 rounded-xl p-6">
          <div className="flex items-start justify-between">
            <div>
              <h2 className="text-lg font-bold text-red-900 mb-2 flex items-center gap-2">
                <AlertTriangle size={20} />
                계정 삭제
              </h2>
              <p className="text-sm text-red-700 mb-4">
                계정을 삭제하면 모든 데이터가 영구적으로 삭제됩니다. 이 작업은 되돌릴 수 없습니다.
              </p>
              <ul className="text-xs text-red-600 space-y-1 mb-4">
                <li>• 작성한 모든 질문과 답변이 삭제됩니다</li>
                <li>• 제공한 피드백 데이터는 통계를 위해 익명화되어 보관됩니다</li>
                <li>• 계정 정보는 즉시 삭제됩니다</li>
              </ul>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowDeleteModal(true)}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors flex items-center gap-2"
            >
              <Trash2 size={18} />
              계정 삭제
            </motion.button>
          </div>
        </div>
      </motion.div>

      {/* Delete Confirmation Modal */}
      <AnimatePresence>
        {showDeleteModal && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black bg-opacity-50 z-50"
              onClick={() => !isDeleting && setShowDeleteModal(false)}
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="fixed inset-0 z-50 flex items-center justify-center p-4"
            >
              <div className="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-red-100 rounded-full">
                      <AlertTriangle className="text-red-600" size={24} />
                    </div>
                    <h3 className="text-xl font-bold text-gray-900">
                      정말 삭제하시겠습니까?
                    </h3>
                  </div>
                  {!isDeleting && (
                    <button
                      onClick={() => setShowDeleteModal(false)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <X size={24} />
                    </button>
                  )}
                </div>

                <div className="mb-6">
                  <p className="text-gray-700 mb-4">
                    <strong>{user?.username}</strong> 계정을 삭제하려고 합니다.
                  </p>
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <p className="text-sm text-red-800 font-semibold mb-2">
                      ⚠️ 이 작업은 되돌릴 수 없습니다!
                    </p>
                    <ul className="text-xs text-red-700 space-y-1">
                      <li>• 모든 질문과 답변 삭제</li>
                      <li>• 계정 정보 영구 삭제</li>
                      <li>• 즉시 로그아웃</li>
                    </ul>
                  </div>
                </div>

                <div className="flex gap-3">
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setShowDeleteModal(false)}
                    disabled={isDeleting}
                    className="flex-1 px-4 py-3 bg-gray-200 hover:bg-gray-300 text-gray-800 font-medium rounded-lg transition-colors disabled:opacity-50"
                  >
                    취소
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleDeleteAccount}
                    disabled={isDeleting}
                    className="flex-1 px-4 py-3 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {isDeleting ? (
                      <>삭제 중...</>
                    ) : (
                      <>
                        <Trash2 size={18} />
                        영구 삭제
                      </>
                    )}
                  </motion.button>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
