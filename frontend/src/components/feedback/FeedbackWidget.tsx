import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ThumbsUp, ThumbsDown, X, LogIn } from 'lucide-react';
import { useAuthStore } from '@/store/useAuthStore';
import { useNavigate } from 'react-router-dom';

interface FeedbackWidgetProps {
  festivalName?: string;
}

export const FeedbackWidget: React.FC<FeedbackWidgetProps> = ({ festivalName }) => {
  const navigate = useNavigate();
  const { isAuthenticated, user } = useAuthStore();
  const [isOpen, setIsOpen] = useState(false);
  const [selectedRating, setSelectedRating] = useState<number | null>(null);
  const [comment, setComment] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showLoginPrompt, setShowLoginPrompt] = useState(false);

  const getSessionId = () => {
    let sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('session_id', sessionId);
    }
    return sessionId;
  };

  const handleRatingClick = async (rating: number) => {
    if (!isAuthenticated) {
      setShowLoginPrompt(true);
      return;
    }
    setSelectedRating(rating);
    setIsOpen(true);
  };

  const handleSubmit = async () => {
    if (selectedRating === null) return;
    if (!isAuthenticated) {
      setShowLoginPrompt(true);
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await fetch('http://localhost:8000/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          page_url: window.location.pathname,
          festival_name: festivalName,
          rating: selectedRating,
          comment: comment || null,
          user_agent: navigator.userAgent,
          session_id: getSessionId(),
          user_id: user?.id || null,
        }),
      });

      if (response.ok) {
        setIsSubmitted(true);
        setTimeout(() => {
          setIsOpen(false);
          setIsSubmitted(false);
          setSelectedRating(null);
          setComment('');
        }, 2000);
      }
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <AnimatePresence>
        {!isOpen && !isSubmitted && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="bg-white rounded-lg shadow-lg p-4 border border-gray-200"
          >
            <p className="text-sm font-medium text-gray-700 mb-3">
              이 페이지가 도움이 되었나요?
            </p>
            <div className="flex gap-3">
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleRatingClick(5)}
                className="flex items-center gap-2 px-4 py-2 bg-green-50 hover:bg-green-100 text-green-700 rounded-lg transition-colors"
              >
                <ThumbsUp size={20} />
                <span className="text-sm font-medium">도움됐어요</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleRatingClick(1)}
                className="flex items-center gap-2 px-4 py-2 bg-red-50 hover:bg-red-100 text-red-700 rounded-lg transition-colors"
              >
                <ThumbsDown size={20} />
                <span className="text-sm font-medium">아쉬워요</span>
              </motion.button>
            </div>
          </motion.div>
        )}

        {isOpen && !isSubmitted && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="bg-white rounded-lg shadow-xl p-6 border border-gray-200 w-80"
          >
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-900">
                피드백 감사합니다!
              </h3>
              <button
                onClick={() => {
                  setIsOpen(false);
                  setSelectedRating(null);
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                <X size={20} />
              </button>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                개선 제안이 있으신가요? (선택)
              </label>
              <textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                rows={3}
                placeholder="의견을 자유롭게 작성해주세요..."
              />
            </div>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors disabled:bg-gray-400"
            >
              {isSubmitting ? '전송 중...' : '제출하기'}
            </motion.button>
          </motion.div>
        )}

        {isSubmitted && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="bg-green-50 border border-green-200 rounded-lg shadow-lg p-6 w-80"
          >
            <div className="text-center">
              <div className="mb-2 text-green-600">
                <ThumbsUp size={32} className="mx-auto" />
              </div>
              <p className="text-lg font-semibold text-green-900 mb-1">
                감사합니다!
              </p>
              <p className="text-sm text-green-700">
                소중한 의견을 반영하겠습니다.
              </p>
            </div>
          </motion.div>
        )}

        {showLoginPrompt && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="bg-blue-50 border border-blue-200 rounded-lg shadow-xl p-6 w-80"
          >
            <div className="flex justify-between items-start mb-4">
              <div className="flex items-center gap-2">
                <LogIn className="text-blue-600" size={24} />
                <h3 className="text-lg font-semibold text-gray-900">
                  로그인이 필요합니다
                </h3>
              </div>
              <button
                onClick={() => setShowLoginPrompt(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X size={20} />
              </button>
            </div>

            <p className="text-sm text-gray-700 mb-4">
              피드백을 제출하려면 로그인이 필요합니다. 모든 기능은 자유롭게 이용하실 수 있지만, 피드백과 Q&A는 회원만 작성할 수 있습니다.
            </p>

            <div className="flex gap-2">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => navigate('/login')}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
              >
                로그인
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => navigate('/register')}
                className="flex-1 px-4 py-2 bg-white border-2 border-blue-600 text-blue-600 hover:bg-blue-50 font-medium rounded-lg transition-colors"
              >
                회원가입
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
