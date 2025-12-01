import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { ThumbsUp, ThumbsDown, Activity, TrendingUp, MessageCircle, HelpCircle, CheckCircle } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

interface AnalyticsSummary {
  summary: {
    total_feedback: number;
    total_ratings: number;
    total_events: number;
  };
  top_events: Array<{
    category: string;
    action: string;
    count: number;
  }>;
  popular_festivals: Array<{
    festival_name: string;
    views: number;
  }>;
}

interface FeatureRating {
  feature_name: string;
  avg_rating: number;
  count: number;
}

interface Feedback {
  id: number;
  page_url: string;
  festival_name: string | null;
  rating: number;
  comment: string | null;
  timestamp: string;
  user_agent: string | null;
}

interface QnAStatistics {
  summary: {
    total_questions: number;
    total_answers: number;
    unanswered_questions: number;
    answer_rate: number;
  };
  questions_by_festival: Array<{
    festival_name: string;
    count: number;
  }>;
  recent_questions: Array<{
    id: number;
    festival_name: string;
    title: string;
    created_at: string;
    views: number;
    author: string;
    answer_count: number;
  }>;
  recent_answers: Array<{
    id: number;
    content: string;
    created_at: string;
    is_accepted: boolean;
    author: string;
    author_role: string;
    question_title: string;
    festival_name: string;
  }>;
}

export default function AdminDashboard() {
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [featureRatings, setFeatureRatings] = useState<FeatureRating[]>([]);
  const [feedbacks, setFeedbacks] = useState<Feedback[]>([]);
  const [qnaStats, setQnaStats] = useState<QnAStatistics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);

      // Fetch analytics summary
      const analyticsRes = await fetch(`${API_BASE_URL}/api/admin/analytics`);
      const analyticsData = await analyticsRes.json();
      setAnalytics(analyticsData);

      // Fetch feature ratings
      const ratingsRes = await fetch(`${API_BASE_URL}/api/admin/feature-ratings`);
      const ratingsData = await ratingsRes.json();
      setFeatureRatings(ratingsData.feature_ratings);

      // Fetch feedbacks
      const feedbacksRes = await fetch(`${API_BASE_URL}/api/admin/feedback`);
      const feedbacksData = await feedbacksRes.json();
      setFeedbacks(feedbacksData.feedback);

      // Fetch Q&A statistics
      const qnaRes = await fetch(`${API_BASE_URL}/api/admin/qna`);
      const qnaData = await qnaRes.json();
      setQnaStats(qnaData);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">대시보드 로딩 중...</p>
        </div>
      </div>
    );
  }

  const positiveFeedbacks = feedbacks.filter((f) => f.rating === 5).length;
  const negativeFeedbacks = feedbacks.filter((f) => f.rating === 1).length;
  const satisfactionRate =
    feedbacks.length > 0 ? ((positiveFeedbacks / feedbacks.length) * 100).toFixed(1) : '0';

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-8"
      >
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">관리자 대시보드</h1>
            <p className="text-gray-600 mt-2">사용자 피드백 및 행동 분석</p>
          </div>
          <button
            onClick={fetchDashboardData}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            새로고침
          </button>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-blue-500"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">총 피드백</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                  {analytics?.summary.total_feedback || 0}
                </p>
              </div>
              <Activity className="text-blue-500" size={40} />
            </div>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-green-500"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">기능 평가</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                  {analytics?.summary.total_ratings || 0}
                </p>
              </div>
              <ThumbsUp className="text-green-500" size={40} />
            </div>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-purple-500"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">총 이벤트</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                  {analytics?.summary.total_events || 0}
                </p>
              </div>
              <TrendingUp className="text-purple-500" size={40} />
            </div>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-yellow-500"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">만족도</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{satisfactionRate}%</p>
              </div>
              <div className="text-yellow-500 text-2xl">😊</div>
            </div>
          </motion.div>
        </div>

        {/* Q&A Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-indigo-500"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">총 질문</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                  {qnaStats?.summary.total_questions || 0}
                </p>
              </div>
              <HelpCircle className="text-indigo-500" size={40} />
            </div>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-teal-500"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">총 답변</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                  {qnaStats?.summary.total_answers || 0}
                </p>
              </div>
              <MessageCircle className="text-teal-500" size={40} />
            </div>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-orange-500"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">미답변 질문</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                  {qnaStats?.summary.unanswered_questions || 0}
                </p>
              </div>
              <HelpCircle className="text-orange-500" size={40} />
            </div>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-cyan-500"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">답변율</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                  {qnaStats?.summary.answer_rate || 0}%
                </p>
              </div>
              <CheckCircle className="text-cyan-500" size={40} />
            </div>
          </motion.div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Feature Ratings Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">기능별 평균 평점</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureRatings}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="feature_name"
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  interval={0}
                />
                <YAxis domain={[0, 5]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="avg_rating" fill="#3b82f6" name="평균 평점" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Popular Festivals Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">인기 축제 Top 10</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={analytics?.popular_festivals || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="festival_name"
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  interval={0}
                />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="views" fill="#10b981" name="조회수" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Q&A Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Questions by Festival Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">축제별 Q&A 현황</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={qnaStats?.questions_by_festival || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="festival_name"
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  interval={0}
                />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" fill="#6366f1" name="질문 수" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Recent Questions Table */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">최근 질문</h2>
            <div className="overflow-x-auto max-h-[300px] overflow-y-auto">
              <table className="w-full text-sm text-left text-gray-500">
                <thead className="text-xs text-gray-700 uppercase bg-gray-50 sticky top-0">
                  <tr>
                    <th scope="col" className="px-4 py-3">축제</th>
                    <th scope="col" className="px-4 py-3">제목</th>
                    <th scope="col" className="px-4 py-3">작성자</th>
                    <th scope="col" className="px-4 py-3">답변</th>
                  </tr>
                </thead>
                <tbody>
                  {qnaStats?.recent_questions.map((question) => (
                    <tr key={question.id} className="bg-white border-b hover:bg-gray-50">
                      <td className="px-4 py-3 text-xs">{question.festival_name}</td>
                      <td className="px-4 py-3 font-medium text-gray-900 max-w-[150px] truncate">
                        {question.title}
                      </td>
                      <td className="px-4 py-3">{question.author}</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                          question.answer_count > 0
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {question.answer_count}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {(!qnaStats?.recent_questions || qnaStats.recent_questions.length === 0) && (
                    <tr>
                      <td colSpan={4} className="px-4 py-8 text-center text-gray-500">
                        아직 등록된 질문이 없습니다.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Recent Answers Table */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">최근 답변</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left text-gray-500">
              <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                <tr>
                  <th scope="col" className="px-4 py-3">축제</th>
                  <th scope="col" className="px-4 py-3">질문</th>
                  <th scope="col" className="px-4 py-3">답변 내용</th>
                  <th scope="col" className="px-4 py-3">작성자</th>
                  <th scope="col" className="px-4 py-3">상태</th>
                </tr>
              </thead>
              <tbody>
                {qnaStats?.recent_answers.map((answer) => (
                  <tr key={answer.id} className="bg-white border-b hover:bg-gray-50">
                    <td className="px-4 py-3 text-xs">{answer.festival_name}</td>
                    <td className="px-4 py-3 font-medium text-gray-900 max-w-[150px] truncate">
                      {answer.question_title}
                    </td>
                    <td className="px-4 py-3 max-w-[200px] truncate">{answer.content}</td>
                    <td className="px-4 py-3">
                      <span className={`${answer.author_role === 'admin' ? 'text-purple-600 font-semibold' : ''}`}>
                        {answer.author}
                        {answer.author_role === 'admin' && ' (관리자)'}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      {answer.is_accepted && (
                        <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs font-semibold">
                          채택됨
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
                {(!qnaStats?.recent_answers || qnaStats.recent_answers.length === 0) && (
                  <tr>
                    <td colSpan={5} className="px-4 py-8 text-center text-gray-500">
                      아직 등록된 답변이 없습니다.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Top Events Table */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">사용자 행동 Top 20</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left text-gray-500">
              <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3">
                    카테고리
                  </th>
                  <th scope="col" className="px-6 py-3">
                    액션
                  </th>
                  <th scope="col" className="px-6 py-3">
                    횟수
                  </th>
                </tr>
              </thead>
              <tbody>
                {analytics?.top_events.map((event, index) => (
                  <tr key={index} className="bg-white border-b hover:bg-gray-50">
                    <td className="px-6 py-4 font-medium text-gray-900">{event.category}</td>
                    <td className="px-6 py-4">{event.action}</td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-semibold">
                        {event.count}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Recent Feedback */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">최근 피드백 (최대 10개)</h2>
          <div className="space-y-4">
            {feedbacks.slice(0, 10).map((feedback) => (
              <div
                key={feedback.id}
                className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      {feedback.rating === 5 ? (
                        <ThumbsUp className="text-green-600" size={20} />
                      ) : (
                        <ThumbsDown className="text-red-600" size={20} />
                      )}
                      <span className="font-medium text-gray-900">
                        {feedback.festival_name || feedback.page_url}
                      </span>
                    </div>
                    {feedback.comment && (
                      <p className="text-gray-700 text-sm mb-2">{feedback.comment}</p>
                    )}
                    <p className="text-xs text-gray-500">
                      {new Date(feedback.timestamp).toLocaleString('ko-KR')}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Feedback/Rating Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">피드백 분포</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={[
                    { name: '도움됐어요', value: positiveFeedbacks },
                    { name: '아쉬워요', value: negativeFeedbacks },
                  ]}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  <Cell fill="#10b981" />
                  <Cell fill="#ef4444" />
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">기능별 평가 수</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={featureRatings}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ feature_name, percent }) =>
                    `${feature_name} ${(percent * 100).toFixed(0)}%`
                  }
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {featureRatings.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
