import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Search, Plus } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function QnAPage() {
  const [searchTerm, setSearchTerm] = useState('');

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
          <p className="text-gray-600">축제에 대해 궁금한 점을 물어보세요!</p>
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
        </div>

        {/* Coming Soon Message */}
        <div className="bg-white rounded-xl shadow-lg p-12 text-center border border-gray-200">
          <MessageSquare size={64} className="text-gray-300 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Q&A 게시판</h2>
          <p className="text-gray-600 mb-4">
            축제별 질문과 답변 기능이 곧 제공됩니다!
          </p>
          <p className="text-sm text-gray-500">
            현재는 축제 정보 검색, AI 분석, 여행 코스 기능을 이용하실 수 있습니다.
          </p>
          <div className="mt-8 flex justify-center gap-4">
            <Link to="/search">
              <button className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                축제 검색하기
              </button>
            </Link>
            <Link to="/mypage">
              <button className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300">
                마이페이지로
              </button>
            </Link>
          </div>
        </div>

        {/* Feature Preview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-200">
            <h3 className="font-bold text-gray-900 mb-2">질문 작성</h3>
            <p className="text-sm text-gray-600">
              축제별로 질문을 작성하고 다른 사용자의 답변을 받아보세요.
            </p>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-teal-50 p-6 rounded-xl border border-green-200">
            <h3 className="font-bold text-gray-900 mb-2">답변 채택</h3>
            <p className="text-sm text-gray-600">
              자신의 질문에 달린 답변 중 가장 도움이 된 답변을 채택할 수 있습니다.
            </p>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-pink-50 p-6 rounded-xl border border-purple-200">
            <h3 className="font-bold text-gray-900 mb-2">권한 관리</h3>
            <p className="text-sm text-gray-600">
              본인이 작성한 글만 수정 가능하고, 삭제는 본인과 관리자만 할 수 있습니다.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
