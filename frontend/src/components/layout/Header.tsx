import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { FaHome, FaSearch, FaRoute, FaUser, FaSignOutAlt, FaCog, FaQuestionCircle } from 'react-icons/fa';
import { useCourseStore } from '@/store/useCourseStore';
import { useAuthStore } from '@/store/useAuthStore';
import { authService } from '@/lib/authApi';
import { useState } from 'react';
import { LogIn, UserPlus, Shield } from 'lucide-react';

export default function Header() {
  const location = useLocation();
  const navigate = useNavigate();
  const courseItems = useCourseStore((state) => state.courseItems);
  const { user, isAuthenticated, isAdmin, logout, refreshToken } = useAuthStore();
  const [showUserMenu, setShowUserMenu] = useState(false);

  const navItems = [
    { path: '/', label: '홈', icon: FaHome },
    { path: '/search', label: '축제 검색', icon: FaSearch },
    { path: '/my-course', label: '내 코스', icon: FaRoute, badge: courseItems.length },
  ];

  const isActive = (path: string) => location.pathname === path;

  const handleLogout = async () => {
    try {
      if (refreshToken) {
        await authService.logout(refreshToken);
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      logout();
      setShowUserMenu(false);
      navigate('/');
    }
  };

  return (
    <header className="sticky top-0 z-50 glass border-b border-white/20 shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3 group">
            <motion.div
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.6 }}
              className="text-4xl"
            >
              🎪
            </motion.div>
            <div>
              <h1 className="text-2xl font-bold gradient-text">FestMoment</h1>
              <p className="text-xs text-slate-600">AI 축제 가이드</p>
            </div>
          </Link>

          {/* Navigation */}
          <div className="flex items-center space-x-4">
            <nav className="hidden md:flex items-center space-x-2">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <Link key={item.path} to={item.path}>
                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={`
                        relative px-6 py-3 rounded-xl font-medium transition-all duration-300
                        ${
                          isActive(item.path)
                            ? 'bg-gradient-to-r from-primary-600 to-primary-700 text-white shadow-lg'
                            : 'text-slate-700 hover:bg-white/70'
                        }
                      `}
                    >
                      <div className="flex items-center space-x-2">
                        <Icon className="text-lg" />
                        <span>{item.label}</span>
                        {item.badge !== undefined && item.badge > 0 && (
                          <span className="ml-1 px-2 py-0.5 text-xs rounded-full bg-accent-500 text-white font-bold">
                            {item.badge}
                          </span>
                        )}
                      </div>
                    </motion.div>
                  </Link>
                );
              })}
            </nav>

            {/* Auth Section */}
            {isAuthenticated ? (
              <div className="relative">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg hover:from-blue-700 hover:to-indigo-700"
                >
                  <FaUser />
                  <span className="font-medium">{user?.username}</span>
                  {isAdmin && <Shield size={16} />}
                </motion.button>

                <AnimatePresence>
                  {showUserMenu && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="absolute right-0 mt-2 w-56 bg-white rounded-lg shadow-xl border border-gray-200 overflow-hidden"
                    >
                      <div className="px-4 py-3 bg-gradient-to-r from-blue-50 to-indigo-50 border-b">
                        <p className="text-sm font-semibold text-gray-900">{user?.full_name || user?.username}</p>
                        <p className="text-xs text-gray-600">{user?.email}</p>
                        {isAdmin && (
                          <span className="inline-flex items-center mt-1 px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800">
                            <Shield size={12} className="mr-1" />
                            관리자
                          </span>
                        )}
                      </div>

                      <div className="py-1">
                        <Link
                          to="/mypage"
                          onClick={() => setShowUserMenu(false)}
                          className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                        >
                          <FaUser className="mr-3" />
                          마이페이지
                        </Link>

                        <Link
                          to="/qna"
                          onClick={() => setShowUserMenu(false)}
                          className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                        >
                          <FaQuestionCircle className="mr-3" />
                          Q&A 게시판
                        </Link>

                        {isAdmin && (
                          <Link
                            to="/admin"
                            onClick={() => setShowUserMenu(false)}
                            className="flex items-center px-4 py-2 text-sm text-purple-700 hover:bg-purple-50"
                          >
                            <FaCog className="mr-3" />
                            관리자 대시보드
                          </Link>
                        )}
                      </div>

                      <div className="border-t">
                        <button
                          onClick={handleLogout}
                          className="flex items-center w-full px-4 py-2 text-sm text-red-700 hover:bg-red-50"
                        >
                          <FaSignOutAlt className="mr-3" />
                          로그아웃
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <Link to="/login">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex items-center space-x-2 px-4 py-2 rounded-lg border-2 border-blue-600 text-blue-600 font-medium hover:bg-blue-50"
                  >
                    <LogIn size={18} />
                    <span>로그인</span>
                  </motion.button>
                </Link>

                <Link to="/register">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 text-white font-medium hover:from-purple-700 hover:to-pink-700"
                  >
                    <UserPlus size={18} />
                    <span>회원가입</span>
                  </motion.button>
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Click outside to close menu */}
      {showUserMenu && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setShowUserMenu(false)}
        />
      )}
    </header>
  );
}
