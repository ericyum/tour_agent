import { useEffect, useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import axios from 'axios'
import Layout from './components/layout/Layout'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
import HomePage from './pages/HomePage'
import SearchPage from './pages/SearchPage'
import FestivalDetailPage from './pages/FestivalDetailPage'
import CourseDetailPage from './pages/CourseDetailPage'
import FacilityDetailPage from './pages/FacilityDetailPage'
import MyCoursePage from './pages/MyCoursePage'
import AdminDashboard from './pages/AdminDashboard'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import MyPage from './pages/MyPage'
import QnAPage from './pages/QnAPage'
import QnANewPage from './pages/QnANewPage'
import QnADetailPage from './pages/QnADetailPage'
import { useAuthStore } from './store/useAuthStore'

// Protected Route Component
function ProtectedRoute({ children, adminOnly = false }: { children: React.ReactNode, adminOnly?: boolean }) {
  const { isAuthenticated, isAdmin } = useAuthStore();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (adminOnly && !isAdmin) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
}

function App() {
  const { refreshToken, accessToken, setAccessToken, logout, isAuthenticated } = useAuthStore()
  const [isInitializing, setIsInitializing] = useState(true)

  // Auto-refresh access token on app load if user has refresh token but no access token
  useEffect(() => {
    const refreshAccessToken = async () => {
      if (isAuthenticated && refreshToken && !accessToken) {
        try {
          const response = await axios.post(`${API_BASE_URL}/api/auth/refresh`, {
            refresh_token: refreshToken,
          })
          setAccessToken(response.data.access_token)
        } catch (error) {
          // Refresh token is invalid/expired - logout user
          console.error('Failed to refresh token:', error)
          logout()
        }
      }
      setIsInitializing(false)
    }

    refreshAccessToken()
  }, []) // Run only on mount

  // Show loading while initializing auth
  if (isInitializing && isAuthenticated && !accessToken) {
    return (
      <Layout>
        <div className="flex items-center justify-center min-h-[50vh]">
          <div className="text-gray-500">로딩 중...</div>
        </div>
      </Layout>
    )
  }

  return (
    <Layout>
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/festival/:festivalName" element={<FestivalDetailPage />} />
          <Route path="/course/:courseTitle" element={<CourseDetailPage />} />
          <Route path="/facility/:facilityTitle" element={<FacilityDetailPage />} />
          <Route path="/my-course" element={<MyCoursePage />} />

          {/* Auth Routes */}
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          {/* Protected Routes */}
          <Route path="/mypage" element={<ProtectedRoute><MyPage /></ProtectedRoute>} />
          <Route path="/qna" element={<QnAPage />} />
          <Route path="/qna/new" element={<ProtectedRoute><QnANewPage /></ProtectedRoute>} />
          <Route path="/qna/:questionId" element={<QnADetailPage />} />
          <Route path="/admin" element={<ProtectedRoute adminOnly><AdminDashboard /></ProtectedRoute>} />
        </Routes>
      </AnimatePresence>
    </Layout>
  )
}

export default App
