import { Routes, Route, Navigate } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Layout from './components/layout/Layout'
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
          <Route path="/admin" element={<ProtectedRoute adminOnly><AdminDashboard /></ProtectedRoute>} />
        </Routes>
      </AnimatePresence>
    </Layout>
  )
}

export default App
