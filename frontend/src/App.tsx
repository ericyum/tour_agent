import { Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Layout from './components/layout/Layout'
import HomePage from './pages/HomePage'
import SearchPage from './pages/SearchPage'
import FestivalDetailPage from './pages/FestivalDetailPage'
import CourseDetailPage from './pages/CourseDetailPage'
import FacilityDetailPage from './pages/FacilityDetailPage'
import MyCoursePage from './pages/MyCoursePage'

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
        </Routes>
      </AnimatePresence>
    </Layout>
  )
}

export default App
