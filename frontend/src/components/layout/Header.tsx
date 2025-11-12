import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { FaHome, FaSearch, FaRoute } from 'react-icons/fa'
import { useCourseStore } from '@/store/useCourseStore'

export default function Header() {
  const location = useLocation()
  const courseItems = useCourseStore((state) => state.courseItems)

  const navItems = [
    { path: '/', label: '홈', icon: FaHome },
    { path: '/search', label: '축제 검색', icon: FaSearch },
    { path: '/my-course', label: '내 코스', icon: FaRoute, badge: courseItems.length },
  ]

  const isActive = (path: string) => location.pathname === path

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
              <h1 className="text-2xl font-bold gradient-text">
                FestMoment
              </h1>
              <p className="text-xs text-slate-600">AI 축제 가이드</p>
            </div>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex items-center space-x-2">
            {navItems.map((item) => {
              const Icon = item.icon
              return (
                <Link key={item.path} to={item.path}>
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className={`
                      relative px-6 py-3 rounded-xl font-medium transition-all duration-300
                      ${isActive(item.path)
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
              )
            })}
          </nav>

          {/* Mobile Menu Button */}
          <button className="md:hidden p-2 rounded-lg hover:bg-white/70 transition-colors">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>
    </header>
  )
}
