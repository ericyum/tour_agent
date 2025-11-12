import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { FaCalendarAlt, FaPlus } from 'react-icons/fa'
import { formatDateRange, getFestivalStatus } from '@/lib/utils'
import { useCourseStore } from '@/store/useCourseStore'

// Make the component more generic
interface CardItem {
  title: string
  image: string
  type: 'festival' | 'course' | 'facility'
  start_date?: string
  end_date?: string
  [key: string]: any // Allow other properties
}

interface CardProps {
  item: CardItem
  index?: number
}

export default function FestivalCard({ item, index = 0 }: CardProps) {
  // Add a guard clause for item
  if (!item) {
    console.error("FestivalCard received undefined item prop");
    return null; // Or render a placeholder/error message
  }

  const addItem = useCourseStore((state) => state.addItem)
  const courseItems = useCourseStore((state) => state.courseItems)

  const isInCourse = courseItems.some((ci) => ci.title === item.title)
  
  // Status only applies to festivals
  const status = item.type === 'festival' ? getFestivalStatus(item.start_date, item.end_date) : null

  const handleAddToCourse = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    // Use the item's type when adding to the course
    addItem({
      ...item,
      type: item.type,
    })
  }

  // Dynamic link based on item type
  const linkTo = `/${item.type}/${encodeURIComponent(item.title)}`

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      whileHover={{ y: -8 }}
      className="card card-hover group relative"
    >
      <Link to={linkTo} className="block">
        {/* Image */}
        <div className="relative h-48 overflow-hidden">
          <img
            src={item.image}
            alt={item.title}
            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
            onError={(e) => {
              e.currentTarget.src = 'https://placehold.co/400x300?text=No+Image'
            }}
          />

          {/* Status Badge (only for festivals) */}
          {status && (
            <div className={`absolute top-4 left-4 px-3 py-1 rounded-full text-sm font-medium ${status.color}`}>
              {status.status}
            </div>
          )}

          {/* Add to Course Button */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={handleAddToCourse}
            disabled={isInCourse}
            className={`
              absolute top-4 right-4 p-3 rounded-full shadow-lg transition-all
              ${isInCourse
                ? 'bg-green-500 text-white cursor-not-allowed'
                : 'bg-white text-primary-600 hover:bg-primary-600 hover:text-white'
              }
            `}
            title={isInCourse ? '이미 추가됨' : '내 코스에 추가'}
          >
            <FaPlus className={isInCourse ? 'rotate-45' : ''} />
          </motion.button>

          {/* Gradient Overlay */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        </div>

        {/* Content */}
        <div className="p-6 space-y-3">
          <h3 className="text-xl font-bold text-slate-900 line-clamp-2 group-hover:text-primary-600 transition-colors">
            {item.title}
          </h3>

          <div className="space-y-2 text-sm text-slate-600">
            {/* Date range only for festivals */}
            {item.type === 'festival' && (
              <div className="flex items-center space-x-2">
                <FaCalendarAlt className="text-primary-600 flex-shrink-0" />
                <span className="line-clamp-1">
                  {formatDateRange(item.start_date, item.end_date)}
                </span>
              </div>
            )}
          </div>

          {/* View Details Link */}
          <div className="pt-4 border-t border-slate-100">
            <span className="text-primary-600 font-medium group-hover:underline flex items-center space-x-1">
              <span>자세히 보기</span>
              <span className="group-hover:translate-x-1 transition-transform">→</span>
            </span>
          </div>
        </div>
      </Link>
    </motion.div>
  )
}
