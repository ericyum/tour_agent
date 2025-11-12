import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { FaSearch, FaStar, FaChartLine, FaRoute, FaRobot, FaImage } from 'react-icons/fa'

export default function HomePage() {
  const features = [
    {
      icon: FaSearch,
      title: '스마트 검색',
      description: '지역, 카테고리, 날짜별로 전국의 축제를 손쉽게 검색하세요',
      color: 'from-blue-500 to-cyan-500',
    },
    {
      icon: FaStar,
      title: 'AI 랭킹',
      description: '트렌드와 감성 분석을 결합한 AI 추천 순위를 확인하세요',
      color: 'from-yellow-500 to-orange-500',
    },
    {
      icon: FaChartLine,
      title: '감성 분석',
      description: '실시간 블로그 리뷰 분석으로 축제의 진짜 분위기를 파악하세요',
      color: 'from-purple-500 to-pink-500',
    },
    {
      icon: FaRoute,
      title: '여행 코스',
      description: 'AI가 최적의 동선과 일정을 제안하는 맞춤형 여행 코스',
      color: 'from-green-500 to-teal-500',
    },
    {
      icon: FaImage,
      title: 'AI 렌더링',
      description: 'AI가 생성한 축제의 야경과 계절별 분위기 이미지',
      color: 'from-red-500 to-rose-500',
    },
    {
      icon: FaRobot,
      title: 'AI 가이드',
      description: '축제별 맞춤 에티켓과 주의사항을 AI가 알려드립니다',
      color: 'from-indigo-500 to-purple-500',
    },
  ]

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100,
      },
    },
  }

  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="text-center py-12 space-y-8"
      >
        <motion.div
          animate={{ y: [0, -10, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          className="text-8xl mb-4"
        >
          🎪
        </motion.div>

        <h1 className="text-5xl md:text-7xl font-bold mb-6">
          <span className="gradient-text">FestMoment</span>에
          <br />
          오신 것을 환영합니다!
        </h1>

        <p className="text-xl md:text-2xl text-slate-600 max-w-3xl mx-auto mb-8 leading-relaxed">
          AI가 추천하는 전국 축제 정보와 감성 분석,
          <br />
          당신의 완벽한 축제 경험을 위한 스마트 가이드
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <Link to="/search">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="btn-primary text-lg px-8 py-4 flex items-center space-x-2"
            >
              <FaSearch />
              <span>축제 찾아보기</span>
            </motion.button>
          </Link>
          <motion.a
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            href="#features"
            className="btn-secondary text-lg px-8 py-4"
          >
            기능 알아보기
          </motion.a>
        </div>
      </motion.section>

      {/* Features Section */}
      <motion.section
        id="features"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="py-12"
      >
        <motion.h2
          variants={itemVariants}
          className="text-4xl md:text-5xl font-bold text-center mb-4"
        >
          <span className="gradient-text">주요 기능</span>
        </motion.h2>
        <motion.p
          variants={itemVariants}
          className="text-center text-slate-600 mb-12 text-lg"
        >
          AI 기술로 축제의 순간을 더 깊이 느껴보세요
        </motion.p>

        <motion.div
          variants={containerVariants}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
        >
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ y: -8 }}
                className="card card-hover p-8"
              >
                <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-6 shadow-lg`}>
                  <Icon className="text-3xl text-white" />
                </div>
                <h3 className="text-2xl font-bold mb-3">{feature.title}</h3>
                <p className="text-slate-600 leading-relaxed">{feature.description}</p>
              </motion.div>
            )
          })}
        </motion.div>
      </motion.section>

      {/* CTA Section */}
      <motion.section
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="card p-12 text-center bg-gradient-to-r from-primary-600 to-accent-500 text-white"
      >
        <h2 className="text-4xl font-bold mb-4">지금 바로 시작하세요!</h2>
        <p className="text-xl mb-8 opacity-90">
          전국의 다양한 축제를 AI와 함께 탐험해보세요
        </p>
        <Link to="/search">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="bg-white text-primary-600 px-10 py-4 rounded-xl font-bold text-lg shadow-2xl hover:shadow-3xl transition-all"
          >
            축제 검색하기 →
          </motion.button>
        </Link>
      </motion.section>
    </div>
  )
}
