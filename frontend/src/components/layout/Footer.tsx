import { FaGithub, FaHeart } from 'react-icons/fa'

export default function Footer() {
  return (
    <footer className="glass border-t border-white/20 mt-auto">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          {/* Brand */}
          <div className="flex items-center space-x-2">
            <span className="text-2xl">🎪</span>
            <div>
              <p className="font-bold text-lg gradient-text">FestMoment</p>
              <p className="text-sm text-slate-600">AI-Powered Festival Guide</p>
            </div>
          </div>

          {/* Info */}
          <div className="text-center md:text-left">
            <p className="text-sm text-slate-600 flex items-center space-x-1">
              <span>Made with</span>
              <FaHeart className="text-red-500 animate-pulse" />
              <span>by Team FestMoment</span>
            </p>
            <p className="text-xs text-slate-500 mt-1">
              염정운, 최가윤 | 2025
            </p>
          </div>

          {/* Links */}
          <div className="flex items-center space-x-4">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-3 rounded-full bg-white hover:bg-primary-50 transition-colors shadow-md"
            >
              <FaGithub className="text-xl text-slate-700" />
            </a>
          </div>
        </div>

        {/* Copyright */}
        <div className="mt-6 pt-6 border-t border-slate-200 text-center">
          <p className="text-xs text-slate-500">
            © 2025 FestMoment. All rights reserved. | Powered by Google Gemini & LangGraph
          </p>
        </div>
      </div>
    </footer>
  )
}
