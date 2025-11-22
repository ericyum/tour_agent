// src/components/common/ProgressIndicator.tsx
/**
 * 비동기 작업 진행률 표시 컴포넌트
 *
 * 장시간 실행되는 분석 작업의 진행 상태를 시각적으로 표시합니다.
 */

import React from 'react';
import { motion } from 'framer-motion';
import { Loader2, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { TaskProgress } from '@/hooks/useAsyncTask';

interface ProgressIndicatorProps {
  progress: TaskProgress | null;
  title?: string;
  showPercentage?: boolean;
  onCancel?: () => void;
}

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  progress,
  title = '분석 중...',
  showPercentage = true,
  onCancel,
}) => {
  if (!progress) return null;

  const { status, message } = progress;
  const percentage = progress.progress || 0;

  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'failed':
        return 'bg-red-500';
      case 'running':
        return 'bg-blue-500';
      default:
        return 'bg-gray-400';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-6 h-6 text-green-500" />;
      case 'failed':
        return <XCircle className="w-6 h-6 text-red-500" />;
      case 'running':
        return <Loader2 className="w-6 h-6 text-blue-500 animate-spin" />;
      default:
        return <AlertCircle className="w-6 h-6 text-gray-400" />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'completed':
        return '완료';
      case 'failed':
        return '실패';
      case 'running':
        return '진행 중';
      case 'pending':
        return '대기 중';
      default:
        return status;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-lg p-6 border border-gray-200"
    >
      {/* 헤더 */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          {getStatusIcon()}
          <div>
            <h3 className="font-semibold text-gray-900">{title}</h3>
            <p className="text-sm text-gray-500">{getStatusText()}</p>
          </div>
        </div>
        {showPercentage && status !== 'completed' && status !== 'failed' && (
          <span className="text-2xl font-bold text-blue-600">{percentage}%</span>
        )}
      </div>

      {/* 진행률 바 */}
      {status !== 'completed' && status !== 'failed' && (
        <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden mb-4">
          <motion.div
            className={`absolute left-0 top-0 h-full ${getStatusColor()} rounded-full`}
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
          />
          {/* 애니메이션 효과 */}
          {status === 'running' && (
            <motion.div
              className="absolute left-0 top-0 h-full w-full bg-gradient-to-r from-transparent via-white/30 to-transparent"
              animate={{ x: ['-100%', '100%'] }}
              transition={{ repeat: Infinity, duration: 1.5, ease: 'linear' }}
            />
          )}
        </div>
      )}

      {/* 메시지 */}
      <p
        className={`text-sm ${
          status === 'failed' ? 'text-red-600' : 'text-gray-600'
        }`}
      >
        {message || '작업을 처리하고 있습니다...'}
      </p>

      {/* 취소 버튼 (필요시) */}
      {onCancel && status === 'running' && (
        <button
          onClick={onCancel}
          className="mt-4 text-sm text-gray-500 hover:text-gray-700 underline"
        >
          취소하고 페이지 떠나기
        </button>
      )}

      {/* 완료 상태 */}
      {status === 'completed' && (
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg"
        >
          <p className="text-green-800 text-sm font-medium">
            분석이 완료되었습니다!
          </p>
        </motion.div>
      )}

      {/* 실패 상태 */}
      {status === 'failed' && (
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg"
        >
          <p className="text-red-800 text-sm font-medium">
            작업 중 오류가 발생했습니다. 다시 시도해주세요.
          </p>
        </motion.div>
      )}
    </motion.div>
  );
};

/**
 * 로딩 오버레이 (전체 화면)
 */
interface LoadingOverlayProps {
  progress: TaskProgress | null;
  title?: string;
  isOpen: boolean;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  progress,
  title,
  isOpen,
}) => {
  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
    >
      <div className="w-full max-w-md mx-4">
        <ProgressIndicator progress={progress} title={title} />
        <p className="text-center text-white/80 text-sm mt-4">
          브라우저를 닫아도 분석은 계속 진행됩니다.
          <br />
          다시 돌아오면 진행 상태를 확인할 수 있습니다.
        </p>
      </div>
    </motion.div>
  );
};

/**
 * 인라인 진행률 표시 (컴팩트)
 */
interface InlineProgressProps {
  progress: number;
  status: string;
  message?: string;
}

export const InlineProgress: React.FC<InlineProgressProps> = ({
  progress,
  status,
  message,
}) => {
  return (
    <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg">
      <Loader2 className="w-5 h-5 text-blue-500 animate-spin flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm font-medium text-blue-900 truncate">
            {message || '분석 중...'}
          </span>
          <span className="text-sm font-bold text-blue-600">{progress}%</span>
        </div>
        <div className="h-1.5 bg-blue-200 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-blue-500 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>
    </div>
  );
};

export default ProgressIndicator;
