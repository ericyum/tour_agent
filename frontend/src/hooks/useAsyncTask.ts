// src/hooks/useAsyncTask.ts
/**
 * 비동기 작업 관리 훅
 *
 * 장시간 실행되는 분석 작업의 진행률을 추적하고,
 * 브라우저를 닫았다가 다시 열어도 상태를 복원합니다.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const POLL_INTERVAL = 1000; // 1초마다 진행률 조회
const STORAGE_KEY = 'festmoment_active_tasks';

export interface TaskProgress {
  task_id: string;
  progress: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message: string;
  updated_at?: string;
}

export interface TaskResult<T = any> {
  task_id: string;
  status: 'completed' | 'failed' | 'running';
  result?: T;
  error?: string;
  progress?: number;
  message?: string;
}

interface StoredTask {
  task_id: string;
  type: 'sentiment' | 'ranking' | 'wordcloud' | 'render';
  festival_name?: string;
  festivals?: string[];
  started_at: string;
}

// 로컬 스토리지에서 활성 작업 목록 관리
const getStoredTasks = (): StoredTask[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
};

const saveStoredTasks = (tasks: StoredTask[]) => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks));
};

const addStoredTask = (task: StoredTask) => {
  const tasks = getStoredTasks().filter((t) => t.task_id !== task.task_id);
  tasks.push(task);
  saveStoredTasks(tasks);
};

const removeStoredTask = (taskId: string) => {
  const tasks = getStoredTasks().filter((t) => t.task_id !== taskId);
  saveStoredTasks(tasks);
};

// 작업 진행률 조회
export const fetchTaskProgress = async (taskId: string): Promise<TaskProgress> => {
  const response = await fetch(`${API_BASE_URL}/api/async/task/${taskId}/progress`);
  if (!response.ok) {
    throw new Error('작업 진행률 조회 실패');
  }
  return response.json();
};

// 작업 결과 조회
export const fetchTaskResult = async <T = any>(taskId: string): Promise<TaskResult<T>> => {
  const response = await fetch(`${API_BASE_URL}/api/async/task/${taskId}/result`);
  if (!response.ok) {
    throw new Error('작업 결과 조회 실패');
  }
  return response.json();
};

// 비동기 작업 시작 함수들
export const startSentimentAnalysis = async (
  festivalName: string,
  numReviews: number = 10
): Promise<{ task_id: string }> => {
  const response = await fetch(`${API_BASE_URL}/api/async/sentiment/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ festival_name: festivalName, num_reviews: numReviews }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || '감성 분석 시작 실패');
  }

  const result = await response.json();

  // 로컬 스토리지에 저장
  addStoredTask({
    task_id: result.task_id,
    type: 'sentiment',
    festival_name: festivalName,
    started_at: new Date().toISOString(),
  });

  return result;
};

export const startRankingAnalysis = async (
  festivals: string[],
  numReviews: number = 5,
  topN: number = 3
): Promise<{ task_id: string }> => {
  const response = await fetch(`${API_BASE_URL}/api/async/ranking/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ festivals, num_reviews: numReviews, top_n: topN }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || '랭킹 분석 시작 실패');
  }

  const result = await response.json();

  addStoredTask({
    task_id: result.task_id,
    type: 'ranking',
    festivals,
    started_at: new Date().toISOString(),
  });

  return result;
};

export const startWordcloudGeneration = async (
  festivalName: string,
  numReviews: number = 20
): Promise<{ task_id: string }> => {
  const response = await fetch(`${API_BASE_URL}/api/async/wordcloud/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ festival_name: festivalName, num_reviews: numReviews }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || '워드클라우드 생성 시작 실패');
  }

  const result = await response.json();

  addStoredTask({
    task_id: result.task_id,
    type: 'wordcloud',
    festival_name: festivalName,
    started_at: new Date().toISOString(),
  });

  return result;
};

/**
 * 비동기 작업 관리 훅
 */
export function useAsyncTask<T = any>(taskId: string | null) {
  const [progress, setProgress] = useState<TaskProgress | null>(null);
  const [result, setResult] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  const pollProgress = useCallback(async () => {
    if (!taskId) return;

    try {
      const progressData = await fetchTaskProgress(taskId);
      setProgress(progressData);

      if (progressData.status === 'completed') {
        stopPolling();
        // 결과 조회
        const resultData = await fetchTaskResult<T>(taskId);
        if (resultData.result) {
          setResult(resultData.result);
        }
        removeStoredTask(taskId);
        setIsLoading(false);
      } else if (progressData.status === 'failed') {
        stopPolling();
        setError(progressData.message || '작업 실패');
        removeStoredTask(taskId);
        setIsLoading(false);
      }
    } catch (err) {
      console.error('Progress poll error:', err);
      // 에러가 발생해도 폴링 계속 (일시적 네트워크 문제일 수 있음)
    }
  }, [taskId, stopPolling]);

  const startPolling = useCallback(() => {
    if (!taskId || pollIntervalRef.current) return;

    setIsLoading(true);
    setError(null);

    // 즉시 한번 조회
    pollProgress();

    // 주기적 폴링 시작
    pollIntervalRef.current = setInterval(pollProgress, POLL_INTERVAL);
  }, [taskId, pollProgress]);

  // taskId가 변경되면 폴링 시작/중지
  useEffect(() => {
    if (taskId) {
      startPolling();
    } else {
      stopPolling();
      setProgress(null);
      setResult(null);
      setError(null);
      setIsLoading(false);
    }

    return () => stopPolling();
  }, [taskId, startPolling, stopPolling]);

  return {
    progress,
    result,
    error,
    isLoading,
    isCompleted: progress?.status === 'completed',
    isFailed: progress?.status === 'failed',
  };
}

/**
 * 활성 작업 복원 훅
 * 페이지 로드 시 이전에 실행 중이던 작업이 있는지 확인
 */
export function useRestoreActiveTask(
  type: 'sentiment' | 'ranking' | 'wordcloud' | 'render',
  festivalName?: string
) {
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);

  useEffect(() => {
    const tasks = getStoredTasks();
    const activeTask = tasks.find(
      (t) =>
        t.type === type &&
        (festivalName ? t.festival_name === festivalName : true)
    );

    if (activeTask) {
      // 작업이 아직 유효한지 확인 (1시간 이내)
      const startedAt = new Date(activeTask.started_at).getTime();
      const now = Date.now();
      const oneHour = 60 * 60 * 1000;

      if (now - startedAt < oneHour) {
        setActiveTaskId(activeTask.task_id);
      } else {
        // 만료된 작업 삭제
        removeStoredTask(activeTask.task_id);
      }
    }
  }, [type, festivalName]);

  const clearActiveTask = useCallback(() => {
    if (activeTaskId) {
      removeStoredTask(activeTaskId);
      setActiveTaskId(null);
    }
  }, [activeTaskId]);

  return { activeTaskId, setActiveTaskId, clearActiveTask };
}

export { getStoredTasks, removeStoredTask };
