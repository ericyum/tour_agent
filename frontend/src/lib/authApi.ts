import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { useAuthStore } from '@/store/useAuthStore';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with auth interceptors
export const authApi: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Flag to prevent multiple refresh requests
let isRefreshing = false;
let failedQueue: Array<{
  resolve: (value?: unknown) => void;
  reject: (reason?: unknown) => void;
}> = [];

const processQueue = (error: Error | null, token: string | null = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });

  failedQueue = [];
};

// Request interceptor - Add access token to headers
authApi.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const { accessToken } = useAuthStore.getState();

    if (accessToken && config.headers) {
      config.headers.Authorization = `Bearer ${accessToken}`;
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - Handle token refresh
authApi.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & {
      _retry?: boolean;
    };

    // If error is 401 and we haven't retried yet
    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        // If already refreshing, wait for it to complete
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        })
          .then((token) => {
            if (originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${token}`;
            }
            return authApi(originalRequest);
          })
          .catch((err) => {
            return Promise.reject(err);
          });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      const { refreshToken } = useAuthStore.getState();

      if (!refreshToken) {
        // No refresh token - logout
        useAuthStore.getState().logout();
        window.location.href = '/login';
        return Promise.reject(error);
      }

      try {
        // Request new access token
        const response = await axios.post(`${API_BASE_URL}/api/auth/refresh`, {
          refresh_token: refreshToken,
        });

        const { access_token } = response.data;

        // Update store with new access token
        useAuthStore.getState().setAccessToken(access_token);

        // Update authorization header
        if (originalRequest.headers) {
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
        }

        processQueue(null, access_token);

        // Retry original request
        return authApi(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError as Error, null);
        // Refresh token expired - logout
        useAuthStore.getState().logout();
        window.location.href = '/login';
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);

// Auth API functions
export const authService = {
  register: async (username: string, email: string, password: string, full_name?: string) => {
    const response = await axios.post(`${API_BASE_URL}/api/auth/register`, {
      username,
      email,
      password,
      full_name,
    });
    return response.data;
  },

  login: async (username: string, password: string) => {
    const response = await axios.post(`${API_BASE_URL}/api/auth/login`, {
      username,
      password,
    });
    return response.data;
  },

  logout: async (refreshToken: string) => {
    const response = await axios.post(`${API_BASE_URL}/api/auth/logout`, {
      refresh_token: refreshToken,
    });
    return response.data;
  },

  getCurrentUser: async () => {
    const response = await authApi.get('/api/auth/me');
    return response.data;
  },

  updateProfile: async (full_name?: string, email?: string) => {
    const response = await authApi.put('/api/auth/profile', {
      full_name,
      email,
    });
    return response.data;
  },
};

// Q&A API functions
export const qnaService = {
  getFestivalQuestions: async (festivalName: string) => {
    const response = await authApi.get(`/api/qna/festival/${festivalName}`);
    return response.data;
  },

  getQuestion: async (questionId: number) => {
    const response = await authApi.get(`/api/qna/question/${questionId}`);
    return response.data;
  },

  createQuestion: async (festivalName: string, title: string, content: string) => {
    const response = await authApi.post('/api/qna/question', {
      festival_name: festivalName,
      title,
      content,
    });
    return response.data;
  },

  updateQuestion: async (questionId: number, title: string, content: string) => {
    const response = await authApi.put(`/api/qna/question/${questionId}`, {
      title,
      content,
    });
    return response.data;
  },

  deleteQuestion: async (questionId: number) => {
    const response = await authApi.delete(`/api/qna/question/${questionId}`);
    return response.data;
  },

  createAnswer: async (questionId: number, content: string) => {
    const response = await authApi.post(`/api/qna/question/${questionId}/answer`, {
      content,
    });
    return response.data;
  },

  updateAnswer: async (answerId: number, content: string) => {
    const response = await authApi.put(`/api/qna/answer/${answerId}`, {
      content,
    });
    return response.data;
  },

  deleteAnswer: async (answerId: number) => {
    const response = await authApi.delete(`/api/qna/answer/${answerId}`);
    return response.data;
  },

  acceptAnswer: async (answerId: number) => {
    const response = await authApi.post(`/api/qna/answer/${answerId}/accept`);
    return response.data;
  },

  getMyQuestions: async () => {
    const response = await authApi.get('/api/user/questions');
    return response.data;
  },

  getMyAnswers: async () => {
    const response = await authApi.get('/api/user/answers');
    return response.data;
  },
};
