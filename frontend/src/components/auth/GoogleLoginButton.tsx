import React, { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/store/useAuthStore';
import axios from 'axios';

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || 'your-google-client-id.apps.googleusercontent.com';
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface GoogleLoginButtonProps {
  onError?: (error: string) => void;
}

export const GoogleLoginButton: React.FC<GoogleLoginButtonProps> = ({ onError }) => {
  const navigate = useNavigate();
  const login = useAuthStore((state) => state.login);
  const buttonRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initialize Google Sign-In
    if (window.google && buttonRef.current) {
      window.google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: handleGoogleLogin,
      });

      window.google.accounts.id.renderButton(
        buttonRef.current,
        {
          theme: 'outline',
          size: 'large',
          width: buttonRef.current.offsetWidth,
          text: 'continue_with',
          locale: 'ko',
        }
      );
    }
  }, []);

  const handleGoogleLogin = async (response: any) => {
    try {
      // Send Google credential to backend
      const result = await axios.post(`${API_URL}/api/auth/google`, {
        credential: response.credential,
      });

      // Store tokens and user info
      login(result.data.access_token, result.data.refresh_token, result.data.user);

      // Navigate to home page
      navigate('/');
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Google 로그인에 실패했습니다.';
      if (onError) {
        onError(errorMessage);
      }
      console.error('Google login error:', err);
    }
  };

  return (
    <div>
      <div ref={buttonRef} style={{ width: '100%' }}></div>
    </div>
  );
};

// Extend Window interface for Google Sign-In
declare global {
  interface Window {
    google: any;
  }
}
