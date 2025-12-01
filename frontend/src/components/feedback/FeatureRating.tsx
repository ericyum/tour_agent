import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Star } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface FeatureRatingProps {
  festivalName: string;
  featureName: string;
  featureLabel: string;
  description?: string;
}

export const FeatureRating: React.FC<FeatureRatingProps> = ({
  festivalName,
  featureName,
  featureLabel,
  description,
}) => {
  const [rating, setRating] = useState<number>(0);
  const [hoveredRating, setHoveredRating] = useState<number>(0);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const getSessionId = () => {
    let sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('session_id', sessionId);
    }
    return sessionId;
  };

  const handleRatingClick = async (selectedRating: number) => {
    setRating(selectedRating);

    try {
      const response = await fetch(`${API_BASE_URL}/api/feature-rating`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          festival_name: festivalName,
          feature_name: featureName,
          rating: selectedRating,
          session_id: getSessionId(),
        }),
      });

      if (response.ok) {
        setIsSubmitted(true);
      }
    } catch (error) {
      console.error('Failed to submit rating:', error);
    }
  };

  return (
    <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h4 className="text-sm font-medium text-gray-900 mb-1">
            {featureLabel}
          </h4>
          {description && (
            <p className="text-xs text-gray-600 mb-2">{description}</p>
          )}
        </div>

        <div className="flex items-center gap-1 ml-4">
          {[1, 2, 3, 4, 5].map((star) => (
            <motion.button
              key={star}
              whileHover={{ scale: 1.2 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => !isSubmitted && handleRatingClick(star)}
              onMouseEnter={() => !isSubmitted && setHoveredRating(star)}
              onMouseLeave={() => !isSubmitted && setHoveredRating(0)}
              disabled={isSubmitted}
              className={`transition-colors ${isSubmitted ? 'cursor-default' : 'cursor-pointer'}`}
            >
              <Star
                size={20}
                className={`
                  ${
                    (hoveredRating || rating) >= star
                      ? 'fill-yellow-400 text-yellow-400'
                      : 'fill-none text-gray-300'
                  }
                  transition-colors
                `}
              />
            </motion.button>
          ))}
        </div>
      </div>

      {isSubmitted && (
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-xs text-green-600 mt-2 font-medium"
        >
          평가해주셔서 감사합니다!
        </motion.p>
      )}
    </div>
  );
};
