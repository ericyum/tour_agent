import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { FaSpinner, FaMapMarkerAlt, FaInfoCircle } from 'react-icons/fa';
import { getCourseDetails } from '@/lib/api'; // Will create this API function
import { removeHtmlTags } from '@/lib/utils';
import { useCourseStore } from '@/store/useCourseStore';
import { FaPlus } from 'react-icons/fa';

const LoadingSpinner = ({ text }: { text: string }) => (
  <div className="flex flex-col items-center justify-center py-12">
    <FaSpinner className="text-4xl text-primary-600 animate-spin mb-4" />
    <p className="text-slate-600">{text}</p>
  </div>
);

const ErrorMessage = ({ text }: { text: string }) => (
  <div className="text-center py-12 text-red-600">
    <p>{text}</p>
  </div>
);

export default function CourseDetailPage() {
  const { courseTitle } = useParams<{ courseTitle: string }>();
  const addItem = useCourseStore((state) => state.addItem);
  const courseItems = useCourseStore((state) => state.courseItems);

  const { data, isLoading, error } = useQuery({
    queryKey: ['course-details', courseTitle],
    queryFn: () => getCourseDetails(courseTitle!),
    enabled: !!courseTitle,
  });

  if (isLoading) {
    return <LoadingSpinner text="코스 정보를 불러오는 중..." />;
  }

  if (error || !data) {
    return <ErrorMessage text="코스 정보를 찾을 수 없습니다" />;
  }

  const { details } = data;
  const isInCourse = courseItems.some((item) => item.title === details.title);

  const handleAddToCourse = () => {
    addItem({ ...details, type: 'course', title: details.title || '' });
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
      {/* Hero Section */}
      <div className="card overflow-hidden">
        <div className="relative h-96">
          <img src={details.firstimage || 'https://placehold.co/1200x400?text=Course'} alt={details.title} className="w-full h-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent" />
          <div className="absolute bottom-0 left-0 right-0 p-8 text-white">
            <div className="container mx-auto">
              <h1 className="text-4xl md:text-5xl font-bold mb-4">{details.title}</h1>
              <div className="flex flex-wrap items-center gap-4 text-lg">
                {details.addr1 && <div className="flex items-center space-x-2"><FaMapMarkerAlt /><span>{details.addr1}</span></div>}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-4">
        <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={handleAddToCourse} disabled={isInCourse} className={`btn-primary flex items-center space-x-2 ${isInCourse ? 'opacity-50 cursor-not-allowed' : ''}`}>
          <FaPlus />
          <span>{isInCourse ? '이미 추가됨' : '내 코스에 추가'}</span>
        </motion.button>
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          {details.overview && <div className="card p-6"><h2 className="text-2xl font-bold mb-4">코스 소개</h2><p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{removeHtmlTags(details.overview)}</p></div>}
          {details.program && <div className="card p-6"><h2 className="text-2xl font-bold mb-4">세부 프로그램</h2><p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{removeHtmlTags(details.program)}</p></div>}
          {details.sub_points && details.sub_points.length > 0 && (
            <div className="card p-6">
              <h2 className="text-2xl font-bold mb-4">코스 구성</h2>
              <div className="space-y-6">
                {details.sub_points.map((subPoint: any, index: number) => (
                  <div key={index} className="flex items-start space-x-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-600 text-white flex items-center justify-center font-bold text-lg">
                      {index + 1}
                    </div>
                    <div className="flex-grow">
                      <h3 className="text-xl font-semibold mb-2">{subPoint.subname}</h3>
                      {subPoint.subdetailoverview && <p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{removeHtmlTags(subPoint.subdetailoverview)}</p>}
                      {subPoint.subdetailimg && (
                        <img src={subPoint.subdetailimg} alt={subPoint.subname} className="mt-4 w-full h-auto rounded-lg shadow-md" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        <div className="space-y-6">
          { (details.addr1 || details.mapx || details.mapy || details.distance || details.taketime || details.theme || details.schedule || details.homepage) && (
            <div className="card p-6">
              <h3 className="text-xl font-bold mb-4">기본 정보</h3>
              <dl className="space-y-3">
              {details.addr1 && <div><dt className="text-sm text-slate-500 mb-1">주소</dt><dd className="text-slate-900">{details.addr1}</dd></div>}
              {details.mapx && details.mapy && <div><dt className="text-sm text-slate-500 mb-1">좌표</dt><dd className="text-slate-900">{details.mapy}, {details.mapx}</dd></div>}
              {details.distance && <div><dt className="text-sm text-slate-500 mb-1">총 거리</dt><dd className="text-slate-900">{details.distance}</dd></div>}
              {details.taketime && <div><dt className="text-sm text-slate-500 mb-1">소요 시간</dt><dd className="text-slate-900">{details.taketime}</dd></div>}
              {details.theme && <div><dt className="text-sm text-slate-500 mb-1">테마</dt><dd className="text-slate-900">{details.theme}</dd></div>}
              {details.schedule && <div><dt className="text-sm text-slate-500 mb-1">일정</dt><dd className="text-slate-900">{details.schedule}</dd></div>}
              {details.homepage && <div><dt className="text-sm text-slate-500 mb-1">홈페이지</dt><dd className="text-slate-900"><a href={removeHtmlTags(details.homepage).match(/href="([^"]+)"/)?.[1] || '#'} target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline">바로가기</a></dd></div>}
            </dl>
          </div>
          )} {/* End of Basic Information Section conditional */}
          <div className="card p-6 bg-gradient-to-br from-primary-50 to-accent-50">
            <h3 className="text-lg font-bold mb-3 flex items-center space-x-2"><span><FaInfoCircle /></span><span>참고</span></h3>
            <p className="text-sm text-slate-700 leading-relaxed">코스 정보는 현장 사정에 따라 변경될 수 있습니다. 방문 전 확인하시는 것을 권장합니다.</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
