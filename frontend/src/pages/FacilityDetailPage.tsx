import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { FaSpinner, FaMapMarkerAlt, FaPhone, FaGlobe, FaInfoCircle } from 'react-icons/fa';
import { getFacilityDetails } from '@/lib/api'; // Will create this API function
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

export default function FacilityDetailPage() {
  const { facilityTitle } = useParams<{ facilityTitle: string }>();
  const addItem = useCourseStore((state) => state.addItem);
  const courseItems = useCourseStore((state) => state.courseItems);

  const { data, isLoading, error } = useQuery({
    queryKey: ['facility-details', facilityTitle],
    queryFn: () => getFacilityDetails(facilityTitle!),
    enabled: !!facilityTitle,
  });

  if (isLoading) {
    return <LoadingSpinner text="관광시설 정보를 불러오는 중..." />;
  }

  if (error || !data) {
    return <ErrorMessage text="관광시설 정보를 찾을 수 없습니다" />;
  }

  const { details } = data;
  const isInCourse = courseItems.some((item) => item.title === details.title);

  const handleAddToCourse = () => {
    addItem({ ...details, type: 'facility', title: details.title || '' });
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
      {/* Hero Section */}
      <div className="card overflow-hidden">
        <div className="relative h-96">
          <img src={details.firstimage || 'https://placehold.co/1200x400?text=Facility'} alt={details.title} className="w-full h-full object-cover" />
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
          {details.overview && <div className="card p-6"><h2 className="text-2xl font-bold mb-4">시설 소개</h2><p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{removeHtmlTags(details.overview)}</p></div>}
          {details.infocenter && <div className="card p-6"><h2 className="text-2xl font-bold mb-4">문의 및 안내</h2><p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{removeHtmlTags(details.infocenter)}</p></div>}
        </div>
        <div className="space-y-6">
          { (details.addr1 || details.mapx || details.mapy || details.tel || details.homepage || details.infocenterculture || details.usefee || details.usetimeculture || details.restdateculture || details.parkingculture || details.chkcreditcardculture || details.chkbabycarriageculture || details.chkpetculture || details.spendtime || details.accomcountculture || details.scale || details.discountinfo) && (
            <div className="card p-6">
              <h3 className="text-xl font-bold mb-4">기본 정보</h3>
              <dl className="space-y-3">
              {details.addr1 && <div><dt className="text-sm text-slate-500 mb-1">주소</dt><dd className="text-slate-900">{details.addr1}</dd></div>}
              {details.mapx && details.mapy && <div><dt className="text-sm text-slate-500 mb-1">좌표</dt><dd className="text-slate-900">{details.mapy}, {details.mapx}</dd></div>}
              {details.tel && <div><dt className="text-sm text-slate-500 mb-1">전화번호</dt><dd className="flex items-center space-x-2"><FaPhone className="text-primary-600" /><a href={`tel:${details.tel}`} className="text-slate-900 hover:text-primary-600">{details.tel}</a></dd></div>}
              {details.homepage && <div><dt className="text-sm text-slate-500 mb-1">홈페이지</dt><dd className="flex items-center space-x-2"><FaGlobe className="text-primary-600" /><a href={removeHtmlTags(details.homepage).match(/href="([^"]+)"/)?.[1] || '#'} target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline">바로가기</a></dd></div>}
              {details.infocenterculture && <div><dt className="text-sm text-slate-500 mb-1">문의 및 안내</dt><dd className="text-slate-900">{removeHtmlTags(details.infocenterculture)}</dd></div>}
              {details.usefee && <div><dt className="text-sm text-slate-500 mb-1">이용 요금</dt><dd className="text-slate-900">{removeHtmlTags(details.usefee)}</dd></div>}
              {details.usetimeculture && <div><dt className="text-sm text-slate-500 mb-1">이용 시간</dt><dd className="text-slate-900">{removeHtmlTags(details.usetimeculture)}</dd></div>}
              {details.restdateculture && <div><dt className="text-sm text-slate-500 mb-1">쉬는 날</dt><dd className="text-slate-900">{removeHtmlTags(details.restdateculture)}</dd></div>}
              {details.parkingculture && <div><dt className="text-sm text-slate-500 mb-1">주차 시설</dt><dd className="text-slate-900">{removeHtmlTags(details.parkingculture)}</dd></div>}
              {details.chkcreditcardculture && <div><dt className="text-sm text-slate-500 mb-1">신용카드</dt><dd className="text-slate-900">{removeHtmlTags(details.chkcreditcardculture)}</dd></div>}
              {details.chkbabycarriageculture && <div><dt className="text-sm text-slate-500 mb-1">유모차</dt><dd className="text-slate-900">{removeHtmlTags(details.chkbabycarriageculture)}</dd></div>}
              {details.chkpetculture && <div><dt className="text-sm text-slate-500 mb-1">반려동물</dt><dd className="text-slate-900">{removeHtmlTags(details.chkpetculture)}</dd></div>}
              {details.spendtime && <div><dt className="text-sm text-slate-500 mb-1">소요 시간</dt><dd className="text-slate-900">{removeHtmlTags(details.spendtime)}</dd></div>}
              {details.accomcountculture && <div><dt className="text-sm text-slate-500 mb-1">수용 인원</dt><dd className="text-slate-900">{removeHtmlTags(details.accomcountculture)}</dd></div>}
              {details.scale && <div><dt className="text-sm text-slate-500 mb-1">규모</dt><dd className="text-slate-900">{removeHtmlTags(details.scale)}</dd></div>}
              {details.discountinfo && <div><dt className="text-sm text-slate-500 mb-1">할인 정보</dt><dd className="text-slate-900">{removeHtmlTags(details.discountinfo)}</dd></div>}
            </dl>
          </div>
          )} {/* End of Basic Information Section conditional */}
          <div className="card p-6 bg-gradient-to-br from-primary-50 to-accent-50">
            <h3 className="text-lg font-bold mb-3 flex items-center space-x-2"><span><FaInfoCircle /></span><span>참고</span></h3>
            <p className="text-sm text-slate-700 leading-relaxed">시설 정보는 현장 사정에 따라 변경될 수 있습니다. 방문 전 확인하시는 것을 권장합니다.</p>
          </div>
          {/* Additional Information Section */}
          { (details.체험프로그램 || details.한국어안내 || details.외국어안내) && (
            <div className="card p-6">
              <h3 className="text-xl font-bold mb-4">추가 정보</h3>
              <dl className="space-y-3">
              {details.체험프로그램 && <div><dt className="text-sm text-slate-500 mb-1">체험 프로그램</dt><dd className="text-slate-900">{removeHtmlTags(details.체험프로그램)}</dd></div>}
              {details.한국어안내 && <div><dt className="text-sm text-slate-500 mb-1">한국어 안내</dt><dd className="text-slate-900">{removeHtmlTags(details.한국어안내)}</dd></div>}
              {details.외국어안내 && <div><dt className="text-sm text-slate-500 mb-1">외국어 안내</dt><dd className="text-slate-900">{removeHtmlTags(details.외국어안내)}</dd></div>}
            </dl>
          </div>
          )} {/* End of Additional Information Section conditional */}
        </div>
      </div>
    </motion.div>
  );
}
