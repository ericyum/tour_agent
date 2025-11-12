interface OutlierBoxPlotProps {
  data: {
    min: number
    q1: number
    median: number
    q3: number
    max: number
    lower_bound: number
    upper_bound: number
    outliers: number[]
  }
}

export default function OutlierBoxPlot({ data }: OutlierBoxPlotProps) {
  if (!data) {
    return <p className="text-center text-slate-500">데이터 없음</p>
  }

  // SVG dimensions
  const width = 800
  const height = 200
  const padding = { left: 60, right: 60, top: 40, bottom: 40 }

  // Calculate scale
  const allValues = [data.min, data.max, ...data.outliers]
  const dataMin = Math.min(...allValues)
  const dataMax = Math.max(...allValues)
  const range = dataMax - dataMin
  const xMin = dataMin - range * 0.1
  const xMax = dataMax + range * 0.1

  // Scale function
  const scale = (value: number) => {
    const plotWidth = width - padding.left - padding.right
    return padding.left + ((value - xMin) / (xMax - xMin)) * plotWidth
  }

  // Box plot vertical position
  const boxY = height / 2
  const boxHeight = 60

  // Scaled positions
  const q1X = scale(data.q1)
  const medianX = scale(data.median)
  const q3X = scale(data.q3)

  // Determine whisker ends (should be within bounds)
  const lowerWhiskerEnd = Math.max(data.min, data.lower_bound)
  const upperWhiskerEnd = Math.min(data.max, data.upper_bound)
  const lowerWhiskerX = scale(lowerWhiskerEnd)
  const upperWhiskerX = scale(upperWhiskerEnd)

  return (
    <div className="w-full">
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="w-full">
        {/* Grid lines */}
        {[...Array(11)].map((_, i) => {
          const x = padding.left + (width - padding.left - padding.right) * (i / 10)
          return (
            <line
              key={`grid-${i}`}
              x1={x}
              y1={padding.top}
              x2={x}
              y2={height - padding.bottom}
              stroke="#e5e7eb"
              strokeDasharray="3,3"
            />
          )
        })}

        {/* Lower whisker line */}
        <line
          x1={lowerWhiskerX}
          y1={boxY}
          x2={q1X}
          y2={boxY}
          stroke="#3b82f6"
          strokeWidth={2}
        />

        {/* Lower whisker cap */}
        <line
          x1={lowerWhiskerX}
          y1={boxY - boxHeight / 4}
          x2={lowerWhiskerX}
          y2={boxY + boxHeight / 4}
          stroke="#1f2937"
          strokeWidth={2}
        />

        {/* Box (Q1 to Q3) */}
        <rect
          x={q1X}
          y={boxY - boxHeight / 2}
          width={q3X - q1X}
          height={boxHeight}
          fill="#93c5fd"
          stroke="#3b82f6"
          strokeWidth={1.5}
        />

        {/* Median line */}
        <line
          x1={medianX}
          y1={boxY - boxHeight / 2}
          x2={medianX}
          y2={boxY + boxHeight / 2}
          stroke="#dc2626"
          strokeWidth={3}
        />

        {/* Upper whisker line */}
        <line
          x1={q3X}
          y1={boxY}
          x2={upperWhiskerX}
          y2={boxY}
          stroke="#3b82f6"
          strokeWidth={2}
        />

        {/* Upper whisker cap */}
        <line
          x1={upperWhiskerX}
          y1={boxY - boxHeight / 4}
          x2={upperWhiskerX}
          y2={boxY + boxHeight / 4}
          stroke="#1f2937"
          strokeWidth={2}
        />

        {/* Outliers */}
        {data.outliers.map((outlier, index) => (
          <circle
            key={`outlier-${index}`}
            cx={scale(outlier)}
            cy={boxY}
            r={6}
            fill="#dc2626"
            opacity={0.6}
          />
        ))}

        {/* X-axis */}
        <line
          x1={padding.left}
          y1={height - padding.bottom}
          x2={width - padding.right}
          y2={height - padding.bottom}
          stroke="#374151"
          strokeWidth={1}
        />

        {/* X-axis labels */}
        {[...Array(11)].map((_, i) => {
          const value = xMin + (xMax - xMin) * (i / 10)
          const x = padding.left + (width - padding.left - padding.right) * (i / 10)
          return (
            <text
              key={`label-${i}`}
              x={x}
              y={height - padding.bottom + 20}
              textAnchor="middle"
              fontSize="11"
              fill="#6b7280"
            >
              {value.toFixed(1)}
            </text>
          )
        })}

        {/* X-axis title */}
        <text
          x={width / 2}
          y={height - 5}
          textAnchor="middle"
          fontSize="14"
          fill="#374151"
          fontWeight="500"
        >
          감성 점수
        </text>

        {/* Outlier count */}
        <text
          x={width - padding.right}
          y={padding.top - 10}
          textAnchor="end"
          fontSize="13"
          fill="#dc2626"
          fontWeight="600"
        >
          이상치 개수: {data.outliers.length}개
        </text>
      </svg>

      {/* Statistics summary */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
        <div className="bg-slate-50 p-3 rounded-lg">
          <p className="text-slate-500 text-xs">최소값</p>
          <p className="font-semibold text-slate-800">{data.min.toFixed(2)}</p>
        </div>
        <div className="bg-slate-50 p-3 rounded-lg">
          <p className="text-slate-500 text-xs">Q1 (25%)</p>
          <p className="font-semibold text-slate-800">{data.q1.toFixed(2)}</p>
        </div>
        <div className="bg-green-50 p-3 rounded-lg">
          <p className="text-green-600 text-xs">중앙값</p>
          <p className="font-semibold text-green-800">{data.median.toFixed(2)}</p>
        </div>
        <div className="bg-slate-50 p-3 rounded-lg">
          <p className="text-slate-500 text-xs">Q3 (75%)</p>
          <p className="font-semibold text-slate-800">{data.q3.toFixed(2)}</p>
        </div>
        <div className="bg-slate-50 p-3 rounded-lg">
          <p className="text-slate-500 text-xs">최대값</p>
          <p className="font-semibold text-slate-800">{data.max.toFixed(2)}</p>
        </div>
        <div className="bg-red-50 p-3 rounded-lg">
          <p className="text-red-600 text-xs">이상치 개수</p>
          <p className="font-semibold text-red-800">{data.outliers.length}개</p>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-sm justify-center">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-4 bg-blue-300 border border-blue-500"></div>
          <span className="text-slate-600">박스 (Q1~Q3)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-8 h-1 bg-red-600"></div>
          <span className="text-slate-600">중앙값</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-8 h-1 bg-blue-500"></div>
          <span className="text-slate-600">위스커</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-600 rounded-full opacity-60"></div>
          <span className="text-slate-600">이상치</span>
        </div>
      </div>
    </div>
  )
}
