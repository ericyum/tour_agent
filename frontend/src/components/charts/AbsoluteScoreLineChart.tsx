import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

interface AbsoluteScoreLineChartProps {
  data: {
    labels: string[]
    counts: number[]
  }
}

export default function AbsoluteScoreLineChart({ data }: AbsoluteScoreLineChartProps) {
  if (!data || data.labels.length === 0) {
    return <p className="text-center text-slate-500">데이터 없음</p>
  }

  const chartData = data.labels.map((label, index) => ({
    name: label,
    count: data.counts[index],
  }))

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white px-4 py-2 border border-slate-200 rounded-lg shadow-lg">
          <p className="font-semibold text-slate-800">{payload[0].payload.name}</p>
          <p className="text-slate-600">{payload[0].value}개 문장</p>
        </div>
      )
    }
    return null
  }

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={chartData} margin={{ top: 40, right: 30, left: 20, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
        <XAxis
          dataKey="name"
          tick={{ fontSize: 11 }}
          angle={-15}
          textAnchor="end"
          height={80}
        />
        <YAxis allowDecimals={false} tick={{ fontSize: 12 }} label={{ value: '문장 수', angle: -90, position: 'insideLeft' }} />
        <Tooltip content={<CustomTooltip />} />
        <Legend verticalAlign="top" height={36} />
        <Line
          type="monotone"
          dataKey="count"
          stroke="#1e90ff"
          strokeWidth={2}
          dot={{ fill: '#1e90ff', r: 5 }}
          activeDot={{ r: 8 }}
          name="문장 수"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
