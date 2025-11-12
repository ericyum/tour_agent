import { PieChart, Pie, Cell, Legend, ResponsiveContainer, Tooltip } from 'recharts'

interface SentimentDonutChartProps {
  data: {
    positive: number
    negative: number
  }
}

export default function SentimentDonutChart({ data }: SentimentDonutChartProps) {
  const total = data.positive + data.negative
  if (total === 0) return <p className="text-center text-slate-500">데이터 없음</p>

  const chartData = [
    { name: '긍정', value: data.positive, percentage: ((data.positive / total) * 100).toFixed(1) },
    { name: '부정', value: data.negative, percentage: ((data.negative / total) * 100).toFixed(1) },
  ]

  const COLORS = {
    긍정: '#5463FF',
    부정: '#FF1818',
  }

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white px-4 py-2 border border-slate-200 rounded-lg shadow-lg">
          <p className="font-semibold text-slate-800">{payload[0].name}</p>
          <p className="text-slate-600">
            {payload[0].value}개 ({payload[0].payload.percentage}%)
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={100}
          paddingAngle={2}
          dataKey="value"
          label={({ percentage }) => `${percentage}%`}
        >
          {chartData.map((entry) => (
            <Cell key={`cell-${entry.name}`} fill={COLORS[entry.name as keyof typeof COLORS]} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend
          verticalAlign="bottom"
          height={36}
          formatter={(value, entry: any) => `${value} (${entry.payload.percentage}%)`}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}
