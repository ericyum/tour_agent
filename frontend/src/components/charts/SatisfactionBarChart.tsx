import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

interface SatisfactionBarChartProps {
  data: {
    labels: string[]
    counts: number[]
  }
}

export default function SatisfactionBarChart({ data }: SatisfactionBarChartProps) {
  if (!data || data.labels.length === 0) {
    return <p className="text-center text-slate-500">데이터 없음</p>
  }

  const chartData = data.labels.map((label, index) => ({
    name: label,
    count: data.counts[index],
  }))

  const COLORS: Record<string, string> = {
    '매우 불만족': '#FF5733',
    '불만족': '#FF8C33',
    '보통': '#FFC300',
    '만족': '#A2D9A0',
    '매우 만족': '#4CAF50',
  }

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
    <ResponsiveContainer width="100%" height={350}>
      <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
        <XAxis dataKey="name" tick={{ fontSize: 12 }} />
        <YAxis allowDecimals={false} tick={{ fontSize: 12 }} label={{ value: '문장 수', angle: -90, position: 'insideLeft' }} />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="count" radius={[8, 8, 0, 0]} label={{ position: 'top', fontSize: 12 }}>
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[entry.name] || '#8884d8'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
