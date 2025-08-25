import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"

const distributionData = [
  { name: "Neurological", count: 437, percentage: 35.0 },
  { name: "Cardiovascular", count: 349, percentage: 28.0 },
  { name: "Oncological", count: 274, percentage: 22.0 },
  { name: "Hepatorenal", count: 187, percentage: 15.0 },
]

const COLORS = ["#15803d", "#84cc16", "#ea580c", "#dc2626"]

export function ClassDistribution() {
  const chartConfig = {
    name: {
      label: "Domain",
      color: "hsl(var(--muted-foreground))",
    },
    count: {
      label: "Articles",
      color: "hsl(var(--primary))",
    },
    percentage: {
      label: "Percentage",
      color: "hsl(var(--primary))",
    },
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Class Distribution - Bar Chart</CardTitle>
          <CardDescription>Number of articles per medical domain</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distributionData}>
                <XAxis dataKey="name" tick={{ fontSize: 12 }} angle={-45} textAnchor="end" height={80} />
                <YAxis tick={{ fontSize: 12 }} />
                <ChartTooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-background border rounded-lg shadow-lg p-3">
                          <p className="font-medium">{label}</p>
                          <p className="text-sm text-muted-foreground">Articles: {payload[0].value}</p>
                        </div>
                      )
                    }
                    return null
                  }}
                />
                <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Class Distribution - Pie Chart</CardTitle>
          <CardDescription>Percentage distribution across domains</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={distributionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percentage }) => `${name}: ${percentage}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {distributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <ChartTooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload
                      return (
                        <div className="bg-background border rounded-lg shadow-lg p-3">
                          <p className="font-medium">{data.name}</p>
                          <p className="text-sm text-muted-foreground">
                            Articles: {data.count} ({data.percentage}%)
                          </p>
                        </div>
                      )
                    }
                    return null
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  )
}
