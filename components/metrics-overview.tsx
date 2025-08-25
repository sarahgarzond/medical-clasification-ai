import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, Target, Zap, CheckCircle } from "lucide-react"

export function MetricsOverview() {
  const metrics = [
    {
      title: "F1-Score",
      value: "0.923",
      description: "Weighted average F1-score",
      icon: Target,
      trend: "+8.9%",
      color: "text-blue-500",
    },
    {
      title: "Accuracy",
      value: "91.7%",
      description: "Overall classification accuracy",
      icon: CheckCircle,
      trend: "+7.6%",
      color: "text-green-500",
    },
    {
      title: "Precision",
      value: "0.918",
      description: "Macro-averaged precision",
      icon: Zap,
      trend: "+10.5%",
      color: "text-teal-500",
    },
    {
      title: "Recall",
      value: "0.928",
      description: "Macro-averaged recall",
      icon: TrendingUp,
      trend: "+7.5%",
      color: "text-indigo-500",
    },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {metrics.map((metric, index) => {
        const Icon = metric.icon
        return (
          <Card key={index} className="relative overflow-hidden border-slate-200 bg-white/80 shadow-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
              <CardTitle className="text-sm font-medium text-slate-700">{metric.title}</CardTitle>
              <Icon className={`h-5 w-5 ${metric.color}`} />
            </CardHeader>
            <CardContent className="pt-0">
              <div className="text-2xl font-bold text-slate-800 mb-1">{metric.value}</div>
              <p className="text-xs text-slate-500 mb-3">{metric.description}</p>
              <Badge variant="secondary" className="text-xs bg-slate-100 text-slate-600">
                {metric.trend} vs Random Forest
              </Badge>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
