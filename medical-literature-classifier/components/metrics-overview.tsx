import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, Target, Zap, CheckCircle } from "lucide-react"

export function MetricsOverview() {
  const metrics = [
    {
      title: "F1-Score",
      value: "0.847",
      description: "Weighted average F1-score",
      icon: Target,
      trend: "+2.3%",
      color: "text-primary",
    },
    {
      title: "Accuracy",
      value: "85.2%",
      description: "Overall classification accuracy",
      icon: CheckCircle,
      trend: "+1.8%",
      color: "text-secondary",
    },
    {
      title: "Precision",
      value: "0.831",
      description: "Macro-averaged precision",
      icon: Zap,
      trend: "+3.1%",
      color: "text-chart-3",
    },
    {
      title: "Recall",
      value: "0.863",
      description: "Macro-averaged recall",
      icon: TrendingUp,
      trend: "+2.7%",
      color: "text-chart-4",
    },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {metrics.map((metric, index) => {
        const Icon = metric.icon
        return (
          <Card key={index} className="relative overflow-hidden">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{metric.title}</CardTitle>
              <Icon className={`h-4 w-4 ${metric.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metric.value}</div>
              <p className="text-xs text-muted-foreground mt-1">{metric.description}</p>
              <Badge variant="secondary" className="mt-2 text-xs">
                {metric.trend} vs baseline
              </Badge>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
