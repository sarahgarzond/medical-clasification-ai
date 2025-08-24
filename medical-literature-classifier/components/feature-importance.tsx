import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from "recharts"

const featureData = [
  { feature: "neural", importance: 0.156, domain: "Neurological" },
  { feature: "cardiac", importance: 0.143, domain: "Cardiovascular" },
  { feature: "tumor", importance: 0.128, domain: "Oncological" },
  { feature: "brain", importance: 0.121, domain: "Neurological" },
  { feature: "heart", importance: 0.115, domain: "Cardiovascular" },
  { feature: "cancer", importance: 0.109, domain: "Oncological" },
  { feature: "kidney", importance: 0.098, domain: "Hepatorenal" },
  { feature: "liver", importance: 0.087, domain: "Hepatorenal" },
  { feature: "stroke", importance: 0.076, domain: "Neurological" },
  { feature: "artery", importance: 0.065, domain: "Cardiovascular" },
]

export function FeatureImportance() {
  const chartConfig = {
    feature: {
      label: "Feature",
      color: "hsl(var(--muted-foreground))",
    },
    importance: {
      label: "Importance Score",
      color: "hsl(var(--primary))",
    },
    domain: {
      label: "Domain",
      color: "hsl(var(--muted-foreground))",
    },
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Top Feature Importance</CardTitle>
          <CardDescription>Most influential terms for classification decisions</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureData} layout="horizontal" margin={{ left: 60 }}>
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="feature" type="category" tick={{ fontSize: 12 }} width={60} />
                <ChartTooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload
                      return (
                        <div className="bg-background border rounded-lg shadow-lg p-3">
                          <p className="font-medium">{data.feature}</p>
                          <p className="text-sm text-muted-foreground">Importance: {data.importance.toFixed(3)}</p>
                          <p className="text-sm text-muted-foreground">Domain: {data.domain}</p>
                        </div>
                      )
                    }
                    return null
                  }}
                />
                <Bar dataKey="importance" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Feature Analysis</CardTitle>
          <CardDescription>Key insights from feature importance</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div className="p-3 bg-muted rounded-lg">
              <h4 className="font-medium text-sm">Domain-Specific Terms</h4>
              <p className="text-xs text-muted-foreground mt-1">
                Medical terminology strongly correlates with correct classification
              </p>
            </div>

            <div className="p-3 bg-muted rounded-lg">
              <h4 className="font-medium text-sm">Anatomical References</h4>
              <p className="text-xs text-muted-foreground mt-1">Body part mentions are highly predictive features</p>
            </div>

            <div className="p-3 bg-muted rounded-lg">
              <h4 className="font-medium text-sm">Clinical Conditions</h4>
              <p className="text-xs text-muted-foreground mt-1">
                Disease and condition names provide strong classification signals
              </p>
            </div>
          </div>

          <div className="pt-4 border-t">
            <h4 className="font-medium text-sm mb-2">Top Features by Domain:</h4>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Neurological:</span>
                <span>neural, brain, stroke</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Cardiovascular:</span>
                <span>cardiac, heart, artery</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Oncological:</span>
                <span>tumor, cancer</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Hepatorenal:</span>
                <span>kidney, liver</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
