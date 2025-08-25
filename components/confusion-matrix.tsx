import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export function ConfusionMatrix() {
  // Sample confusion matrix data
  const confusionData = [
    [89, 3, 2, 1], // Neurological
    [4, 76, 3, 2], // Cardiovascular
    [2, 5, 68, 3], // Oncological
    [1, 2, 4, 45], // Hepatorenal
  ]

  const classes = ["Neurological", "Cardiovascular", "Oncological", "Hepatorenal"]
  const maxValue = Math.max(...confusionData.flat())

  const getIntensity = (value: number) => {
    const intensity = value / maxValue
    // Using primary blue color with varying opacity
    return `rgba(29, 78, 216, ${intensity})`
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Confusion Matrix</CardTitle>
        <CardDescription>Model performance across different medical domains</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-5 gap-2 text-sm">
            <div></div>
            {classes.map((cls, idx) => (
              <div key={idx} className="text-center font-medium text-xs p-2">
                {cls}
              </div>
            ))}
          </div>

          {confusionData.map((row, rowIdx) => (
            <div key={rowIdx} className="grid grid-cols-5 gap-2">
              <div className="text-sm font-medium p-2 text-right">{classes[rowIdx]}</div>
              {row.map((value, colIdx) => (
                <div
                  key={colIdx}
                  className="aspect-square flex items-center justify-center text-sm font-medium rounded border"
                  style={{
                    backgroundColor: getIntensity(value),
                    color: value > maxValue * 0.5 ? "white" : "black",
                  }}
                >
                  {value}
                </div>
              ))}
            </div>
          ))}

          <div className="flex items-center justify-between text-xs text-muted-foreground mt-4">
            <span>Predicted Labels</span>
            <div className="flex items-center gap-2">
              <span>Low</span>
              <div className="w-20 h-3 bg-gradient-to-r from-white to-primary rounded"></div>
              <span>High</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
