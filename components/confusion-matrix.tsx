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
    return `rgba(59, 130, 246, ${intensity})` // Using blue color with varying opacity
  }

  return (
    <Card className="border-slate-200 bg-white/80 shadow-sm">
      <CardHeader>
        <CardTitle className="text-slate-700">Confusion Matrix</CardTitle>
        <CardDescription className="text-slate-500">Model performance across different medical domains</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-5 gap-3 text-sm">
            <div className="w-24"></div>
            {classes.map((cls, idx) => (
              <div key={idx} className="text-center font-medium text-xs p-2 text-slate-600">
                <div className="truncate">{cls}</div>
              </div>
            ))}
          </div>

          {confusionData.map((row, rowIdx) => (
            <div key={rowIdx} className="grid grid-cols-5 gap-3">
              <div className="text-sm font-medium p-2 text-right text-slate-600 w-24 flex items-center justify-end">
                <div className="truncate">{classes[rowIdx]}</div>
              </div>
              {row.map((value, colIdx) => (
                <div
                  key={colIdx}
                  className="w-16 h-16 flex items-center justify-center text-sm font-medium rounded border border-slate-200"
                  style={{
                    backgroundColor: getIntensity(value),
                    color: value > maxValue * 0.5 ? "white" : "#374151",
                  }}
                >
                  {value}
                </div>
              ))}
            </div>
          ))}

          <div className="flex items-center justify-between text-xs text-slate-500 mt-6 pt-4 border-t border-slate-200">
            <span>Predicted Labels</span>
            <div className="flex items-center gap-2">
              <span>Low</span>
              <div className="w-20 h-3 bg-gradient-to-r from-slate-200 to-blue-500 rounded"></div>
              <span>High</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
