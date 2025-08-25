"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Loader2, FileText, Brain } from "lucide-react"

export function ClassificationDemo() {
  const [title, setTitle] = useState("")
  const [abstract, setAbstract] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<{
    prediction: string
    confidence: number
    probabilities: { class: string; probability: number }[]
  } | null>(null)

  const sampleArticles = [
    {
      title: "Deep brain stimulation for treatment-resistant depression",
      abstract:
        "Background: Treatment-resistant depression affects millions worldwide. This study evaluates deep brain stimulation as a therapeutic intervention. Methods: We conducted a randomized controlled trial with 120 patients. Results: Significant improvement in depression scores was observed.",
      expectedClass: "Neurological",
    },
    {
      title: "Cardiac catheterization outcomes in elderly patients",
      abstract:
        "Objective: To assess the safety and efficacy of cardiac catheterization in patients over 75 years. Methods: Retrospective analysis of 500 procedures. Results: Low complication rates with good clinical outcomes were observed in this population.",
      expectedClass: "Cardiovascular",
    },
    {
      title: "Novel immunotherapy approaches in lung cancer treatment",
      abstract:
        "Introduction: Lung cancer remains a leading cause of cancer mortality. This review examines recent advances in immunotherapy. We analyze checkpoint inhibitors and their mechanisms. Clinical trials show promising results for patient survival.",
      expectedClass: "Oncological",
    },
  ]

  const handleClassify = async () => {
    if (!title.trim() || !abstract.trim()) return

    setIsLoading(true)

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000))

    // Mock classification result
    const mockResult = {
      prediction: "Neurological",
      confidence: 0.847,
      probabilities: [
        { class: "Neurological", probability: 0.847 },
        { class: "Cardiovascular", probability: 0.089 },
        { class: "Oncological", probability: 0.041 },
        { class: "Hepatorenal", probability: 0.023 },
      ],
    }

    setResult(mockResult)
    setIsLoading(false)
  }

  const loadSample = (sample: (typeof sampleArticles)[0]) => {
    setTitle(sample.title)
    setAbstract(sample.abstract)
    setResult(null)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Article Input
          </CardTitle>
          <CardDescription>Enter a medical article title and abstract for classification</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="title">Article Title</Label>
            <Input
              id="title"
              placeholder="Enter the article title..."
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="abstract">Abstract</Label>
            <Textarea
              id="abstract"
              placeholder="Enter the article abstract..."
              value={abstract}
              onChange={(e) => setAbstract(e.target.value)}
              rows={8}
            />
          </div>

          <Button onClick={handleClassify} disabled={!title.trim() || !abstract.trim() || isLoading} className="w-full">
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Classifying...
              </>
            ) : (
              <>
                <Brain className="mr-2 h-4 w-4" />
                Classify Article
              </>
            )}
          </Button>

          <div className="pt-4 border-t">
            <h4 className="font-medium text-sm mb-3">Try Sample Articles:</h4>
            <div className="space-y-2">
              {sampleArticles.map((sample, idx) => (
                <Button
                  key={idx}
                  variant="outline"
                  size="sm"
                  onClick={() => loadSample(sample)}
                  className="w-full text-left justify-start h-auto p-3"
                >
                  <div>
                    <div className="font-medium text-xs">{sample.title}</div>
                    <div className="text-xs text-muted-foreground mt-1">Expected: {sample.expectedClass}</div>
                  </div>
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Classification Results
          </CardTitle>
          <CardDescription>AI model predictions and confidence scores</CardDescription>
        </CardHeader>
        <CardContent>
          {result ? (
            <div className="space-y-6">
              <div className="text-center p-6 bg-muted rounded-lg">
                <div className="text-2xl font-bold text-primary mb-2">{result.prediction}</div>
                <div className="text-sm text-muted-foreground">Confidence: {(result.confidence * 100).toFixed(1)}%</div>
              </div>

              <div className="space-y-3">
                <h4 className="font-medium text-sm">Probability Distribution:</h4>
                {result.probabilities.map((prob, idx) => (
                  <div key={idx} className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>{prob.class}</span>
                      <span>{(prob.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all duration-500"
                        style={{ width: `${prob.probability * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="p-4 bg-muted rounded-lg">
                <h4 className="font-medium text-sm mb-2">Interpretation:</h4>
                <p className="text-xs text-muted-foreground">
                  The model classified this article as <strong>{result.prediction}</strong> with
                  {result.confidence > 0.8 ? " high" : result.confidence > 0.6 ? " moderate" : " low"} confidence.
                  {result.confidence > 0.8 && " This indicates strong domain-specific terminology was detected."}
                </p>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>Enter an article title and abstract to see classification results</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
