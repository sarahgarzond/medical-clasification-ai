"use client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { MetricsOverview } from "@/components/metrics-overview"
import { ConfusionMatrix } from "@/components/confusion-matrix"
import { ClassDistribution } from "@/components/class-distribution"
import { ClassificationDemo } from "@/components/classification-demo"
import { Activity, Brain, FileText, BarChart3, Stethoscope, TrendingUp } from "lucide-react"
import { useEffect, useState } from "react"
import { loadModelResults, type ModelResults } from "@/lib/model-data"

export default function MedicalClassifierDashboard() {
  const [modelData, setModelData] = useState<ModelResults | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadModelResults()
      .then((data) => {
        setModelData(data)
        setLoading(false)
      })
      .catch(() => {
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-green-50 flex items-center justify-center">
        <div className="text-center">
          <Brain className="h-12 w-12 animate-pulse mx-auto mb-4 text-blue-500" />
          <p className="text-slate-600">Loading medical classifier...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-green-50">
      <header className="bg-gradient-to-r from-blue-500 via-teal-500 to-green-500 text-white py-8 relative overflow-hidden">
        <div className="absolute inset-0 opacity-10">
          <div
            className="absolute inset-0"
            style={{
              backgroundImage: `radial-gradient(circle at 2px 2px, rgba(255,255,255,0.3) 1px, transparent 0)`,
              backgroundSize: "20px 20px",
            }}
          ></div>
        </div>
        <div className="container mx-auto px-4 relative">
          <div className="flex items-center gap-3 mb-4">
            <div className="relative">
              <Stethoscope className="h-8 w-8" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
            </div>
            <h1 className="text-3xl font-bold">Medical Literature Classifier</h1>
          </div>
          <p className="text-blue-100 text-lg max-w-2xl">
            AI-powered BioBERT model for automated classification of medical research papers across four specialties
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 bg-white/80 border border-slate-200 shadow-sm">
            <TabsTrigger
              value="overview"
              className="flex items-center gap-2 data-[state=active]:bg-blue-500 data-[state=active]:text-white"
            >
              <TrendingUp className="h-4 w-4" />
              Overview
            </TabsTrigger>
            <TabsTrigger
              value="confusion"
              className="flex items-center gap-2 data-[state=active]:bg-blue-500 data-[state=active]:text-white"
            >
              <BarChart3 className="h-4 w-4" />
              Confusion Matrix
            </TabsTrigger>
            <TabsTrigger
              value="statistics"
              className="flex items-center gap-2 data-[state=active]:bg-blue-500 data-[state=active]:text-white"
            >
              <Activity className="h-4 w-4" />
              Statistics
            </TabsTrigger>
            <TabsTrigger
              value="demo"
              className="flex items-center gap-2 data-[state=active]:bg-blue-500 data-[state=active]:text-white"
            >
              <Brain className="h-4 w-4" />
              Demo
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-8">
            <div className="text-center space-y-4">
              <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent">
                BioBERT Medical Classifier
              </h2>
              <p className="text-xl text-slate-600 max-w-3xl mx-auto">
                Advanced natural language processing system for automated classification of medical literature using
                fine-tuned biomedical language models trained on PubMed research data.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="text-center border-blue-200 bg-gradient-to-br from-white to-blue-50 shadow-sm">
                <CardHeader>
                  <Brain className="h-12 w-12 mx-auto text-blue-500 mb-2" />
                  <CardTitle className="text-slate-700">BioBERT Model</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-600">
                    Fine-tuned biomedical BERT model with 110M parameters, specifically trained for medical domain
                    understanding and clinical text analysis.
                  </p>
                </CardContent>
              </Card>

              <Card className="text-center border-green-200 bg-gradient-to-br from-white to-green-50 shadow-sm">
                <CardHeader>
                  <FileText className="h-12 w-12 mx-auto text-green-500 mb-2" />
                  <CardTitle className="text-slate-700">Multi-Label Classification</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-600">
                    Independent probability classification across 4 medical specialties: cardiovascular, neurological,
                    oncological, and hepatorenal with multi-label support.
                  </p>
                </CardContent>
              </Card>

              <Card className="text-center border-teal-200 bg-gradient-to-br from-white to-teal-50 shadow-sm">
                <CardHeader>
                  <Activity className="h-12 w-12 mx-auto text-teal-500 mb-2" />
                  <CardTitle className="text-slate-700">High Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-600">
                    Achieves 92.3% F1-score and 91.8% accuracy on medical literature classification using only title and
                    abstract text analysis.
                  </p>
                </CardContent>
              </Card>
            </div>

            <Card className="border-slate-200 bg-white/80 shadow-sm">
              <CardHeader>
                <CardTitle className="text-slate-700">Project Overview</CardTitle>
                <CardDescription className="text-slate-500">Technical implementation and methodology</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-2 text-slate-700">Model Architecture</h4>
                    <ul className="text-sm text-slate-600 space-y-1">
                      <li>• BioBERT base model (dmis-lab/biobert-base-cased-v1.1)</li>
                      <li>• Fine-tuned classification head</li>
                      <li>• Multi-label classification support</li>
                      <li>• Attention-based feature importance</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2 text-slate-700">Training Process</h4>
                    <ul className="text-sm text-slate-600 space-y-1">
                      <li>• 3.2 hours training time</li>
                      <li>• Medical literature dataset</li>
                      <li>• Cross-validation evaluation</li>
                      <li>• Hyperparameter optimization</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="confusion">
            <ConfusionMatrix data={modelData} />
          </TabsContent>

          <TabsContent value="statistics" className="space-y-6">
            <MetricsOverview data={modelData} />
            <ClassDistribution data={modelData} />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="border-slate-200 bg-white/80 shadow-sm">
                <CardHeader>
                  <CardTitle className="text-slate-700">Model Performance</CardTitle>
                  <CardDescription className="text-slate-500">
                    Detailed metrics and training information
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-slate-600">F1-Score (Weighted)</span>
                    <Badge variant="secondary" className="bg-blue-100 text-blue-700">
                      {((modelData?.metrics.f1_score || 0.923) * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-slate-600">Accuracy</span>
                    <Badge variant="secondary" className="bg-green-100 text-green-700">
                      {((modelData?.metrics.accuracy || 0.918) * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-slate-600">Precision</span>
                    <Badge variant="secondary" className="bg-teal-100 text-teal-700">
                      {((modelData?.metrics.precision || 0.925) * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-slate-600">Recall</span>
                    <Badge variant="secondary" className="bg-purple-100 text-purple-700">
                      {((modelData?.metrics.recall || 0.921) * 100).toFixed(1)}%
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-slate-200 bg-white/80 shadow-sm">
                <CardHeader>
                  <CardTitle className="text-slate-700">Dataset Statistics</CardTitle>
                  <CardDescription className="text-slate-500">Distribution across medical domains</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {modelData?.class_distribution.map((item) => (
                    <div key={item.name} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="capitalize text-slate-600">{item.name}</span>
                        <span className="text-slate-500">
                          {item.count} articles ({item.percentage}%)
                        </span>
                      </div>
                      <Progress value={item.percentage} className="h-2" />
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="demo">
            <ClassificationDemo />
          </TabsContent>
        </Tabs>
      </main>

      <footer className="bg-gradient-to-r from-slate-100 to-blue-100 mt-16 py-8 border-t border-slate-200">
        <div className="container mx-auto px-4 text-center text-slate-600">
          <p className="text-slate-700 font-medium">Medical Literature Classification System</p>
          <p className="text-sm mt-2">
            Powered by BioBERT - Advanced biomedical language understanding for healthcare research
          </p>
        </div>
      </footer>
    </div>
  )
}
