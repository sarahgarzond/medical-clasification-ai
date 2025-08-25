"use client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { MetricsOverview } from "@/components/metrics-overview"
import { ConfusionMatrix } from "@/components/confusion-matrix"
import { ClassDistribution } from "@/components/class-distribution"
import { FeatureImportance } from "@/components/feature-importance"
import { ClassificationDemo } from "@/components/classification-demo"
import { Activity, Brain, FileText, BarChart3 } from "lucide-react"

export default function MedicalClassifierDashboard() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header Section */}
      <header className="bg-primary text-primary-foreground py-8">
        <div className="container mx-auto px-4">
          <div className="flex items-center gap-3 mb-4">
            <Brain className="h-8 w-8" />
            <h1 className="text-3xl font-bold">Medical Literature Classifier</h1>
          </div>
          <p className="text-primary-foreground/90 text-lg max-w-2xl">
            AI-powered system for classifying medical articles into specialized domains using title and abstract
            analysis
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="confusion" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Confusion Matrix
            </TabsTrigger>
            <TabsTrigger value="distribution" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Distribution
            </TabsTrigger>
            <TabsTrigger value="features" className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Features
            </TabsTrigger>
            <TabsTrigger value="demo" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Demo
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <MetricsOverview />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Model Information</CardTitle>
                  <CardDescription>Details about the trained classification model</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Algorithm</span>
                    <Badge variant="secondary">Random Forest + TF-IDF</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Training Samples</span>
                    <span className="text-sm">1,247 articles</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Features</span>
                    <span className="text-sm">5,000 TF-IDF terms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Classes</span>
                    <span className="text-sm">4 medical domains</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Dataset Overview</CardTitle>
                  <CardDescription>Statistics about the medical literature dataset</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Neurological</span>
                      <span>35%</span>
                    </div>
                    <Progress value={35} className="h-2" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Cardiovascular</span>
                      <span>28%</span>
                    </div>
                    <Progress value={28} className="h-2" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Oncological</span>
                      <span>22%</span>
                    </div>
                    <Progress value={22} className="h-2" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Hepatorenal</span>
                      <span>15%</span>
                    </div>
                    <Progress value={15} className="h-2" />
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="confusion">
            <ConfusionMatrix />
          </TabsContent>

          <TabsContent value="distribution">
            <ClassDistribution />
          </TabsContent>

          <TabsContent value="features">
            <FeatureImportance />
          </TabsContent>

          <TabsContent value="demo">
            <ClassificationDemo />
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="bg-muted mt-16 py-8">
        <div className="container mx-auto px-4 text-center text-muted-foreground">
          <p>Medical Literature Classification System - Built with AI for Academic Research</p>
          <p className="text-sm mt-2">
            Developed using hybrid ML approach combining Random Forest and TF-IDF vectorization
          </p>
        </div>
      </footer>
    </div>
  )
}
