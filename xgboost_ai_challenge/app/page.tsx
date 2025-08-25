"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from "recharts"
import { Activity, Brain, Heart, Zap, TrendingUp, FileText, CheckCircle, Clock } from "lucide-react"

interface PredictionResult {
  text: string;
  predicted_categories: string[];
  probabilities: Record<string, number>;
}

interface ApiStats {
  model_performance: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    training_samples: number;
    test_samples: number;
    categories: string[];
  };
  confusion_matrix: Record<string, { tp: number; fp: number; fn: number; tn: number }>;
  feature_importance: Array<{ feature: string; importance: number }>;
  training_history: Array<{ iteration: number; train_loss: number; val_loss: number; train_accuracy: number; val_accuracy: number }>;
}

interface ApiExamples {
  examples: Array<{
    title: string;
    text: string;
    expected_categories: string[];
  }>;
}

export default function MedicalDashboard() {
  const [inputText, setInputText] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)

  // Estados para datos reales de APIs
  const [realStats, setRealStats] = useState<ApiStats | null>(null)
  const [realExamples, setRealExamples] = useState<ApiExamples | null>(null)
  const [apiHealth, setApiHealth] = useState<any>(null)

  // Cargar datos reales al inicializar
  useEffect(() => {
    loadRealData()
  }, [])

  const loadRealData = async () => {
    try {
      // Cargar estadísticas reales
      const statsResponse = await fetch('/api/statistics')
      if (statsResponse.ok) {
        const stats = await statsResponse.json()
        setRealStats(stats)
      }

      // Cargar ejemplos reales
      const examplesResponse = await fetch('/api/demo-examples')
      if (examplesResponse.ok) {
        const examples = await examplesResponse.json()
        setRealExamples(examples)
      }

      // Verificar health de API
      const healthResponse = await fetch('/api/health')
      if (healthResponse.ok) {
        const health = await healthResponse.json()
        setApiHealth(health)
      }
    } catch (error) {
      console.error('Error cargando datos reales:', error)
    }
  }

  const handlePrediction = async () => {
    if (!inputText.trim()) return
    
    setIsLoading(true)
    
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText })
      })
      
      if (!response.ok) {
        throw new Error('Error en la predicción')
      }
      
      const data = await response.json()
      
      // Adaptar la respuesta a nuestro formato esperado
      const predictionResult: PredictionResult = {
        text: data.text,
        predicted_categories: data.predictions.filter((p: any) => p.predicted).map((p: any) => p.category),
        probabilities: data.predictions.reduce((acc: any, pred: any) => {
          acc[pred.category] = pred.probability
          return acc
        }, {}),
      }
      
      setPrediction(predictionResult)
    } catch (error) {
      console.error('Error en predicción:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const getCategoryColor = (category: string) => {
    const colors = {
      cardiovascular: "#ef4444", // red-500
      neurologico: "#3b82f6", // blue-500
      hepatorenal: "#10b981", // emerald-500
      oncologico: "#f59e0b", // amber-500
    }
    return colors[category as keyof typeof colors] || "#6b7280"
  }

  const getCategoryIcon = (category: string) => {
    const icons = {
      cardiovascular: <Heart className="h-4 w-4" />,
      neurologico: <Brain className="h-4 w-4" />,
      hepatorenal: <Activity className="h-4 w-4" />,
      oncologico: <Zap className="h-4 w-4" />,
    }
    return icons[category as keyof typeof icons] || <FileText className="h-4 w-4" />
  }

  const loadExampleText = (example: string) => {
    setInputText(example)
  }

  if (!realStats || !realExamples) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 p-8 flex items-center justify-center">
        <Card className="w-96">
          <CardContent className="pt-6">
            <div className="flex items-center justify-center space-x-2">
              <div className="w-4 h-4 bg-blue-500 rounded-full animate-pulse"></div>
              <div className="w-4 h-4 bg-blue-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-4 h-4 bg-blue-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
            </div>
            <p className="text-center mt-4 text-muted-foreground">Cargando datos médicos reales...</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100">
      <div className="container mx-auto p-8 space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold text-primary">Medical AI Dashboard</h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            XGBoost Multi-label Classification for Medical Literature Analysis
          </p>
          <div className="flex items-center justify-center space-x-4">
            <Badge variant="secondary" className="px-3 py-1">
              <CheckCircle className="h-4 w-4 mr-2" />
              {apiHealth?.status === 'healthy' ? 'Sistema Activo' : 'Verificando...'}
            </Badge>
            <Badge variant="outline" className="px-3 py-1">
              <Clock className="h-4 w-4 mr-2" />
              Actualizado: {new Date().toLocaleDateString()}
            </Badge>
          </div>
        </div>

        {/* Overview Cards - DATOS REALES */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Accuracy</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-primary">{(realStats.model_performance.accuracy * 100).toFixed(1)}%</div>
              <Badge variant="secondary" className="mt-1">
                <CheckCircle className="h-3 w-3 mr-1" />
                Modelo Entrenado
              </Badge>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">F1-Score</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-primary">{(realStats.model_performance.f1_score * 100).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground mt-1">Promedio macro</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Training Samples</CardTitle>
              <FileText className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-primary">{realStats.model_performance.training_samples.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground mt-1">Muestras médicas</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Categories</CardTitle>
              <Brain className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-primary">{realStats.model_performance.categories.length}</div>
              <p className="text-xs text-muted-foreground mt-1">Especialidades médicas</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="predictor" className="space-y-4">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="predictor">Predictor</TabsTrigger>
            <TabsTrigger value="performance">Rendimiento</TabsTrigger>
            <TabsTrigger value="features">Características</TabsTrigger>
            <TabsTrigger value="training">Entrenamiento</TabsTrigger>
            <TabsTrigger value="examples">Ejemplos</TabsTrigger>
          </TabsList>

          {/* Real-time Predictor */}
          <TabsContent value="predictor" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Predictor de Texto Médico</CardTitle>
                  <CardDescription>Ingresa texto médico para clasificación automática</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    placeholder="Ejemplo: Paciente de 65 años con dolor torácico, disnea y elevación de troponinas..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    rows={6}
                    className="resize-none"
                  />
                  <Button onClick={handlePrediction} disabled={!inputText.trim() || isLoading} className="w-full">
                    {isLoading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Analizando...
                      </>
                    ) : (
                      "Clasificar Texto"
                    )}
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Resultados de Predicción</CardTitle>
                  <CardDescription>Probabilidades y categorías predichas</CardDescription>
                </CardHeader>
                <CardContent>
                  {prediction ? (
                    <div className="space-y-4">
                      <div className="text-center">
                        <Badge variant="default" className="text-lg px-4 py-2">
                          {prediction.predicted_categories.length > 0 ? (
                            <>
                              {getCategoryIcon(prediction.predicted_categories[0])}
                              <span className="ml-2 capitalize">{prediction.predicted_categories[0]}</span>
                            </>
                          ) : (
                            <span>Sin predicción clara</span>
                          )}
                        </Badge>
                        <p className="text-sm text-muted-foreground mt-2">
                          Categorías detectadas: {prediction.predicted_categories.length}
                        </p>
                      </div>

                      <div className="space-y-3">
                        {Object.entries(prediction.probabilities).map(([category, probability]) => (
                          <div key={category} className="space-y-2">
                            <div className="flex justify-between items-center">
                              <div className="flex items-center">
                                {getCategoryIcon(category)}
                                <span className="ml-2 capitalize text-sm font-medium">{category}</span>
                              </div>
                              <span className="text-sm font-mono">{(probability * 100).toFixed(1)}%</span>
                            </div>
                            <Progress 
                              value={probability * 100} 
                              className="h-2"
                              style={{ 
                                backgroundColor: `${getCategoryColor(category)}20`,
                              }}
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Ingresa texto médico para ver predicciones</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Performance Metrics - DATOS REALES */}
          <TabsContent value="performance" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Métricas por Categoría</CardTitle>
                  <CardDescription>Precisión y recall por especialidad médica</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {realStats.model_performance.categories.map((category) => {
                      const confMatrix = realStats.confusion_matrix[category]
                      const precision = confMatrix.tp / (confMatrix.tp + confMatrix.fp)
                      const recall = confMatrix.tp / (confMatrix.tp + confMatrix.fn)
                      const f1 = 2 * (precision * recall) / (precision + recall)
                      
                      return (
                        <div key={category} className="space-y-2">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center">
                              {getCategoryIcon(category)}
                              <span className="ml-2 capitalize font-medium">{category}</span>
                            </div>
                            <Badge variant="outline">{(f1 * 100).toFixed(1)}% F1</Badge>
                          </div>
                          <div className="grid grid-cols-3 gap-2 text-sm">
                            <div className="text-center">
                              <div className="font-mono">{(precision * 100).toFixed(1)}%</div>
                              <div className="text-muted-foreground text-xs">Precisión</div>
                            </div>
                            <div className="text-center">
                              <div className="font-mono">{(recall * 100).toFixed(1)}%</div>
                              <div className="text-muted-foreground text-xs">Recall</div>
                            </div>
                            <div className="text-center">
                              <div className="font-mono">{confMatrix.tp + confMatrix.fn}</div>
                              <div className="text-muted-foreground text-xs">Muestras</div>
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Matriz de Confusión</CardTitle>
                  <CardDescription>Verdaderos positivos por categoría</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    {realStats.model_performance.categories.map((category) => {
                      const confMatrix = realStats.confusion_matrix[category]
                      const total = confMatrix.tp + confMatrix.fp + confMatrix.fn + confMatrix.tn
                      
                      return (
                        <div key={category} className="text-center p-4 rounded-lg border">
                          <div className="flex items-center justify-center mb-2">
                            {getCategoryIcon(category)}
                            <span className="ml-2 text-sm capitalize font-medium">{category}</span>
                          </div>
                          <div className="text-2xl font-bold text-primary">{confMatrix.tp}</div>
                          <div className="text-xs text-muted-foreground">VP de {total}</div>
                        </div>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Feature Importance - DATOS REALES */}
          <TabsContent value="features" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Importancia de Características</CardTitle>
                <CardDescription>Top 10 características más importantes del modelo</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={realStats.feature_importance.slice(0, 10)} layout="horizontal">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="feature" type="category" width={100} />
                    <Tooltip />
                    <Bar dataKey="importance" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Training History - DATOS REALES */}
          <TabsContent value="training" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Historia de Entrenamiento</CardTitle>
                <CardDescription>Progreso de accuracy y loss durante el entrenamiento</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={realStats.training_history}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="iteration" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="train_accuracy" stroke="#10b981" name="Train Accuracy" />
                    <Line type="monotone" dataKey="val_accuracy" stroke="#ef4444" name="Val Accuracy" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Medical Examples - DATOS REALES */}
          <TabsContent value="examples" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {realExamples.examples.map((example, index) => (
                <Card key={index} className="cursor-pointer hover:shadow-md transition-shadow" 
                      onClick={() => loadExampleText(example.text)}>
                  <CardHeader>
                    <CardTitle className="text-lg">{example.title}</CardTitle>
                    <CardDescription>
                      Categorías: {example.expected_categories.map((cat, i) => (
                        <Badge key={i} variant="outline" className="ml-1">
                          {getCategoryIcon(cat)}
                          <span className="ml-1 capitalize">{cat}</span>
                        </Badge>
                      ))}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground line-clamp-3">
                      {example.text}
                    </p>
                    <Button variant="ghost" size="sm" className="mt-2">
                      Usar este ejemplo
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
