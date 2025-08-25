import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { title, abstract, max_length = 512 } = body

    if (!title || !abstract) {
      return NextResponse.json({ error: "Title and abstract are required" }, { status: 400 })
    }

    // This simulates the BioBERT classification without requiring external server
    const mockResult = {
      predictions: [
        { class: "neurological", probability: 0.923, predicted: 1 },
        { class: "cardiovascular", probability: 0.045, predicted: 0 },
        { class: "oncological", probability: 0.021, predicted: 0 },
        { class: "hepatorenal", probability: 0.011, predicted: 0 },
      ],
      top_prediction: { class: "neurological", probability: 0.923 },
      active_labels: ["neurological"],
    }

    // Simple classification logic based on keywords in title and abstract
    const text = `${title} ${abstract}`.toLowerCase()

    if (
      text.includes("heart") ||
      text.includes("cardiac") ||
      text.includes("myocardial") ||
      text.includes("cardiovascular")
    ) {
      return NextResponse.json({
        predictions: [
          { class: "cardiovascular", probability: 0.891, predicted: 1 },
          { class: "neurological", probability: 0.067, predicted: 0 },
          { class: "oncological", probability: 0.028, predicted: 0 },
          { class: "hepatorenal", probability: 0.014, predicted: 0 },
        ],
        top_prediction: { class: "cardiovascular", probability: 0.891 },
        active_labels: ["cardiovascular"],
      })
    }

    if (
      text.includes("cancer") ||
      text.includes("tumor") ||
      text.includes("oncolog") ||
      text.includes("metasta") ||
      text.includes("melanoma")
    ) {
      return NextResponse.json({
        predictions: [
          { class: "oncological", probability: 0.876, predicted: 1 },
          { class: "neurological", probability: 0.078, predicted: 0 },
          { class: "cardiovascular", probability: 0.031, predicted: 0 },
          { class: "hepatorenal", probability: 0.015, predicted: 0 },
        ],
        top_prediction: { class: "oncological", probability: 0.876 },
        active_labels: ["oncological"],
      })
    }

    if (
      text.includes("kidney") ||
      text.includes("liver") ||
      text.includes("hepat") ||
      text.includes("renal") ||
      text.includes("hepatorenal")
    ) {
      return NextResponse.json({
        predictions: [
          { class: "hepatorenal", probability: 0.834, predicted: 1 },
          { class: "neurological", probability: 0.089, predicted: 0 },
          { class: "cardiovascular", probability: 0.045, predicted: 0 },
          { class: "oncological", probability: 0.032, predicted: 0 },
        ],
        top_prediction: { class: "hepatorenal", probability: 0.834 },
        active_labels: ["hepatorenal"],
      })
    }

    // Default to neurological for brain/neuro related terms or fallback
    return NextResponse.json(mockResult)
  } catch (error) {
    console.error("Prediction API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
