import { NextResponse } from 'next/server';

export async function GET() {
  // Ejemplos reales extraídos del dataset challenge_data-18-ago.csv
  const examples = [
    {
      id: 1,
      text: "BRCA1 mutations in breast and ovarian cancer families from northeast Brazil: genetic counseling implications and clinical management.",
      predicted_categories: ["oncological"],
      confidence: 0.94,
      source: "Real medical literature"
    },
    {
      id: 2,
      text: "Hypertensive response during dobutamine stress echocardiography: a diagnostic marker for coronary artery disease in diabetic patients.",
      predicted_categories: ["cardiovascular"],
      confidence: 0.89,
      source: "Real medical literature"
    },
    {
      id: 3,
      text: "Brain changes in early-onset Alzheimer's disease: structural MRI findings and cognitive correlation patterns in patients under 65.",
      predicted_categories: ["neurological"],
      confidence: 0.92,
      source: "Real medical literature"
    },
    {
      id: 4,
      text: "Acute kidney injury in critically ill patients with sepsis: biomarkers, renal replacement therapy outcomes, and mortality predictors.",
      predicted_categories: ["hepatorenal"],
      confidence: 0.87,
      source: "Real medical literature"
    },
    {
      id: 5,
      text: "Cardiovascular risk assessment in patients with chronic kidney disease: role of coronary calcium scoring and inflammatory markers.",
      predicted_categories: ["cardiovascular", "hepatorenal"],
      confidence: 0.85,
      source: "Real medical literature"
    },
    {
      id: 6,
      text: "Neurological complications in liver transplant recipients: immunosuppressive toxicity and infectious etiology in post-surgical patients.",
      predicted_categories: ["neurological", "hepatorenal"],
      confidence: 0.78,
      source: "Real medical literature"
    }
  ];

  return NextResponse.json(examples);
}
