import { NextResponse } from 'next/server';

export async function GET() {
  // Ejemplos reales extraídos del dataset challenge_data-18-ago.csv
  const examples = {
    examples: [
      {
        title: "Mutaciones BRCA1 en Familias con Cáncer",
        text: "BRCA1 mutations in breast and ovarian cancer families from northeast Brazil: genetic counseling implications and clinical management.",
        expected_categories: ["oncological"]
      },
      {
        title: "Respuesta Hipertensiva Durante Ecocardiografía",
        text: "Hypertensive response during dobutamine stress echocardiography: a diagnostic marker for coronary artery disease in diabetic patients.",
        expected_categories: ["cardiovascular"]
      },
      {
        title: "Cambios Cerebrales en Alzheimer Temprano",
        text: "Brain changes in early-onset Alzheimer's disease: structural MRI findings and cognitive correlation patterns in patients under 65.",
        expected_categories: ["neurological"]
      },
      {
        title: "Lesión Renal Aguda en Sepsis",
        text: "Acute kidney injury in critically ill patients with sepsis: biomarkers, renal replacement therapy outcomes, and mortality predictors.",
        expected_categories: ["hepatorenal"]
      },
      {
        title: "Evaluación de Riesgo Cardiovascular en Enfermedad Renal",
        text: "Cardiovascular risk assessment in patients with chronic kidney disease: role of coronary calcium scoring and inflammatory markers.",
        expected_categories: ["cardiovascular", "hepatorenal"]
      },
      {
        title: "Complicaciones Neurológicas en Trasplante Hepático",
        text: "Neurological complications in liver transplant recipients: immunosuppressive toxicity and infectious etiology in post-surgical patients.",
        expected_categories: ["neurological", "hepatorenal"]
      }
    ]
  };

  return NextResponse.json(examples);
}
