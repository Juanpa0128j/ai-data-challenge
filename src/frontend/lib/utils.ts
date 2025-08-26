import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

// Get API base URL from environment variable
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "https://ai-data-challenge-9e5b.onrender.com/";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
