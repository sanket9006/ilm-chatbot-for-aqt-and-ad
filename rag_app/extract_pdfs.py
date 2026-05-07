import os
import json
import re
from pypdf import PdfReader
from google import genai
from pydantic import BaseModel
from typing import List, Optional

# Set up Gemini Client
# Use the API key from environment or fallback
api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyDBdW_kv7K4auMHqnMsejCfqMdi4y4X0_w"
client = genai.Client(api_key=api_key)

class FAQ(BaseModel):
    question: str
    answer: str

class Product(BaseModel):
    name: str
    category: str
    description: str
    specs: dict
    faqs: List[FAQ]

def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from: {pdf_path}")
    text = ""
    try:
        reader = PdfReader(pdf_path)
        # Limit to first 20 pages to avoid massive prompts, usually specs/descriptions are at the start/end
        num_pages = min(20, len(reader.pages))
        for i in range(num_pages):
            page = reader.pages[i]
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def process_pdf_with_gemini(text, pdf_name):
    print(f"Processing with Gemini for {pdf_name}...")
    prompt = f"""
    You are an expert product data extractor. I am providing you with the text extracted from a product manual/guide: {pdf_name}.
    
    Extract the following information about the main product described in the manual:
    1. Product Name (e.g. AirDoctor 3500)
    2. Category (always "Air Purifier")
    3. Description (a short 2-3 sentence overview of the product)
    4. Specifications (a dictionary of specs like coverage, CADR, noise_level, dimensions, weight, power, etc. Make keys lowercase with underscores. Include whatever is in the manual).
    5. FAQs: A list of 2-3 common questions and answers derived from the manual (e.g. filter replacement schedule, cleaning).
    
    Output ONLY a pure JSON object matching this structure:
    {{
        "name": "...",
        "category": "Air Purifier",
        "description": "...",
        "specs": {{...}},
        "faqs": [{{"question": "...", "answer": "..."}}]
    }}
    
    Do not include markdown blocks like ```json.
    
    Extracted Text:
    {text[:30000]}  # limit text length to avoid going overboard
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Clean up response text if it has markdown
        out_text = response.text.strip()
        if out_text.startswith("```json"):
            out_text = out_text[7:]
        if out_text.startswith("```"):
            out_text = out_text[3:]
        if out_text.endswith("```"):
            out_text = out_text[:-3]
            
        return json.loads(out_text.strip())
    except Exception as e:
        print(f"Failed to process with Gemini: {e}")
        return None

def main():
    pdf_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pdf")
    data_json_path = os.path.join(os.path.dirname(__file__), "data.json")
    
    # Load existing data
    with open(data_json_path, "r") as f:
        data = json.load(f)
        
    existing_names = {p["name"].lower() for p in data}
    new_products = []
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                print(f"No text extracted from {filename}")
                continue
                
            product_data = process_pdf_with_gemini(text, filename)
            
            if product_data and "name" in product_data:
                print(f"Extracted data for: {product_data['name']}")
                
                # Check if we should update or append
                match_found = False
                for i, existing in enumerate(data):
                    # Simple matching logic
                    if existing["name"].lower() in product_data["name"].lower() or product_data["name"].lower() in existing["name"].lower():
                        print(f"Updating existing product: {existing['name']} with new specs/FAQs")
                        # Update specs
                        data[i]["specs"].update(product_data.get("specs", {}))
                        # Add new FAQs
                        existing_qs = {q["question"].lower() for q in data[i].get("faqs", [])}
                        for faq in product_data.get("faqs", []):
                            if faq["question"].lower() not in existing_qs:
                                data[i]["faqs"].append(faq)
                        match_found = True
                        break
                
                if not match_found:
                    print(f"Adding new product: {product_data['name']}")
                    data.append(product_data)
                    
    # Save back
    with open(data_json_path, "w") as f:
        json.dump(data, f, indent=2)
        
    print("Done updating data.json!")

if __name__ == "__main__":
    main()
