import json
from typing import List, Dict
from chunking import split_text

def load_products(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_chunks(products: List[Dict]) -> List[Dict]:
    """
    Converts product JSON into a list of chunks with metadata.
    Each chunk is a dict: {"text": "...", "metadata": {"name": "...", "category": "..."}}
    """
    all_chunks = []
    
    for product in products:
        # Create a single string containing all product info for context
        content = f"Product: {product['name']}\n"
        content += f"Category: {product['category']}\n"
        content += f"Description: {product['description']}\n"
        content += f"Specs: {json.dumps(product['specs'])}\n"
        
        for faq in product.get('faqs', []):
            content += f"FAQ: {faq['question']} Answer: {faq['answer']}\n"
            
        # Split the flattened text into chunks
        text_chunks = split_text(content, chunk_size=400, overlap=50)
        
        for text in text_chunks:
            all_chunks.append({
                "text": text,
                "metadata": {
                    "product_name": product['name'],
                    "category": product['category']
                }
            })
            
    return all_chunks

if __name__ == "__main__":
    products = load_products("data.json")
    chunks = prepare_chunks(products)
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print(f"Example chunk: {chunks[0]}")
