import streamlit as st
import json
import os
import time
import pymupdf
import ollama

def extract_structured_data_vision(pdf_path, document_type="invoice"):
    """Converts the first page to an image and sends it to qwen2.5vl for structured extraction."""
    doc = pymupdf.open(pdf_path)
    if len(doc) == 0:
        return {"error": "Uploaded PDF is empty."}
    
    # Render only the first page as a lightweight image (DPI 100 for speed)
    page = doc[0]
    pix = page.get_pixmap(dpi=100)
    temp_image_path = f"temp_page_{int(time.time())}.png"
    pix.save(temp_image_path)
    doc.close()
    
    if document_type == "invoice":
        schema_hint = """
        Extract into this strict JSON format:
        {
          "vendor_name": "",
          "invoice_date": "",
          "invoice_number": "",
          "line_items": [{"description": "", "quantity": 0, "price": 0.0}],
          "total_amount": 0.0
        }
        """
    else:  # Resume / CV
        schema_hint = """
        Extract into this strict JSON format:
        {
          "full_name": "",
          "email": "",
          "phone": "",
          "skills": [],
          "experience_summary": ""
        }
        """

    prompt = f"""
    You are an expert document data extraction engine. Analyze the attached document image 
    and extract the information accurately. 
    CRITICAL: Return ONLY valid raw JSON matching this structure, with no extra markdown blocks or text:
    {schema_hint}
    """

    try:
        response = ollama.chat(
            model='qwen2.5vl',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [temp_image_path]
            }],
            options={'temperature': 0.1}
        )
        
        raw_content = response['message']['content']
        
        # Cleanup temp image immediately
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            
        cleaned_content = raw_content.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_content)

    except Exception as e:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return {"error": str(e), "raw_output": locals().get('raw_content', '')}


# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="AI Document Extractor", page_icon="📄", layout="centered")

st.title("📄 AI Document Data Extraction Pipeline")
st.write("Upload a PDF document, and let local Vision AI extract structured data into clean JSON instantly!")

st.sidebar.header("Configuration")
doc_type = st.sidebar.selectbox(
    "Select Document Type",
    options=["invoice", "resume"],
    format_func=lambda x: "Invoice / Receipt" if x == "invoice" else "Resume / CV"
)

uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])

if uploaded_file is not None:
    temp_pdf_path = f"temp_uploaded_{int(time.time())}.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.info(f"File uploaded successfully: {uploaded_file.name}")
    
    if st.button("Extract Data with Vision AI"):
        with st.spinner("AI is analyzing your document image... Please wait."):
            result_data = extract_structured_data_vision(temp_pdf_path, document_type=doc_type)
            
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            
        st.success("Extraction Completed!")
        
        st.subheader("Extracted Structured Data (JSON)")
        st.json(result_data)
        
        json_string = json.dumps(result_data, indent=4)
        st.download_button(
            label="Download JSON File",
            data=json_string,
            file_name="extracted_data.json",
            mime="application/json"
        )       