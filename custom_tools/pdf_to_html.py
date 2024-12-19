import fitz  # PyMuPDF
import base64
import io

def pdf_to_single_html(pdf_path, output_html_path):
    # Open the PDF document
    pdf_doc = fitz.open(pdf_path)
    html_content = []
    
    for page_num in range(len(pdf_doc)):
        # Load the page
        page = pdf_doc.load_page(page_num)
        
        # Extract the HTML content of the page
        page_html = page.get_text("html")
        
        # Add images in Base64 format to the HTML
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]  # The image reference number in the PDF
            base_image = pdf_doc.extract_image(xref)  # Extract the image
            image_bytes = base_image["image"]  # Get the image bytes
            image_ext = base_image["ext"]  # Get the image extension (e.g., "png", "jpeg")
            
            # Convert the image to Base64 format
            img_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Create an img tag with the Base64 image
            img_tag = f'<img src="data:image/{image_ext};base64,{img_base64}" />'
            
            # Replace the image placeholder in the HTML content with the Base64 image
            page_html = page_html.replace(f"img_{img_index}", img_tag)
        
        # Append the processed HTML of the page to the overall HTML content list
        html_content.append(page_html)

    # Write the full HTML content to the output file
    with open(output_html_path, "w", encoding="utf-8") as html_file:
        html_file.write("\n".join(html_content))


# Example usage
pdf_path = "C:/Users/kvriz/Desktop/pneumatic/pdf_papers/acs.macromol.0c02719/acs.macromol.0c02719.pdf"
output_html_path = "output.html"
pdf_to_single_html(pdf_path, output_html_path)
