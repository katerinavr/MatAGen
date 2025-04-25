import chromadb
import json
import os
import uuid
import time 
from chromadb.utils import embedding_functions

def load_image_data(json_filepath):
    """Loads image records from the specified JSON file (expects a 'records' key)."""
    if not os.path.exists(json_filepath):
        print(f"Error: JSON file not found at {json_filepath}")
        return []
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        records = data.get("records", [])
        if not isinstance(records, list):
             print(f"Error: Expected a list under the 'records' key in {json_filepath}, but found {type(records)}.")
             return []
        print(f"Successfully loaded {len(records)} image records from {json_filepath}")
        return records
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}. Check file format.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading JSON: {e}")
        return []

def find_text_files(text_folder_path):
    """Finds all .txt files within the specified directory."""
    txt_files = []
    if not os.path.isdir(text_folder_path):
        print(f"Error: Text folder not found at {text_folder_path}")
        return []
    try:
        for filename in os.listdir(text_folder_path):
            if filename.lower().endswith(".txt"):
                full_path = os.path.join(text_folder_path, filename)
                txt_files.append(full_path)
        return txt_files
    except Exception as e:
        print(f"An error occurred scanning for text files in {text_folder_path}: {e}")
        return []
    finally:
        if txt_files:
            print(f"Finished finding {len(txt_files)} text files.")

def chunk_text(text, filename, chunk_size=1000, chunk_overlap=150):
    """Splits a given text into overlapping chunks with metadata."""
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    chunks = []
    start_index = 0
    chunk_id_counter = 0
    base_filename = os.path.basename(filename)

    while start_index < len(text):
        end_index = start_index + chunk_size
        chunk_content = text[start_index:end_index]
        chunk_unique_id = f"text_{base_filename}_{chunk_id_counter}_{uuid.uuid4()}"
        chunks.append({
            "id": chunk_unique_id,
            "text": chunk_content,
            "metadata": {"source_file": filename, "chunk_index": chunk_id_counter, "type": "text_chunk"}
        })
        start_index += (chunk_size - chunk_overlap)
        chunk_id_counter += 1
        if start_index >= len(text) or chunk_size - chunk_overlap <= 0:
             break
    return chunks

def read_and_chunk_text_files(txt_files):
    """Reads content from text files and applies chunking."""
    all_text_chunks = []
    processed_files = 0
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if content and content.strip():
                file_chunks = chunk_text(content, file_path)
                all_text_chunks.extend(file_chunks)
                processed_files += 1
                if processed_files % 50 == 0: # Print progress occasionally
                     print(f"  Chunked {processed_files}/{len(txt_files)} text files...")

        except Exception as e:
            print(f"Error reading or chunking file {file_path}: {e}")
    if all_text_chunks:
        print(f"Finished chunking {processed_files} text files.")
    return all_text_chunks


def create_multimodal_chromadb(outputs_folder:str, json_file_name:str, 
                               text_folder_name:str, chroma_db_path:str, 
                               collection_name:str, text_embedding_model:str):
    """
    Function that gets txt files, and a json with the image-text descriptions and 
    creates a multimodal ChromaDB for RAG
    """
    print("Starting ChromaDB Population (Text Chunks + Image Captions)")
    start_time = time.time()

    # --- 1. Define Paths ---
    script_dir = os.getcwd() #os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # outputs_dir = os.path.join(script_dir, outputs_folder)
    json_filepath = os.path.join(outputs_folder, json_file_name)
    text_folder_path = os.path.join(outputs_folder, text_folder_name)
    chroma_db_dir = os.path.join(script_dir, chroma_db_path)

    print(f"Resolved paths:")
    print(f"  Outputs folder: {outputs_folder}")
    print(f"  JSON file path: {json_filepath}")
    print(f"  Text folder path: {text_folder_path}")
    print(f"  ChromaDB storage path: {chroma_db_dir}")

    # --- 2. Load Source Data ---
    print("\n--- Loading Source Data ---")
    image_records = load_image_data(json_filepath)
    text_files = find_text_files(text_folder_path)
    text_chunks = read_and_chunk_text_files(text_files) # Text chunks are ready

    if not image_records and not text_chunks:
        print("\nError: No valid data found in JSON 'records' or text files. Cannot proceed. Exiting.")
        exit(1)

    # --- 3. Prepare Image Caption Documents ---
    image_caption_docs = [] 
    if image_records:
        print(f"\n--- Preparing Documents from Image Captions ({len(image_records)} records) ---")
        processed_image_count = 0
        project_root = script_dir

        for i, item in enumerate(image_records):
            caption_text = item.get("caption")
            img_uri_relative = item.get("image")

            if not img_uri_relative or not isinstance(img_uri_relative, str):
                continue
            if not caption_text or not isinstance(caption_text, str):
                 print(f"Warning: Skipping image record {i+1} (URI: {img_uri_relative}) due to missing or invalid '{CAPTION_FIELD_FOR_EMBEDDING}' field.")
                 continue 

            normalized_relative_path = os.path.normpath(img_uri_relative)
            resolved_img_abs_path = os.path.abspath(os.path.join(project_root, normalized_relative_path))

            if not os.path.exists(resolved_img_abs_path):

                 print(f"Warning: Image file path for record {i+1} not found: {resolved_img_abs_path}. Adding caption text anyway.")

            caption_summary_list = item.get("caption_summary", [])
            article_name = item.get("article_name")
            caption_summary_str = ", ".join(caption_summary_list) if isinstance(caption_summary_list, list) else ""

            img_metadata = {
                "source_file": json_filepath,
                "original_uri": img_uri_relative, 
                "resolved_uri_checked": resolved_img_abs_path,
                "type": "image_caption",
                "caption": caption_text, 
                "caption_summary": caption_summary_str,
                "article_name": article_name if article_name else ""
            }

            # create document entry
            image_unique_id = f"imgcap_{i}_{uuid.uuid4()}" 
            image_caption_docs.append({
                "id": image_unique_id,
                "text": caption_text,
                "metadata": img_metadata
            })
            processed_image_count += 1

        print(f"Finished preparing {processed_image_count} documents from image captions.")
    else:
        print("\nNo image records found to process.")

    # 4. initialize ChromaDB client
    print(f"\n--- Initializing ChromaDB ---")
    print(f"Using persistent storage at: {chroma_db_dir}")
    client = chromadb.PersistentClient(path=chroma_db_dir)

    # 5. Setup TEXT Embedding Function
    print(f"\nSetting up TEXT embedding function: {text_embedding_model or 'ChromaDB Default'}")
    try:
        if not text_embedding_model or text_embedding_model.lower() == 'default':
             print("Using ChromaDB default text embedding function.")
             text_ef = None # Pass None to use default
        else:
            text_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=text_embedding_model
            )
        print("Text embedding function ready.")
    except Exception as e:
        print(f"Error loading text embedding function model '{text_embedding_model}': {e}")
        print("Will attempt to use ChromaDB default.")
        text_ef = None 

    # 6. get or create ChromaDB collection
    print(f"\nAccessing collection: '{collection_name}'")
    print(f"DEBUG: Embedding function being used: {text_ef}")

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=text_ef,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection '{collection_name}' ready (using text embeddings).")
    except Exception as e:
        print(f"Error getting or creating ChromaDB collection: {e}")
        exit(1)


    # 7. prepare and Add ALL Text Data to Collection
    print("\n--- Adding Data to ChromaDB Collection ---")

    # combine text chunks and image caption documents
    all_documents_to_add = text_chunks + image_caption_docs

    if all_documents_to_add:
        print(f"\nPreparing {len(all_documents_to_add)} total text documents (chunks + image captions) for batch addition...")
        ids_to_add = [doc["id"] for doc in all_documents_to_add]
        documents_to_add = [doc["text"] for doc in all_documents_to_add]
        metadatas_to_add = [doc["metadata"] for doc in all_documents_to_add]

        # simple batching
        batch_size = 500
        num_batches = (len(ids_to_add) + batch_size - 1) // batch_size
        print(f"Adding data in {num_batches} batches of size {batch_size}...")

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(ids_to_add))
            batch_ids = ids_to_add[start_idx:end_idx]
            batch_docs = documents_to_add[start_idx:end_idx]
            batch_metas = metadatas_to_add[start_idx:end_idx]

            print(f"  Adding batch {i+1}/{num_batches} ({len(batch_ids)} items)...")
            try:

                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs, 
                    metadatas=batch_metas
                )
    
                print(f"  Batch {i+1} upserted successfully.")
            except Exception as e:
                print(f"  Error during batch {i+1} upsert: {e}")

        print(f"\nFinished adding/updating {len(ids_to_add)} text documents.")
    else:
        print("\nNo text chunks or image captions were prepared to add.")


    # 8. final report
    print("\n--- Script Finished ---")
    try:
        final_count = collection.count()
        print(f"Collection '{collection_name}' now contains {final_count} items.")
    except Exception as e:
        print(f"Could not get final count from collection: {e}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"ChromaDB data is stored in: {chroma_db_dir}")

    # # --- Note on Querying ---
    # print("\n--- Querying Note ---")
    # print("To query this data:")
    # print("1. Use collection.query(query_texts=['Your question...'], n_results=N, include=['metadatas', 'documents'])")
    # print("2. The results will contain text chunks and/or image captions.")
    # print("3. To display an image related to a result, check if result metadata['type'] == 'image_caption'.")
    # print("4. If it is, retrieve the image path from result metadata['original_uri'].")
    # print("5. Optionally, pass the retrieved documents to an LLM for generating a final answer.")


if __name__ == "__main__":
    outputs_folder = "pdf_papers"
    json_file_name = "retrieved_image_caption_pairs.json"  # Name of the JSON file inside OUTPUTS_FOLDER
    text_folder_name = "pdf"      # Name of the subfolder with .txt files inside OUTPUTS_FOLDER
    chroma_db_path = "./my_multimodal_chroma_db" # Directory to store the persistent ChromaDB database
    collection_name = "multimodal_rag_collection" # Name for the ChromaDB collection
    text_embedding_model = "all-MiniLM-L6-v2" # Standard good default

    create_multimodal_chromadb(outputs_folder,
                               json_file_name, 
                               text_folder_name, 
                               chroma_db_path, 
                               collection_name, 
                               text_embedding_model)
