# -*- coding: utf-8 -*-
import json
from openai import OpenAI
import re
import time


def get_context(query, documents, embeddings):
    pass
    return None

def remove_control_characters(s):
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', s)

def remove_control_characters(s):
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', s)

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, str):
            return obj.encode('utf-8', 'ignore').decode('utf-8')
        return super().default(obj)

def replace_unicode_spaces(input_str):
    return input_str.replace('\u2009', ' ')
    

def remove_control_characters(s):
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', s)

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, str):
            return obj.encode('utf-8', 'ignore').decode('utf-8')
        return super().default(obj)

def replace_unicode_spaces(input_str):
    # Replace thin spaces with regular spaces
    return input_str.replace('\u2009', ' ')
    


def separate_captions(caption, api, llm):
  if llm=='gpt-4o':

    client = OpenAI(
    api_key=api,
    )
    caption_prompt = f"Please separate the given full caption into the exact subcaptionsand format as a dictionary with keys the letter of each subcaption. If there is no full caption then return an empty dictionary. Do not hallucinate \n{caption}"

    completion = client.chat.completions.create(
      model = 'gpt-4o',
      messages = [
        {'role': 'assistant', 'content': caption_prompt}
      ],
      temperature = 0
    )
    output_string = completion.choices[0].message.content
    print('output_string', output_string)
    pattern = r'```python\s*(.*?)\s*```'
    match = re.search(pattern, output_string, re.DOTALL)

    if match:
        json_str = match.group(1).strip()  # Extract the text inside the code block
        try:
            # Load the string as JSON
            output_dict = json.loads(json_str)
            print("Successfully parsed dictionary:", output_dict)
        except json.JSONDecodeError as e:
            print("JSON decoding failed:", e)
            output_dict = None
    else:
        print("No JSON dictionary found in the string.")
        output_dict = None

  return output_dict


def get_keywords(caption, api, llm):
#   llm = ChatOpenAI(model_name='gpt-4o', openai_api_key=api)
  caption_prompt = f"You are an experienced material scientist. Summarize the text in a less than three keywords separated by comma. The keywords should be a broad and general description of the caption and can be related to the materials used, characterization techniques or any other scientific related keyword. Do not halucinate or create content that does not exist in the provided text: {caption}"
  client = OpenAI(
    api_key=api,
    )
  completion = client.chat.completions.create(
      model = 'gpt-4o',
      messages = [
        {'role': 'assistant', 'content': caption_prompt}
      ],
      temperature = 0
    )
  output_string = completion.choices[0].message.content
  output_string = output_string.strip()
  output_string = output_string.replace("\\", "\\\\")
  output_string = output_string.replace("'", "\"")
  output_string = remove_control_characters(output_string)
  return output_string

  
def safe_summarize_caption(*args, **kwargs):
    """Safely call the get_keywords function with exponential backoff."""
    max_retries = 5
    base_wait_time = 2  # starting with 2 seconds

    for attempt in range(max_retries):
        try:
            # Attempt to call the get_keywords function
            return get_keywords(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:  # if it's not the last attempt
                wait_time = base_wait_time * (2 ** attempt)  # double the wait time with every retry
                print(f"Error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # If we've reached the maximum retries, raise the exception
                print("Max retries reached. Skipping this caption.")
                return None  # or return a default value, or raise the exception
            
def safe_separate_captions(*args, **kwargs):
    """Safely call the get_keywords function with exponential backoff."""
    max_retries = 5
    base_wait_time = 2 

    for attempt in range(max_retries):
        try:
            return separate_captions(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:  
                wait_time = base_wait_time * (2 ** attempt) 
                print(f"Error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Skipping this caption.")
                return None  