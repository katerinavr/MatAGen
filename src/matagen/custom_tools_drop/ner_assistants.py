# A series of LLM assistants for guided and adaptable Named Entity Recognition
import pandas as pd
import numpy as np
import openai
import time
import os
from bs4 import BeautifulSoup


def get_elements(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    elements = []
    for span in soup.find_all('span'):
        class_name = span.get('class')[0] if span.get('class') else 'unknown'
        text = span.text
        elements.append((text, class_name))
    return elements


def identify_related_section(client, file, gpt_model, selected_section):
    """This function reads a scientific paper and extracts only the paragraphs that refer to a specific section."""
    file_contents= []
    answers = ''
    try:
        with open(file, 'r', encoding='utf-8') as f:
            file_contents = f.read()
    except:
            file_contents= file
    print("Start to analyze paper")
    user_heading = f"This is a research paper: {file_contents}."
    user_ending = f"""Your task is read the paper and identify all the paragraphs related to {selected_section}. 
    Your output should be strictly only these paragraphs without modifying their context and without adding any other wording."""

    attempts = 3
    while attempts > 0:
        try:
            response = client.chat.completions.create(
                model= gpt_model, #'gpt-4-turbo-preview',
                temperature = 0,
                messages=[{
                    "role": "system",
                    "content": """Answer the question as truthfully as possible using the provided context."""
                },
                    {"role": "user", "content": user_heading + user_ending}]
            )
            answer_str = response.choices[0].message.content
            # print('gpt-answer', answer_str)
            if not answer_str.lower().startswith("n/a"):
                answers += '\n' + answer_str
            break
        except Exception as e:
            attempts -= 1
            if attempts <= 0:
                print(f"Error: Failed to process paper.")
                break
            print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
            time.sleep(60)
    return answers


def ner_extraction(client, paragraph, gpt_model, user_defined_ner):
    """This LLM assistant will read the paragraph and extract all the user defined named entities."""
    user_heading = f"This is a paragraph from a scientific paper.\n\nContext:\n{paragraph}"
    user_ending = f"""Your task is to perform named entity recognition for materials science related entities. Then generate an HTML version of the input text,
    marking up only the specific materials entities. The named entities that need to be extracted are the following: {user_defined_ner}.
    Use HTML tags to highlight these entities. Return only a short paragraph as an example with the HTML tags for the entities. Each entity should have a class attribute indicating the type of the entity.
    Do not modify the original text.
    """
    attempts = 3
    while attempts > 0:
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                temperature = 0,
                messages=[{
                    "role": "system",
                    "content": """You are a highly intelligent and accurate polymers domain expert.
                    Answer the question as truthfully as possible using the provided context. If you cannot identify the entities return "N/A"."""
                },
                    {"role": "user", "content": user_heading + user_ending}]
            )
            answer_str = response.choices[0].message.content
            break

        except Exception as e:
            attempts -= 1
            if attempts <= 0:
                print(f"Error: Failed to process paper. Skipping. (model 1)")
                break
            print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
            time.sleep(60)
    return answer_str


def NER_assistant(client, gpt_model, uploaded_file, selected_section, user_defined_ner):
    """The adaptive NER assistant. For extractring user defined named entities from certain sections from a scientific paper."""
    
    related_content = identify_related_section(client, uploaded_file, gpt_model, selected_section)
    print('related_content', related_content)
    ner_paragraph = ner_extraction(client, related_content, gpt_model, user_defined_ner) 
    return ner_paragraph


def topic_assistant(client, gpt_model, uploaded_file):
    """LLM assistant to recognize the topic of a paper based on the abstract"""
    
    abstract = identify_related_section(client, uploaded_file, gpt_model, 'Abstract')
    user_intstruct = f"This is the abstract from a research paper. \n\nContext:\n{abstract}"
    user_ending = """Your task is to identify the main materials science topic. Return only a very short and descriptive title as topic.
    """
    attempts = 3
    while attempts > 0:
        try:
            response = client.chat.completions.create(
                model= gpt_model,
                temperature = 0,
                messages=[{
                    "role": "system",
                    "content": """You are a highly intelligent and accurate polymers domain expert.
                    Answer the question as truthfully as possible using the provided context. If you cannot identify a topic return "Topic not avalailable". """
                },
                    {"role": "user", "content": user_intstruct + user_ending}]
            )
            answer_str = response.choices[0].message.content
            break

        except Exception as e:
            attempts -= 1
            if attempts <= 0:
                print(f"Error: Failed to process paper. Skipping. (model 1)")
                break
            print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
            time.sleep(60)
    return answer_str