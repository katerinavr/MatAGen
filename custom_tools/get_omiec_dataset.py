import pandas as pd
import numpy as np
import openai
import time
import os
from bs4 import BeautifulSoup
from openai import OpenAI

# def identify_related_section(client, file, gpt_model):
#     """This function reads a scientific paper and extracts only the paragraphs that refer to a specific section."""
#     file_contents= []
#     answers = ''
#     try:
#         with open(file, 'r', encoding='utf-8') as f:
#             file_contents = f.read()
#     except:
#             file_contents= file
#     print("Start to analyze paper")
#     user_heading = f"This is a research paper: {file_contents}. Answer the questions as truthfully as possible using the provided context."
#     user_ending = f"""Please summarize the following details in a table: polymer name, mobility (μC*), ionic conductivity, electrical conductivity,
#     film thickness, device architecture. If any information is not provided or you are unsure, use "N/A". 
#     The table should have 6 columns, all in lowercase:
#     columns, all in lowercase:| polymer name | mobility (μC*) | ionic conductivity | electrical conductivity | film thickness | device architecture |
#     """

#     attempts = 3
#     while attempts > 0:
#         try:
#             response = client.chat.completions.create(
#                 model= gpt_model, #'gpt-4-turbo-preview',
#                 temperature = 0,
#                 messages=[{
#                     "role": "system",
#                     "content": """Answer the question as truthfully as possible using the provided context."""
#                 },
#                     {"role": "user", "content": user_heading + user_ending}]
#             )
#             answer_str = response.choices[0].message.content
#             print('gpt-answer', answer_str)
#             if not answer_str.lower().startswith("n/a"):
#                 answers += '\n' + answer_str
#             break
#         except Exception as e:
#             attempts -= 1
#             if attempts <= 0:
#                 print(f"Error: Failed to process paper.")
#                 break
#             print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
#             time.sleep(60)
#     return answers


# file_path = "C:/Users/kvriz/Desktop/pneumatic/pneumatic/tools/exsclaim/output/pdf-scraping-omiecs/pdf/"
# file = "C:/Users/kvriz/Desktop/pneumatic/pneumatic/tools/exsclaim/output/pdf-scraping-omiecs/pdf/4T-PEO4.txt"
# OPENAI_API_KEY = "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"
# client = OpenAI(api_key=OPENAI_API_KEY)
# gpt_model= 'gpt-4o'
# identify_related_section(client, file, gpt_model)

# import pandas as pd
# import numpy as np
# import openai
# import time
# import os
# from bs4 import BeautifulSoup
# from openai import OpenAI

# def extract_information(client, file, gpt_model):
#     """This function reads a scientific paper and extracts only the paragraphs that refer to a specific section."""
#     file_contents = []
#     answers = ''
#     try:
#         with open(file, 'r', encoding='utf-8') as f:
#             file_contents = f.read()
#     except:
#         file_contents = file
#     print(f"Start to analyze paper: {file}")
    
#     user_heading = f"This is a research paper: {file_contents}. Answer the questions as truthfully as possible using the provided context."
#     user_ending = f"""Please summarize the following details in a table: polymer name, mobility (μC*), ionic conductivity, electrical conductivity,
#     film thickness, device architecture. If any information is not provided or you are unsure, use "N/A". 
#     The table should have 6 columns, all in lowercase:
#     | polymer name | mobility (μC*) | ionic conductivity | electrical conductivity | film thickness | device architecture |
#     """

#     attempts = 3
#     while attempts > 0:
#         try:
#             response = client.chat.completions.create(
#                 model=gpt_model,
#                 temperature=0,
#                 messages=[{
#                     "role": "system",
#                     "content": """Answer the question as truthfully as possible using the provided context."""
#                 },
#                     {"role": "user", "content": user_heading + user_ending}]
#             )
#             answer_str = response.choices[0].message.content
#             # print('gpt-answer', answer_str)
#             if not answer_str.lower().startswith("n/a"):
#                 answers += '\n' + answer_str
#             break
#         except Exception as e:
#             attempts -= 1
#             if attempts <= 0:
#                 print(f"Error: Failed to process paper {file}.")
#                 break
#             print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
#             time.sleep(60)
#     return answers


# def scrape_and_create_csv(client, directory, gpt_model, output_csv):
#     """Scrapes data from all text files in the specified directory and creates a CSV file."""
#     data = []

#     for i, filename in enumerate(os.listdir(directory)):
#         if i ==2:
#             break
#         if filename.endswith('.txt'):
#             file_path = os.path.join(directory, filename)
#             result = extract_information(client, file_path, gpt_model)
#             print('result', result)
#             if result:
#                 # Skip the first two lines which are table description and headers
#                 rows = result.strip().split('\n')[2:]  
#                 for row in rows:
#                     columns = row.split('|')
#                     print('columns', columns)
#                     # Ensure correct number of columns (ignoring empty spaces on the sides)
#                     if len(columns) == 6:  
#                         data.append({
#                             'polymer name': columns[1].strip(),
#                             'mobility (μC*)': columns[2].strip(),
#                             'ionic conductivity': columns[3].strip(),
#                             'electrical conductivity': columns[4].strip(),
#                             'film thickness': columns[5].strip(),
#                             'device architecture': columns[6].strip(),
#                         })

#     # Create a DataFrame from the collected data
#     df = pd.DataFrame(data)
#     print('df', df)
#     # Save the DataFrame to a CSV file
#     df.to_csv(output_csv, index=False)

#     print(f"Data successfully saved to {output_csv}")




import pandas as pd
import numpy as np
import openai
import time
import os

def extract_information(client, file, gpt_model):
    """This function reads a scientific paper and extracts only the paragraphs that refer to a specific section."""
    file_contents = ''
    try:
        with open(file, 'r', encoding='utf-8') as f:
            file_contents = f.read()
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        return None
    print(f"Start to analyze paper: {file}")
    
    user_heading = f"This is a research paper: {file_contents}. Answer the questions as truthfully as possible using the provided context."
    user_ending = f"""Please summarize the following details in a table: polymer name, mobility (μC*), ionic conductivity, electrical conductivity,
film thickness, device architecture, following the instructions:
* Examples like p(g2T-T) and g2T-T are equivalent. The p() means polymer so you can convert all of them to the monomer name, so  p(g2T-T) can be recorder as g2T-T and so on
* The output table should have 16 columns, all in lowercase:
| polymer name | molecular weight (MW) distribution | polydispertity | mobility | ionic conductivity | electrical conductivity | film thickness | device architecture | processing atmosphere | electrolyte | processing solvent | year | chemical formula |  homo-lumo energy | tg | Stability/lifetime | 
* If an entry has multiple reference of mobility values create separate entries to record all these values where the polymer name is the same
* The mobility units are : F cm-1 V-1 s-1. you do not need to have the units in the csv, only the number
* The available options for the processing atmosphere are 'air', 'n2 glovebox', 'n2 purged'
* The available device architecture are 'vertical', 'planar', 'internal gate'
* If any information is not provided or you are unsure, use "N/A". 

"""

    attempts = 3
    while attempts > 0:
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": """Answer the question as truthfully as possible using the provided context."""
                    },
                    {"role": "user", "content": user_heading + user_ending}
                ]
            )
            answer_str = response.choices[0].message.content
            # print('gpt-answer', answer_str)
            if not answer_str.lower().startswith("n/a"):
                return answer_str
            else:
                print(f"No relevant data found in {file}.")
                return None
        except Exception as e:
            attempts -= 1
            if attempts <= 0:
                print(f"Error: Failed to process paper {file}.")
                break
            print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining.")
            time.sleep(60)
    return None


def fix_encoding_errors(text):
    replacements = {
        'âˆ’': '−',
        'â€™': "'",
        'â€œ': '"',
        'â€�': '"',
        # Add more replacements as needed
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text

def scrape_and_create_csv(client, directory, gpt_model, output_csv):
    """Scrapes data from all text files in the specified directory and creates a CSV file."""
    data = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            result = extract_information(client, file_path, gpt_model)
            if result:
                # Fix encoding errors in the result
                result = fix_encoding_errors(result)
                # Split the result into lines and find the start of the table
                lines = result.strip().split('\n')
                # Skip lines until the table header is found
                for i, line in enumerate(lines):
                    if '|' in line and 'polymer name' in line.lower():
                        table_lines = lines[i+1:]  # Lines after the header
                        break
                else:
                    print(f"No table found in the response for {filename}.")
                    continue  # Skip to the next file if no table is found

                # Process each row in the table
                for row in table_lines:
                    if not row.strip() or not '|' in row:
                        continue  # Skip empty lines
                    columns = [col.strip() for col in row.strip().split('|')]
                    # Remove empty strings from split (due to leading/trailing '|')
                    columns = [col for col in columns if col]
                    # Ensure correct number of columns
                    if len(columns) == 16:
                        columns = [fix_encoding_errors(col) for col in columns]
                        data.append({
                            'polymer name': columns[0],
                            'molecular weight (MW) distribution': columns[1],
                            'polydispertity': columns[2],
                            'mobility (μC*, F cm-1 V-1 s-1)': columns[3],
                            'ionic conductivity': columns[4],
                            'electrical conductivity': columns[5],
                            'film thickness': columns[6],
                            'device architecture': columns[7],
                            'processing atmosphere': columns[8],
                            'electrolyte': columns[9],
                            'processing solvent': columns[10],
                            'year': columns[11],
                            'chemical formula': columns[12],
                            'homo-lumo energy' : columns[13], 
                            'tg' : columns[14],
                            'Stability/lifetime': columns[15],
                            'file name': filename
                        })
                    else:
                        print(f"Unexpected number of columns in row: {row}")

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    if df.empty:
        print("No data collected to save.")
        return
    print('DataFrame:', df)
    # Save the DataFrame to a JSON file
    df.to_json('output_data_reviews_new1.json', orient='records', lines=True, force_ascii=False)
    # Save the DataFrame to a CSV file with UTF-8 encoding
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Data successfully saved to 'output_data.json' and '{output_csv}'")

# def scrape_and_create_csv(client, directory, gpt_model, output_csv):
#     """Scrapes data from all text files in the specified directory and creates a CSV file."""
#     data = []

#     for i, filename in enumerate(os.listdir(directory)):
#         #     for i, filename in enumerate(os.listdir(directory)):
#         # if i ==2:
#         #      break
#         if filename.endswith('.txt'):
#             file_path = os.path.join(directory, filename)
#             result = extract_information(client, file_path, gpt_model)
#             if result:
#                 # Split the result into lines and find the start of the table
#                 lines = result.strip().split('\n')
#                 # Skip lines until the table header is found
#                 for i, line in enumerate(lines):
#                     if '|' in line and 'polymer name' in line.lower():
#                         table_lines = lines[i+1:]  # Lines after the header
#                         break
#                 else:
#                     print(f"No table found in the response for {filename}.")
#                     continue  # Skip to the next file if no table is found

#                 # Process each row in the table
#                 for row in table_lines:
#                     if not row.strip() or not '|' in row:
#                         continue  # Skip empty lines
#                     columns = [col.strip() for col in row.strip().split('|')]
#                     # Remove empty strings from split (due to leading/trailing '|')
#                     columns = [col for col in columns if col]
#                     # Ensure correct number of columns
#                     if len(columns) == 6:
#                         data.append({
#                             'polymer name': columns[0],
#                             'mobility (μC*)': columns[1],
#                             'ionic conductivity': columns[2],
#                             'electrical conductivity': columns[3],
#                             'film thickness': columns[4],
#                             'device architecture': columns[5],
#                         })
#                     else:
#                         print(f"Unexpected number of columns in row: {row}")

#     # Create a DataFrame from the collected data
#     df = pd.DataFrame(data)
#     if df.empty:
#         print("No data collected to save.")
#         return
#     print('DataFrame:', df)
#     # Save the DataFrame to a JSON file
#     df.to_json('output_data.json', orient='records', lines=True)
#     # Save the DataFrame to a CSV file
#     df.to_csv(output_csv, index=False)
#     print(f"Data successfully saved to 'output_data.json' and '{output_csv}'")

# Define paths and API key
# file_directory = "C:/Users/kvriz/Desktop/pneumatic/pneumatic/tools/exsclaim/output/omiecs_reviews/pdf/"

file_directory = "C:/Users/kvriz/Desktop/pneumatic/pneumatic/tools/exsclaim/output/pdf-scraping-omiecs/pdf"
OPENAI_API_KEY =  "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"
# Define paths and API key

client= OpenAI(api_key=OPENAI_API_KEY)
gpt_model = 'gpt-4o'
output_csv_path = "output_data_reviews_new1.csv"

# Run the scraping function
scrape_and_create_csv(client, file_directory, gpt_model, output_csv_path)



