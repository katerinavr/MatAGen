import base64
import httpx

image1_url = "https://storage.googleapis.com/polymer_abs_test/Screen%20Shot%202022-11-21%20at%201.32.01%20PM.png"
#"https://storage.googleapis.com/polymer_abs_test/Screen%20Shot%202022-11-21%20at%2011.54.39%20AM.png"
#"https://storage.googleapis.com/polymer_abs_test/Screen%20Shot%202022-11-21%20at%201.32.01%20PM.png"
image1_media_type = "image/png"
image1_data = base64.standard_b64encode(httpx.get(image1_url).content).decode("utf-8")

# image2_url = "https://upload.wikimedia.org/wikipedia/commons/b/b5/Iridescent.green.sweat.bee1.jpg"
# image2_media_type = "image/jpeg"
# image2_data = base64.standard_b64encode(httpx.get(image2_url).content).decode("utf-8")


import anthropic
import os
from anthropic import InternalServerError
import time

def make_api_request(client, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.messages.create(
            model= "claude-3-5-sonnet-20240620",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image1_media_type,
                                "data": image1_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Extract all the points from the plot. Give me the tuples of wavelength and absorbance
                              for each curve. And provide as many points as possible. Return the coordinates in a json format 
                              and assign the correct name to the coordinates as shown in the legend. Follow strictly this example 
                              and do not output any other information.

                              Example:
                              {
                              "pentamer a" : [
                                (300, 0.55),
                                (366, 1.15),
                                (450, 0.18),
                                (522, 1.00),
                                (600, 0.20),
                                (700, 0.02),
                                (800, 0.01),
                                (900, 0.00),
                                (1000, 0.00)
                            ],
                           
                               "pentamer_b" : [
                                (300, 0.55),
                                (385, 1.22),
                                (450, 0.15),
                                (558, 1.00),
                                (650, 0.15),
                                (700, 0.05),
                                (800, 0.02),
                                (900, 0.01),
                                (1000, 0.00)
                            }
                              
                              """
                        }
                    ],
                }
            ],
        )
        except InternalServerError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  #

os.environ["ANTHROPIC_API_KEY"] = "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"
client = anthropic.Anthropic()
message = make_api_request(client)
# message = client.messages.create(
#     model= "claude-3-5-sonnet-20240620",
#     max_tokens=2048,
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "source": {
#                         "type": "base64",
#                         "media_type": image1_media_type,
#                         "data": image1_data,
#                     },
#                 },
#                 {
#                     "type": "text",
#                     "text": "Extract all the points from the plot. Give me the tuples of wavelength and absorbance for each curve. And provide as many points as possible. Return the coordinates in a json format and assign the correct name to the coordinates as shown in the legend."
#                 }
#             ],
#         }
#     ],
# )
print(message)
# Get the response text from the message
response = message.content[0].text
print("response", response)

# # Extract just the JSON data between the first and last curly braces
# import re
# json_str = re.search(r'({[\s\S]*})', response).group(1)

# # Save to file
# import json
# with open('spectral_data.json', 'w') as f:
#     # Parse the string to ensure it's valid JSON then save with formatting
#     data = json.loads(json_str)
#     json.dump(data, f, indent=2)