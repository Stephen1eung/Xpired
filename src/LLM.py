import os

from dotenv import load_dotenv
from openai import AzureOpenAI

TEMPERATURE = 1
MAX_TOKENS = 1000

user_prompt = """
Based on the provided list of ingredients generate 1 recipe using only the ingredients on hand — the recipe should be practical, easy to follow, and make the most of the available items.
Suggest recipe that require minimal additional ingredients — for each recipe, specify what extra ingredients are needed and why they improve the dish.

Output format:

Recipe Using Only Available Ingredients:
[Recipe Title]
Ingredients: [List]
Instructions: [Step-by-step guide]
Time Estimate: X minutes
Dietary Restrictions: [List]

Recipe with Additional Ingredients:
[Recipe Title]
Additional Ingredients Needed: [List]
Why These Ingredients Help: [Brief explanation]
Ingredients (Full List): [Available + additional]
Instructions: [Step-by-step guide]
Time Estimate: X minutes
Dietary Restrictions: [List]

Here is the list of ingredients: 
"""


system_prompt = """
You are a culinary assistant specialized in food optimization and waste reduction. Your primary role is to help users make the most of their available ingredients by generating practical, easy-to-follow recipes. 

You must:
Prioritize using ingredients the user already has.
Suggest recipes requiring minimal additional ingredients and explain their importance.
Ensure all recipes are structured clearly, listing ingredients, instructions, and estimated preparation times.
Adapt to various dietary preferences and restrictions when provided.
Suggest substitutions when an ingredient is missing or limited.
Always encourage efficient use of food to help reduce waste.
Input format:

"""

ingredients = ['egg','rice']

def generate_recipe(ingredients):
    try:
        # Get and load config settings
        load_dotenv()

        client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key = os.getenv("AZURE_OPENAI_KEY"),
                api_version = "2024-02-01")

        # Send request to Azure OpenAI model
        response = client.chat.completions.create(
            model = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            temperature = TEMPERATURE,
            max_tokens = MAX_TOKENS,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + str(ingredients)}
            ],
        )
        # Print response to console
        print("Output for: {}\n".format(ingredients) + response.choices[0].message.content + "\n")
        print("Prompt tokens: " + str(response.usage.prompt_tokens)  + "\n" +
              "Completion tokens: " + str(response.usage.completion_tokens)  + "\n" +
              "Total tokens: " + str(response.usage.total_tokens)  + "\n")

    except Exception as ex:
        print("ERROR: FAILED TO ALLOCATE RESOURCES")
        print(ex)

print(generate_recipe(ingredients))
#print(user_prompt + str(ingredients))