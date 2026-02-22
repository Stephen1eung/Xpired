"""OpenAI recipe generation utilities."""

import os

from dotenv import load_dotenv
from openai import OpenAI

TEMPERATURE = 1
MAX_TOKENS = 1000

USER_PROMPT = """
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

SYSTEM_PROMPT = """
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

INGREDIENTS = ['egg', 'rice']

def generate_recipe(ingredients_list):
    """Generate a recipe using the configured OpenAI chat model."""
    try:
        load_dotenv()
        client = OpenAI(
            base_url=os.getenv("OPENAI_ENDPOINT"),
            api_key=os.getenv("OPENAI_KEY"),
        )

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_DEPLOYMENT_NAME"),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT + str(ingredients_list)},
            ],
        )

        output = response.choices[0].message.content
        usage = response.usage
        print(f"Output for: {ingredients_list}\n{output}\n")
        print(
            f"Prompt tokens: {usage.prompt_tokens}\n"
            f"Completion tokens: {usage.completion_tokens}\n"
            f"Total tokens: {usage.total_tokens}\n"
        )
        return output
    except (OSError, ValueError) as ex:
        print("ERROR: FAILED TO ALLOCATE RESOURCES")
        print(ex)
        return None


if __name__ == "__main__":
    generate_recipe(INGREDIENTS)