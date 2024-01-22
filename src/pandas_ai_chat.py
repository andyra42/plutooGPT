from pandasai import SmartDataframe
from pandasai.llm import OpenAI
llm = OpenAI(api_token="OPENAI_API_KEY")

# You can instantiate a SmartDataframe with a path to a CSV file
df = SmartDataframe("/usercode/data.csv", config={"llm": llm})

answer = df.chat('Which movie has the best IMDb rating but a shorter runtime?')
print(answer)