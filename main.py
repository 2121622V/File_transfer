import os
import pandas as pd
import yaml, json, asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI
from jinja2 import Template
from datetime import datetime
import difflib
from dotenv import load_dotenv
from dummy_travel_bot import travel_bot_response


load_dotenv()

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview"
)


import difflib

def highlight_diff(original, augmented):
    diff = difflib.ndiff(original.split(), augmented.split())
    html = []
    for token in diff:
        if token.startswith("+ "):
            html.append(f"<span class='added'>{token[2:]}</span>")
        elif token.startswith("- "):
            html.append(f"<span class='removed'>{token[2:]}</span>")
        else:
            html.append(token[2:])
    return " ".join(html)

async def call_llm(prompt):
    r = await client.chat.completions.create(
        model="gpt-5",
        messages=[{"role":"user","content":prompt}],
        response_format={"type":"json_object"},
        temperature=0.2
    )
    return json.loads(r.choices[0].message.content)

async def run_simulation(row, attr, prompts):
    aug_prompt = Template(prompts['augmentation']).render(**row, **attr, severity=3)
    aug_res = await call_llm(aug_prompt)

    actual_ans = travel_bot_response(aug_res['augmented_question'])

    eval_prompt = Template(prompts['evaluation']).render(**row, **aug_res, actual_output=actual_ans)
    eval_res = await call_llm(eval_prompt)

    diff_html = highlight_diff(row['original_question'], aug_res['augmented_question'])

    return {**row, **attr, **aug_res, **eval_res, "actual_output": actual_ans, "diff_html": diff_html}

async def main():
    ontology = yaml.safe_load(open("ontology.yaml"))['ontology']
    prompts = yaml.safe_load(open("prompts.yaml"))['prompts']
    df_input = pd.read_excel("input_cases.xlsx")

    tasks=[]
    for _,row in df_input.iterrows():
        for cat in ontology:
            for attr in cat['attributes']:
                tasks.append(run_simulation(row.to_dict(), attr, prompts))

    results = await asyncio.gather(*tasks)
    df_res=pd.DataFrame(results)

    df_res.to_excel("rai_detailed_results.xlsx",index=False)

    from jinja2 import Environment, FileSystemLoader
    env=Environment(loader=FileSystemLoader('.'))
    template=env.get_template("report.html")

    html=template.render(
        data=df_res.to_dict('records'),
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        flip_rate=round(df_res['decision_flip'].mean()*100,2)
    )

    open("enterprise_full_bias_report.html","w").write(html)

asyncio.run(main())