"""Code for generating training data.

run:
python -m generate_training_data generate_training_data \
  --total_pages 100 \
  --output_dir ./ \
"""

import os
import re
import time
import json
import tqdm
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import (
    LLMChain,
)
from bing_news import get_news
from api_request_parallel_processor import process_api_requests_from_file
import utils

chat = ChatOpenAI(temperature=0)
prompt_template_for_training_data = (
    open("./prompt_template_for_training_data.txt").read() + "\n"
)
prompt = PromptTemplate(
    input_variables=["title", "description", "published_date", "provider"],
    template=prompt_template_for_training_data,
)
chain = LLMChain(llm=chat, prompt=prompt)


def process_output(output: str):
    """Generates dictionary from raw llm output."""
    # Split string into tasks and outputs using regex
    if re.search(r"Task[\s\d]*:[\s]*Task:", output) is not None:
        task_output_pairs = re.split(r"Task[\s\d]*Task:", output)[1:]
    else:
        task_output_pairs = re.split(
            r"Task[\s\d]*:[\s\d]*|Output[\s\d]*:[\s\d]*|\d\. Task:|\d\. Output:",
            output,
        )[1:]

    # Group tasks and outputs into pairs
    task_output_pairs = list(zip(task_output_pairs[::2], task_output_pairs[1::2]))

    # Create JSON objects for each pair
    json_objects = [
        {"task": pair[0].strip(), "output": pair[1].strip()}
        for pair in task_output_pairs
        if (len(pair[0]) > 0 and len(pair[1]) > 0)
    ]

    return json_objects


async def generate_training_data_requests(output_dir="./"):
    """Generates request body for openai completions API using recent news article data."""
    request_start = time.time()
    news = await get_news(output_dir=output_dir)
    progress_bar = tqdm.tqdm(total=len(news))

    for title, description, published_date, provider in news:
        utils.append_to_jsonl(
            {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt.format(
                            title=title,
                            description=description,
                            published_date=published_date,
                            provider=provider,
                        ),
                    }
                ],
            },
            os.path.join(output_dir, "data_for_request.jsonl"),
        )
        progress_bar.update(1)

    request_duration = time.time() - request_start

    print(f"Process took {request_duration:.2f}s")


async def process_output_data(output_dir: str = "./"):
    """Process raw output data"""
    with open(os.path.join(output_dir, "request_results.jsonl")) as f:
        raw_output = f.readlines()
        for line in raw_output:
            try:
                data = json.loads(line)
                generated_content = data[-1]["choices"][0]["message"]["content"]
                processed_content = process_output(generated_content)
                for content in processed_content:
                    utils.append_to_jsonl(
                        content,
                        os.path.join(output_dir, "processed_output.jsonl"),
                    )
            except:
                print("Wrong format")


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./")
    parser.add_argument("--type", default="generate_requests")
    args = parser.parse_args()

    # run script
    if args.type == "process_output":
        asyncio.run(process_output_data(output_dir=args.output_dir))
    elif args.type == "generate":
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath="./data_for_request.jsonl",
                save_filepath="./request_results.jsonl",
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=os.environ["OPENAI_API_KEY"],
                max_requests_per_minute=1500,
                max_tokens_per_minute=90000,
                token_encoding_name="cl100k_base",
                max_attempts=3,
                logging_level=20,
            )
        )
    else:
        asyncio.run(
            generate_training_data_requests(
                output_dir=args.output_dir,
            )
        )
