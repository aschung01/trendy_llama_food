"""Code for generating training data.

run:
python -m generate_training_data generate_training_data \
  --total_pages 100 \
  --output_dir ./ \
"""

import os
import re
import time
import tqdm
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import (
    LLMChain,
)
from bing_news import get_news
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
    raw_data_list = re.split("\n\n", output)
    processed_data = []
    for raw_data in raw_data_list:
        splitted_data = re.split(r"(Task|Output)\d: ", raw_data)
        processed_data.append(
            {
                "task": splitted_data[1:][1].strip(),
                "output": splitted_data[1:][3].strip(),
            }
        )

    return processed_data


def generate_training_data(total_pages: int, output_dir="./"):
    """Generates training data set of task-output pairs using recent news article data & llm."""
    # 300 task-output pairs are generated for each page
    page = 0
    training_data = []
    request_start = time.time()

    if os.path.exists(os.path.join(output_dir, "trendy_llama_food.json")):
        training_data = utils.jload(os.path.join(output_dir, "trendy_llama_food.json"))
        print(f"Loaded {len(training_data)} machine-generated tasks")
        page = (len(training_data) // 300) + 1

    progress_bar = tqdm.tqdm(total=total_pages)
    progress_bar.update(page)

    while page < total_pages:
        news = get_news(page)
        raw_generated_data = []

        for title, description, published_date, provider in news:
            raw_generated_data.append(
                chain.run(
                    {
                        "title": title,
                        "description": description,
                        "published_date": published_date,
                        "provider": provider,
                    }
                )
            )

        page += 1
        progress_bar.update(1)

        for raw_data in raw_generated_data:
            training_data.append(process_output(raw_data))

        utils.jdump(training_data, os.path.join(output_dir, "trendy_llama_food.json"))

    request_duration = time.time() - request_start

    print(f"Request took {request_duration:.2f}s")


def generate_training_data_requests(total_pages: int, output_dir="./"):
    """Generates training data set of task-output pairs using recent news article data & llm."""
    # 300 task-output pairs are generated for each page
    page = 0
    request_start = time.time()

    progress_bar = tqdm.tqdm(total=total_pages)
    progress_bar.update(page)

    while page < total_pages:
        news = get_news(page)

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
                os.path.join(output_dir, "data_for_request.json"),
            )

        page += 1
        progress_bar.update(1)

    request_duration = time.time() - request_start

    print(f"Request took {request_duration:.2f}s")


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_pages", default=100)
    parser.add_argument("--output_dir", default="./")
    args = parser.parse_args()

    # run script
    asyncio.run(
        generate_training_data_requests(
            total_pages=int(args.total_pages),
            output_dir=args.output_dir,
        )
    )
