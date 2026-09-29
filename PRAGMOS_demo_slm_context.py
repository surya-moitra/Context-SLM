#!/usr/bin/env python3

import os
import getpass
import csv
import time
from transformers import AutoTokenizer, AutoConfig
from PRAGMOS_context_layer_org import ContextLayer

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from prometheus_client import start_http_server
from opentelemetry.exporter.prometheus import PrometheusMetricReader

from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

LOCAL_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LLM_PATH = "./../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf"
CONTEXT_LENGTH = 2048
MAX_RECENT_TURNS = 2  # N (verbatim conversation turns before summarization)
NUM_THREADS = 8
GPU_LAYERS = 40
OTEL_CONSOLE_READER_INTERVAL = 60000
#prompt = "You are a helpful AI assistant understanding Natural Language Query. You need to break down the natural language query text into parts which can be queried with SQL.Your job is to understand the query text in natural language , understand the intent and break it down into attributes . The intent can be QUery, Create, Update or Delete.For example - if the query text is : 'find all SR about faulty modem raised in the last month' ; your job will be to understand that user wants to search about SR or tickets. The intent is Query or Read.Now, the Category on which to Query should be 'Service Request'. Then you should understand that SR description should be 'faulty modem' and time should be within last month. Your final response in JSON should be {\"Intent\" : \"Query\", \"Category\" : \"SR\", \"Description\":\"faulty modem\", \"time\" :\"last month\"}. Let us take another example. Let the next input text be: 'Create an Opportunity with total revenue greater than $10000 but less than $20000 raised in the last quarter in the APAC region'. In this case, your query breakdown response should identify Intent as 'Create' , Category as 'Opportunity' with attributes as total revenue having attribute value as greater than $10000 and less than $20000, then the next attribute should be time with value as last quarter and finally the third attribute should be geographic area with value as APAC. If any attribute is having a range, then the value should with 'greater than' and 'less than'. The response for this example should be {\"Intent\" : \"Create\", \"Category\" : \"Opportunity\", \"Total revenue\":{\"greater than\" : \"$10000\", \"lesser than\" : \"$20000\"}, \"time\" :\"last quarter\", \"geographic area\" : \"APAC\".}. The final response should only contain the Intent, Category Name, Category Value, Attribute name and the Attribute Value. Response Format should be in JSON.No other explanation or text is required in the answer. With the above instructions, please breakdown the below input query text into response JSON only.The categories are usually SR, Tickets, Opportunity, Leads, Quote, Product, Contact, Account and alike. The query text is: Tickets raised in the last month with status as closed and about poor coverage"
#prompt = "Remove all PII like name, phone number, identity information, AADHAR, PAN, Credit card from the below text. Use ** to replcae the PII words. DO not alter the text. Only replace the PII words with **. The text is: Harish Kumar called from Punjab. His phone number was 9887793445. We can reply to harku1990@gmail.com. AADHAR number is 123454327689. His complaint was regarding Balance transfer."
# 
# ----------- OPEN TELEMETRY (OTeL) ---------
start_http_server(port=9464, addr="localhost")  ## start local Prometheus server

console_exporter = ConsoleMetricExporter()
console_reader = PeriodicExportingMetricReader(console_exporter, export_interval_millis=OTEL_CONSOLE_READER_INTERVAL)
#prometheus_reader = PrometheusMetricReader()
try:
    provider = MeterProvider(metric_readers=[console_reader])
    #provider = MeterProvider(metric_readers=[prometheus_reader])
    metrics.set_meter_provider(provider)
except Exception:
    # If already set, just get the existing one
    provider = metrics.get_meter_provider()
meter = metrics.get_meter("pragmos.local.test")
token_counter = meter.create_counter("tokens_used")
token_histogram = meter.create_histogram(
    name="tokens_used", unit = "1", description = "Distribution of Tokens"
)

def record_usage(token_count, agent_id="default"):
        token_counter.add(token_count, {"agent.id": agent_id})
        #token_histogram.record(token_count)

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
config = AutoConfig.from_pretrained(LOCAL_MODEL_ID)
max_window = config.max_position_embeddings 
print(f"Model Max Context: {max_window} tokens")
# ----------- OPEN TELEMETRY (OTeL) end ---------

amnesty_qa = load_dataset("vibrantlabsai/amnesty_qa", "english_v1")
questions = amnesty_qa["eval"]["question"]
ground_truths = amnesty_qa["eval"]["ground_truths"]
print(questions[:5])

# ---------- EXAMPLE RUN ----------

print("Siebel Co-Pilot Ready! Type 'exit' to stop.")
contextLayer = ContextLayer()  # Initialize the context layer
print(f"Session ID: {contextLayer.session_id}")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    user_input_with_context = contextLayer.inject_context(user_input)
    prompt = f"<|user|>\n{user_input_with_context}\n<|assistant|>"
    output = contextLayer.llm(prompt,max_tokens=512, stop=["<|end|>"], echo=False,)
    response = output['choices'][0]['text']

    contextLayer.update_response(response)

    print("Bot:", response)
