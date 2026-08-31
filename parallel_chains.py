"""
Understanding Chains in LangChain v1
LCEL patterns, composition and debugging
"""

from dedent import dedent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnablePassthrough, RunnableLambda
from langsmith import traceable, Client


load_dotenv()


def demo_parallel_chains():
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize in two sentences: {text}")
    sentiment_prompt = ChatPromptTemplate.from_template(
        "What is the sentiment of the followin text? {text}")
    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract 5 keywords in the following text: {text}\nReturn as comma-separated list")

    model = init_chat_model("gpt-5-nano")
    parser = StrOutputParser()

    analysis_chain = RunnableParallel(
        summary=summarize_prompt | model | parser,
        keywords=keywords_prompt | model | parser,
        sentiment=sentiment_prompt | model | parser,
    )

    text = dedent("""
            The new AI features are absolutely incredible! Users love the faster response times and improved accuracy. However,
            some have noted the pricing could be more competitive. Overall, the product launch has been a massive success with
            record breaking adoption rates.""")

    result = analysis_chain.invoke({"text": text})
    print(dedent(f"""
      Analysis results:
        Summary:   {result['summary']}
        Keywords:  {result['keywords']}
        Sentiment: {result['sentiment']}
            """))


@traceable(name="demo_passthrough", run_type="chain")
def demo_passthrough():
    model = init_chat_model('gpt-5-nano')
    prompt = ChatPromptTemplate.from_template(dedent("""
              Original question: {question}
              Context: {context}\n
              Answer the question based on the context."""))

    def fake_retriever(input):
        return "LangChain was created by Harrison Chase in 2022"

    chain = (
        RunnableParallel(context=RunnableLambda(
            fake_retriever), question=RunnablePassthrough())
        | RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]["question"]})
        | prompt
        | model
        | StrOutputParser())

    response = chain.invoke({"question": "Who created LangChain?"})
    print(f"Answer: {response}")


@traceable(name="demo_chain_branching", run_type="chain")
def demo_chain_branching():
    code_prompt = ChatPromptTemplate.from_template(
        "You are a coding expert. Help with: {input}")
    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer: {input}")
    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify this as 'code' or 'general': {input}\nReturn only the classification")

    model = init_chat_model("gpt-5-nano")
    classifier = classifier_prompt | model | StrOutputParser()

    # Branching chain
    def is_code_question(input_dict):
        result = classifier.invoke(input_dict)
        return 'code' in result.lower()

    branch = RunnableBranch(
        (is_code_question, code_prompt | model | StrOutputParser()),
        general_prompt | model | StrOutputParser()  # default branch
    )

    # Test
    questions = [
        "How do I write a for-loop in Python?",
        "What's the weather like today?"
    ]
    for q in questions:
        result = branch.invoke({"input": q})
        print(f"Q: {q}")
        print(f"A: {result[:100]}...")


@traceable(name="demo_debugging", run_type="chain")
def demo_debugging():
    model = init_chat_model("gpt-5-nano")
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    chain = prompt | model | StrOutputParser()

    # Method 1: Get configuration
    print("Chain input schema: ", chain.input_schema.model_json_schema())
    print("Chain output schema: ", chain.output_schema.model_json_schema())

    # Method 2. Use with_config for tracing
    result = chain.with_config(
        run_name="greeting_chain").invoke({"name": "Alice"})
    print(result)

    # Method 3: Inspect intermediate steps
    # Using RunnableLambda for logging
    def log_step(x, step_name):
        print(f"[{step_name}] {type(x).__name__}: {str(x)[:100]}")
        return x

    debug_chain: RunnableLambda[dict[str, str], str] = (
        prompt
        | RunnableLambda[str, str](lambda x: log_step(x, "after_prompt"))
        | model
        | RunnableLambda[str, str](lambda x: log_step(x, "after_model"))
        | StrOutputParser()
    )

    print("Debug chain execution with lambdas:")
    result = debug_chain.invoke({"name": "Debug"})
    print(f"Greeting: {result}")


if __name__ == '__main__':
    # demo_parallel_chains()
    # demo_passthrough()
    # demo_chain_branching()
    demo_debugging()
