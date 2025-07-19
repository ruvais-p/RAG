import os
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class RetrievalEvaluator:
    def __init__(self):
        self.queries = []
        self.results = []
        
    def log_query(self, query, results, feedback=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieved_docs": [(doc.metadata.get('source', 'Unknown'), doc.page_content[:100]) 
                             for doc in results],
            "feedback": feedback
        }
        self.queries.append(entry)
    
    def analyze_performance(self):
        if not self.queries:
            print("No queries logged for analysis.")
            return
        
        print("\n=== Retrieval Performance Analysis ===")
        print(f"Total queries processed: {len(self.queries)}")
        
        # Calculate average retrieved docs per query
        avg_docs = sum(len(query['retrieved_docs']) for query in self.queries) / len(self.queries)
        print(f"Average documents retrieved per query: {avg_docs:.1f}")
        
        # Show sample queries
        print("\nSample queries and retrievals:")
        for query in self.queries[:3]:  # Show first 3 as sample
            print(f"\nQuery: {query['query']}")
            for i, (source, content) in enumerate(query['retrieved_docs'], 1):
                print(f"  Doc {i} from {source}: {content}...")

def reciprocal_rank_fusion(results: List[List[Dict]], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1/(rank + k)
    
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return [doc for doc, score in reranked_results]

def initialize_components():
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(current_dir, "data")
    persistant_dir = os.path.join(current_dir, "db", "chroma_db_with_metadata")

    # Initialize embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Create or load Chroma DB
    if not os.path.exists(persistant_dir):
        print("Creating enhanced Chroma DB...")
        
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input directory {input_file_path} does not exist.")

        input_docs = [f for f in os.listdir(input_file_path) if f.endswith(".txt")]
        documents = []
        for input_doc in input_docs:
            file_path = os.path.join(input_file_path, input_doc)
            loader = TextLoader(file_path)
            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata = {
                    "source": input_doc,
                    "filename": os.path.basename(input_doc),
                    "filepath": file_path,
                    "timestamp": datetime.now().isoformat()
                }
                documents.append(doc)

        # Enhanced text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        docs = text_splitter.split_documents(documents)

        print(f"\nProcessed {len(docs)} document chunks from {len(input_docs)} files")

        # Create Chroma DB
        docsearch = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persistant_dir
        )
        
        # Create BM25 retriever
        texts = [doc.page_content for doc in docs]
        bm25_retriever = BM25Retriever.from_texts(texts)
        bm25_retriever.k = 3
        
        print("Database created with hybrid retrieval capabilities")
    else:
        print("Loading existing enhanced Chroma DB...")
        docsearch = Chroma(
            persist_directory=persistant_dir,
            embedding_function=embeddings
        )
        
        # For BM25, we need to recreate from texts (not persisted)
        # In a production system, you'd want to persist/reload BM25 index
        print("Note: BM25 retriever will be recreated on each run")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Updated to newer model
        temperature=0.2,  # Slightly higher for more natural responses
        max_tokens=2000,
        timeout=60
    )

    # Create base retrievers
    vector_retriever = docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.7,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )
    
    # Create hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    # Enhanced query understanding
    query_understanding_prompt = """Analyze this query and extract:
1. Key entities (people, places, things)
2. Action verbs (what information is being sought)
3. Time references (if any)
4. Contextual clues

Return a JSON structure with this analysis.

Query: {input}"""

    query_understanding_chain = (
        ChatPromptTemplate.from_template(query_understanding_prompt)
        | llm 
        | StrOutputParser()
    )

    # Create history-aware retriever
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, perform these steps:
1. Identify key entities and concepts in the question
2. Generate synonyms and related terms for these concepts
3. Formulate 2-3 alternative phrasings of the question
4. Combine these into a comprehensive standalone query

Return ONLY the final enhanced query, nothing else.

Chat History: {chat_history}
Original Question: {input}"""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, hybrid_retriever, contextualize_q_prompt
    )

    # Enhanced QA chain
    qa_system_prompt = """You are an expert document analyst. Use these steps to answer:

1. Verify if the question is answerable from the documents
2. Identify which documents contain relevant information
3. Extract precise information from those documents
4. Synthesize a clear, concise answer
5. Cite your sources by filename when appropriate

If uncertain, ask clarifying questions.

Context:
{context}

Query Analysis:
{query_analysis}

Chat History:
{chat_history}

Question: {input}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return qa_chain, llm, query_understanding_chain, hybrid_retriever

def enhanced_invoke(qa_chain, query_understanding_chain, user_input, chat_history):
    # First understand the query
    query_analysis = query_understanding_chain.invoke({"input": user_input})
    
    # Then retrieve and generate answer
    response = qa_chain.invoke({
        "input": user_input,
        "query_analysis": query_analysis,
        "chat_history": chat_history
    })
    return response

def main():
    qa_chain, llm, query_understanding_chain, retriever = initialize_components()
    chat_history = []
    evaluator = RetrievalEvaluator()

    print("\n=== Enhanced Document Assistant ===")
    print("Type 'quit' to exit or 'debug' to see retrieval details\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! Analyzing session...")
                evaluator.analyze_performance()
                break
                
            if user_input.lower() == 'debug':
                if chat_history:
                    last_query = chat_history[-2].content if len(chat_history) >= 2 else "No recent query"
                    print("\nDebug - Last query:", last_query)
                    print("Retrieving fresh results...")
                    debug_results = retriever.invoke(last_query)
                    evaluator.log_query(last_query, debug_results, "debug")
                    print(f"Retrieved {len(debug_results)} documents:")
                    for i, doc in enumerate(debug_results, 1):
                        print(f"\nDocument {i} from {doc.metadata.get('source', 'Unknown')}:")
                        print(doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""))
                else:
                    print("No chat history available for debugging")
                continue
                
            if not user_input:
                continue
                
            # Process the query
            response = enhanced_invoke(qa_chain, query_understanding_chain, user_input, chat_history)
            answer = response["answer"]
            
            print(f"\nAssistant: {answer}\n")
            
            # Update chat history
            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=answer)
            ])
            
            # Log the retrieval performance
            evaluator.log_query(user_input, response["context"])
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")
            continue

if __name__ == "__main__":
    main()