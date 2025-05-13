"""
ChromaDB implementation of the Retrieval-Augmented Generation (RAG) pipeline.

This module implements the RAG pattern using ChromaDB as the vector store,
which excels in low-latency scenarios with benchmark latency of 28ms.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from vector_llm_integration.rag.base import RAGPipeline
from vector_llm_integration.observability.monitoring import trace_vectordb_operation

logger = logging.getLogger(__name__)

class ChromaRAGPipeline(RAGPipeline):
    """
    ChromaDB implementation of the RAG pipeline.
    
    This class implements the RAG pipeline using ChromaDB as the vector store,
    which offers excellent performance characteristics for low-latency applications.
    """
    
    def _initialize_components(self) -> None:
        """Initialize vector store, embeddings model, and LLM."""
        # Set up embeddings model
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
        )
        logger.info(f"Initialized embeddings with model {self.embedding_model_name}")
        
        # Set up text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        # Set up LLM
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        logger.info(f"Initialized LLM with model {self.model_name}")
        
        # Initialize empty vector store
        self.vectorstore = None
        self.persist_directory = "./chroma_db"
        logger.info("ChromaDB RAG pipeline initialized")
    
    @trace_vectordb_operation("chroma", "add_documents")
    def add_documents(
        self, 
        documents: Union[str, List[str], Path, List[Path]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Add documents to the ChromaDB vector store.
        
        Args:
            documents: Document(s) to add - can be text strings, file paths,
                      or a list of either
            metadata: Optional metadata for each document
            
        Returns:
            Boolean indicating success
        """
        try:
            # Process input documents into a standardized format
            processed_docs = self._process_documents(documents, metadata)
            
            # Initialize or update vector store
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=processed_docs,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                logger.info(f"Created new ChromaDB with {len(processed_docs)} documents")
            else:
                self.vectorstore.add_documents(processed_docs)
                logger.info(f"Added {len(processed_docs)} documents to existing ChromaDB")
            
            # Persist the vector store
            if hasattr(self.vectorstore, "persist"):
                self.vectorstore.persist()
                logger.info("Persisted ChromaDB to disk")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return False
    
    def _process_documents(
        self, 
        documents: Union[str, List[str], Path, List[Path]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """
        Process raw documents into LangChain Document objects.
        
        Args:
            documents: Document(s) to process
            metadata: Optional metadata for documents
            
        Returns:
            List of processed Document objects
        """
        processed_docs = []
        
        # Handle string or Path input
        if isinstance(documents, (str, Path)):
            documents = [documents]
        
        for i, doc in enumerate(documents):
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            
            if isinstance(doc, str) and os.path.exists(doc):
                # It's a file path as string
                processed_docs.extend(self._load_and_split_file(doc, doc_metadata))
            elif isinstance(doc, Path):
                # It's a Path object
                processed_docs.extend(self._load_and_split_file(str(doc), doc_metadata))
            elif isinstance(doc, str):
                # It's a text string
                doc_metadata = {**doc_metadata, "source": "text_input"}
                text_chunks = self.text_splitter.split_text(doc)
                processed_docs.extend([
                    Document(page_content=chunk, metadata=doc_metadata)
                    for chunk in text_chunks
                ])
            else:
                logger.warning(f"Skipping document of unsupported type: {type(doc)}")
        
        logger.info(f"Processed {len(processed_docs)} document chunks")
        return processed_docs
    
    def _load_and_split_file(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Load and split a file into Document objects.
        
        Args:
            file_path: Path to the file
            metadata: Metadata to attach to documents
            
        Returns:
            List of Document objects
        """
        file_metadata = {**metadata, "source": file_path}
        
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                # Add page numbers to metadata
                for doc in documents:
                    doc.metadata.update(file_metadata)
                return documents
            else:
                # Assume text file for any other extension
                loader = TextLoader(file_path)
                documents = loader.load()
                # Split into chunks
                text_chunks = self.text_splitter.split_documents(documents)
                # Update metadata
                for chunk in text_chunks:
                    chunk.metadata.update(file_metadata)
                return text_chunks
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []
    
    @trace_vectordb_operation("chroma", "get_relevant_documents")
    def get_relevant_documents(
        self, 
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to the query from ChromaDB.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve (overrides instance setting)
            filters: Optional filters to apply
            
        Returns:
            List of documents with text and metadata
        """
        if self.vectorstore is None:
            logger.warning("Attempted to query without initializing the vector store")
            return []
        
        # Set fetch count
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Perform similarity search
            retrieved_docs = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filters
            )
            
            # Format results
            results = []
            for doc in retrieved_docs:
                results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })
            
            logger.info(f"Retrieved {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    @trace_vectordb_operation("chroma", "query")
    def query(
        self,
        query: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Query the ChromaDB RAG system to get a response.
        
        Args:
            query: User query
            temperature: Temperature for generation (overrides instance setting)
            max_tokens: Max tokens to generate (overrides instance setting)
            filters: Optional filters for retrieval
            
        Returns:
            Generated response
        """
        if self.vectorstore is None:
            return "The system has not been initialized with documents yet."
        
        # Override parameters if provided
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Update LLM parameters if needed
        if temp != self.temperature or tokens != self.max_tokens:
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=temp,
                max_tokens=tokens,
            )
        
        try:
            # Get relevant documents
            documents = self.get_relevant_documents(query, filters=filters)
            
            if not documents:
                return "I couldn't find any relevant information to answer your question."
            
            # Format context from retrieved documents
            context = self._format_context(documents)
            
            # Create prompt
            prompt = self._create_prompt(query, context)
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return f"An error occurred while processing your query: {str(e)}"
    
    def build_retrieval_qa_chain(self) -> RetrievalQA:
        """
        Build a RetrievalQA chain for more advanced queries.
        
        Returns:
            RetrievalQA chain
        """
        if self.vectorstore is None:
            raise ValueError("The vector store must be initialized before building a chain")
        
        # Define prompt template
        template = """
        Answer the question based on the following context. If you don't know the answer, just say you don't know.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # Create the retrieval chain
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return chain
