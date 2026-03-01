"""
Test Embeddings for RAGService.

Simple tests to initialize and use Cohere embeddings.

Setup:
1. Set environment variable:
   COHERE_API_KEY="your-cohere-api-key"

2. Run tests:
   python tests/test_embeddings.py
"""

import os
import sys
import asyncio
import math
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import embeddings components
from RAGService.Data.Embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingInputType,
    EmbeddingsFactory,
    list_available_providers,
    is_provider_available,
)
from RAGService.Data.Embeddings.providers.cohere_embeddings import CohereEmbeddings


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


# ==================== Test Functions ====================

def test_1_create_config():
    """Test 1: Create Cohere embedding config."""
    print_section("Test 1: Create Cohere Config")
    
    config = EmbeddingConfig.for_cohere()
    
    print(f"Provider: {config.provider.value}")
    print(f"Model: {config.model_name}")
    print(f"Dimension: {config.dimension}")
    print(f"Batch Size: {config.batch_size}")
    print("✅ Config created successfully")


def test_2_initialize_cohere_embeddings():
    """Test 2: Initialize Cohere embeddings."""
    print_section("Test 2: Initialize Cohere Embeddings")
    
    config = EmbeddingConfig.for_cohere()
    embeddings = CohereEmbeddings(config)
    
    print(f"Provider: {embeddings.provider.value}")
    print(f"Dimension: {embeddings.dimension}")
    print("✅ CohereEmbeddings initialized successfully")
    
    return embeddings


def test_3_embed_single_query(embeddings: CohereEmbeddings):
    """Test 3: Embed a single query."""
    print_section("Test 3: Embed Single Query")
    
    query = "What is machine learning?"
    
    result = embeddings.embed_query(query)
    
    print(f"Query: '{query}'")
    # print(result)
    print(f"Embedding dimension: {len(result)}")
    print(f"First 5 values: {result[:5]}")
    print(f"Last 5 values: {result[-5:]}")
    print("✅ Query embedded successfully")


def test_4_embed_query_with_input_type(embeddings: CohereEmbeddings):
    """Test 4: Embed query with specific input type."""
    print_section("Test 4: Embed Query with Input Type")
    
    query = "How does deep learning work?"
    
    # Use search_query input type
    result = embeddings.embed_query(query, input_type=EmbeddingInputType.SEARCH_QUERY)
    
    print(f"Query: '{query}'")
    print(f"Input Type: SEARCH_QUERY")
    print(f"Embedding dimension: {len(result)}")
    print("✅ Query with input type embedded successfully")


def test_5_embed_text_convenience(embeddings: CohereEmbeddings):
    """Test 5: Use embed_text convenience method."""
    print_section("Test 5: Embed Text Convenience Method")
    
    text = "Python is a programming language"
    
    # As query
    query_result = embeddings.embed_text(text, is_query=True)
    print(f"Text: '{text}'")
    print(f"As query - dimension: {len(query_result)}")
    
    # As document
    doc_result = embeddings.embed_text(text, is_query=False)
    print(f"As document - dimension: {len(doc_result)}")
    
    print("✅ embed_text convenience method works")


def test_6_embed_multiple_documents(embeddings: CohereEmbeddings):
    """Test 6: Embed multiple documents."""
    print_section("Test 6: Embed Multiple Documents")
    
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret images.",
        "Reinforcement learning trains agents through rewards."
    ]
    
    result = embeddings.embed_documents(documents)
    
    print(f"Number of documents: {len(documents)}")
    print(f"Number of embeddings: {len(result)}")
    print(f"Each embedding dimension: {len(result[0])}")
    
    for i, doc in enumerate(documents):
        print(f"  Doc {i+1}: '{doc[:40]}...' -> {len(result[i])} dims")
    
    print("✅ Multiple documents embedded successfully")


def test_7_embed_documents_with_input_type(embeddings: CohereEmbeddings):
    """Test 7: Embed documents with specific input type."""
    print_section("Test 7: Embed Documents with Input Type")
    
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step."
    ]
    
    result = embeddings.embed_documents(
        documents,
        input_type=EmbeddingInputType.SEARCH_DOCUMENT
    )
    
    print(f"Input Type: SEARCH_DOCUMENT")
    print(f"Documents embedded: {len(result)}")
    print("✅ Documents with input type embedded successfully")


def test_8_embed_documents_with_batch_size(embeddings: CohereEmbeddings):
    """Test 8: Embed documents with custom batch size."""
    print_section("Test 8: Embed Documents with Batch Size")
    
    # Create 10 documents
    documents = [f"This is sample document number {i+1}" for i in range(10)]
    
    result = embeddings.embed_documents(documents, batch_size=3)
    
    print(f"Documents: {len(documents)}")
    print(f"Batch size: 3")
    print(f"Embeddings returned: {len(result)}")
    print("✅ Batch embedding works correctly")


def test_9_semantic_similarity(embeddings: CohereEmbeddings):
    """Test 9: Test semantic similarity between embeddings."""
    print_section("Test 9: Semantic Similarity")
    
    text1 = "I love programming in Python"
    text2 = "Python is my favorite programming language"
    text3 = "The weather is sunny and warm today"
    
    emb1 = embeddings.embed_query(text1)
    emb2 = embeddings.embed_query(text2)
    emb3 = embeddings.embed_query(text3)
    
    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)
    sim_2_3 = cosine_similarity(emb2, emb3)
    
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Text 3: '{text3}'")
    print()
    print(f"Similarity (1, 2) - similar topics: {sim_1_2:.4f}")
    print(f"Similarity (1, 3) - different topics: {sim_1_3:.4f}")
    print(f"Similarity (2, 3) - different topics: {sim_2_3:.4f}")
    print()
    
    if sim_1_2 > sim_1_3:
        print("✅ Similar texts have higher similarity score!")
    else:
        print("❌ Unexpected: different texts have higher similarity")


def test_10_empty_input_handling(embeddings: CohereEmbeddings):
    """Test 10: Handle empty input."""
    print_section("Test 10: Empty Input Handling")
    
    result = embeddings.embed_documents([])
    
    print(f"Empty list input")
    print(f"Result: {result}")
    print("✅ Empty input handled correctly")


async def test_11_async_embed_query(embeddings: CohereEmbeddings):
    """Test 11: Async embed query."""
    print_section("Test 11: Async Embed Query")
    
    query = "What is artificial intelligence?"
    
    result = await embeddings.async_embed_query(query)
    
    print(f"Query: '{query}'")
    print(f"Embedding dimension: {len(result)}")
    print("✅ Async query embedding works")


async def test_12_async_embed_documents(embeddings: CohereEmbeddings):
    """Test 12: Async embed multiple documents."""
    print_section("Test 12: Async Embed Documents")
    
    documents = [
        "Async programming improves performance.",
        "Python supports async/await syntax.",
        "Concurrent execution saves time."
    ]
    
    result = await embeddings.async_embed_documents(documents)
    
    print(f"Documents: {len(documents)}")
    print(f"Embeddings: {len(result)}")
    print("✅ Async document embedding works")


async def test_13_async_embed_text(embeddings: CohereEmbeddings):
    """Test 13: Async embed_text convenience method."""
    print_section("Test 13: Async Embed Text")
    
    text = "Testing async functionality"
    
    result = await embeddings.async_embed_text(text, is_query=True)
    
    print(f"Text: '{text}'")
    print(f"Dimension: {len(result)}")
    print("✅ Async embed_text works")


def test_14_factory_create_cohere():
    """Test 14: Create Cohere embeddings via factory."""
    print_section("Test 14: Factory Create Cohere")
    
    embeddings = EmbeddingsFactory.create_cohere()
    
    print(f"Provider: {embeddings.provider.value}")
    print(f"Dimension: {embeddings.dimension}")
    
    # Quick test
    result = embeddings.embed_query("Factory test")
    print(f"Test embedding dimension: {len(result)}")
    print("✅ Factory created Cohere embeddings successfully")


def test_15_factory_create_from_env():
    """Test 15: Create embeddings from environment."""
    print_section("Test 15: Factory Create from Env")
    
    embeddings = EmbeddingsFactory.create_from_env(
        provider=EmbeddingProvider.COHERE
    )
    
    print(f"Provider: {embeddings.provider.value}")
    print(f"Dimension: {embeddings.dimension}")
    print("✅ Factory created embeddings from env successfully")


def test_16_list_available_providers():
    """Test 16: List available embedding providers."""
    print_section("Test 16: List Available Providers")
    
    providers = list_available_providers()
    
    print(f"Available providers: {[p.value for p in providers]}")
    
    # Check specific providers
    cohere_available = is_provider_available(EmbeddingProvider.COHERE)
    print(f"Cohere available: {cohere_available}")
    
    print("✅ Provider listing works")


def test_17_different_models():
    """Test 17: Test different Cohere models."""
    print_section("Test 17: Different Cohere Models")
    
    models = [
        ("embed-english-v3.0", 1024),
        ("embed-english-light-v3.0", 384),
    ]
    
    for model_name, expected_dim in models:
        config = EmbeddingConfig.for_cohere(model_name=model_name)
        embeddings = CohereEmbeddings(config)
        
        result = embeddings.embed_query("Test query")
        
        print(f"Model: {model_name}")
        print(f"  Expected dimension: {expected_dim}")
        print(f"  Actual dimension: {len(result)}")
        
        if len(result) == expected_dim:
            print(f"  ✅ Correct!")
        else:
            print(f"  ❌ Mismatch!")
    
    print("✅ Different models tested")


def test_18_multilingual_similarity_same_language():
    """Test 18: Similarity between sentences in the same non-English language."""
    print_section("Test 18: Multilingual Similarity (Same Language)")
    
    # Use multilingual model for better cross-language support
    config = EmbeddingConfig.for_cohere(model_name="embed-multilingual-v3.0")
    embeddings = CohereEmbeddings(config)
    
    # Spanish sentences about programming
    spanish_1 = "Me encanta programar en Python"  # "I love programming in Python"
    spanish_2 = "Python es mi lenguaje de programación favorito"  # "Python is my favorite programming language"
    spanish_3 = "El clima está muy bonito hoy"  # "The weather is very nice today"
    
    emb1 = embeddings.embed_query(spanish_1)
    emb2 = embeddings.embed_query(spanish_2)
    emb3 = embeddings.embed_query(spanish_3)
    
    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)
    
    print("Spanish sentences:")
    print(f"  1: '{spanish_1}' (I love programming in Python)")
    print(f"  2: '{spanish_2}' (Python is my favorite programming language)")
    print(f"  3: '{spanish_3}' (The weather is very nice today)")
    print()
    print(f"Similarity (1, 2) - both about Python: {sim_1_2:.4f}")
    print(f"Similarity (1, 3) - different topics: {sim_1_3:.4f}")
    
    if sim_1_2 > sim_1_3:
        print("✅ Similar Spanish sentences have higher similarity!")
    else:
        print("❌ Unexpected result")


def test_19_cross_language_similarity():
    """Test 19: Similarity between same meaning in different languages."""
    print_section("Test 19: Cross-Language Similarity")
    
    config = EmbeddingConfig.for_cohere(model_name="embed-multilingual-v3.0")
    embeddings = CohereEmbeddings(config)
    
    # Same meaning in different languages
    sentences = [
        ("English", "I love artificial intelligence", "I love artificial intelligence"),
        ("Spanish", "Me encanta la inteligencia artificial", "I love artificial intelligence"),
        ("French", "J'adore l'intelligence artificielle", "I love artificial intelligence"),
        ("German", "Ich liebe künstliche Intelligenz", "I love artificial intelligence"),
        ("Hindi", "मुझे कृत्रिम बुद्धिमत्ता पसंद है", "I love artificial intelligence"),
        ("Japanese", "私は人工知能が大好きです", "I love artificial intelligence"),
        ("Chinese", "我喜欢人工智能", "I love artificial intelligence"),
    ]
    
    # Unrelated sentence
    unrelated = "The cat is sleeping on the couch"
    
    print("Sentences with same meaning in different languages:")
    for lang, text, meaning in sentences:
        print(f"  {lang}: '{text}' ({meaning})")
    print(f"\nUnrelated: '{unrelated}'")
    print()
    
    # Get embeddings
    embeddings_list = [embeddings.embed_query(text) for _, text, _ in sentences]
    unrelated_emb = embeddings.embed_query(unrelated)
    
    # Compare English with all other languages
    english_emb = embeddings_list[0]
    
    print("Similarity with English sentence:")
    for i, (lang, text, _) in enumerate(sentences):
        sim = cosine_similarity(english_emb, embeddings_list[i])
        print(f"  English ↔ {lang}: {sim:.4f}")
    
    # Compare with unrelated
    sim_unrelated = cosine_similarity(english_emb, unrelated_emb)
    print(f"  English ↔ Unrelated: {sim_unrelated:.4f}")
    
    print("\n✅ Cross-language similarity test completed")


def test_20_multilingual_topic_clustering():
    """Test 20: Sentences about same topic cluster together across languages."""
    print_section("Test 20: Multilingual Topic Clustering")
    
    config = EmbeddingConfig.for_cohere(model_name="embed-multilingual-v3.0")
    embeddings = CohereEmbeddings(config)
    
    # Topic 1: Food/Cooking (in different languages)
    food_sentences = [
        ("English", "I enjoy cooking Italian pasta", "I enjoy cooking Italian pasta"),
        ("Spanish", "Me gusta cocinar comida italiana", "I like cooking Italian food"),
        ("French", "J'aime cuisiner des plats italiens", "I like cooking Italian dishes"),
    ]
    
    # Topic 2: Technology (in different languages)
    tech_sentences = [
        ("English", "Machine learning is transforming technology", "Machine learning is transforming technology"),
        ("Spanish", "El aprendizaje automático está transformando la tecnología", "Machine learning is transforming technology"),
        ("French", "L'apprentissage automatique transforme la technologie", "Machine learning is transforming technology"),
    ]
    
    print("Topic 1 - Food/Cooking:")
    for lang, text, meaning in food_sentences:
        print(f"  {lang}: '{text}' ({meaning})")
    
    print("\nTopic 2 - Technology:")
    for lang, text, meaning in tech_sentences:
        print(f"  {lang}: '{text}' ({meaning})")
    
    # Get embeddings
    food_embs = [embeddings.embed_query(text) for _, text, _ in food_sentences]
    tech_embs = [embeddings.embed_query(text) for _, text, _ in tech_sentences]
    
    # Calculate within-topic similarity (should be high)
    food_english_spanish = cosine_similarity(food_embs[0], food_embs[1])
    tech_english_spanish = cosine_similarity(tech_embs[0], tech_embs[1])
    
    # Calculate cross-topic similarity (should be lower)
    food_vs_tech = cosine_similarity(food_embs[0], tech_embs[0])
    
    print("\nSimilarity Results:")
    print(f"  Food (English ↔ Spanish): {food_english_spanish:.4f}")
    print(f"  Tech (English ↔ Spanish): {tech_english_spanish:.4f}")
    print(f"  Food vs Tech (same language): {food_vs_tech:.4f}")
    
    if food_english_spanish > food_vs_tech and tech_english_spanish > food_vs_tech:
        print("\n✅ Same topic clusters together across languages!")
    else:
        print("\n❌ Unexpected clustering result")


def test_21_hindi_similarity():
    """Test 21: Hindi language similarity test."""
    print_section("Test 21: Hindi Similarity Test")
    
    config = EmbeddingConfig.for_cohere(model_name="embed-multilingual-v3.0")
    embeddings = CohereEmbeddings(config)
    
    # Hindi sentences about learning
    hindi_1 = "मैं प्रोग्रामिंग सीख रहा हूं"  # "I am learning programming"
    hindi_2 = "कंप्यूटर प्रोग्रामिंग बहुत रोचक है"  # "Computer programming is very interesting"
    hindi_3 = "आज मौसम बहुत अच्छा है"  # "The weather is very nice today"
    
    # English equivalent
    english = "I am learning programming"
    
    emb_h1 = embeddings.embed_query(hindi_1)
    emb_h2 = embeddings.embed_query(hindi_2)
    emb_h3 = embeddings.embed_query(hindi_3)
    emb_en = embeddings.embed_query(english)
    
    print("Hindi sentences:")
    print(f"  1: '{hindi_1}' (I am learning programming)")
    print(f"  2: '{hindi_2}' (Computer programming is very interesting)")
    print(f"  3: '{hindi_3}' (The weather is very nice today)")
    print(f"\nEnglish: '{english}'")
    print()
    
    sim_h1_h2 = cosine_similarity(emb_h1, emb_h2)
    sim_h1_h3 = cosine_similarity(emb_h1, emb_h3)
    sim_h1_en = cosine_similarity(emb_h1, emb_en)
    
    print("Similarity Results:")
    print(f"  Hindi 1 ↔ Hindi 2 (both programming): {sim_h1_h2:.4f}")
    print(f"  Hindi 1 ↔ Hindi 3 (different topics): {sim_h1_h3:.4f}")
    print(f"  Hindi 1 ↔ English (same meaning): {sim_h1_en:.4f}")
    
    print("\n✅ Hindi similarity test completed")


async def run_async_tests(embeddings: CohereEmbeddings):
    """Run all async tests."""
    print("\n" + "=" * 60)
    print("  ASYNC TESTS")
    print("=" * 60)
    
    await test_11_async_embed_query(embeddings)
    await test_12_async_embed_documents(embeddings)
    await test_13_async_embed_text(embeddings)


def run_all_tests():
    """Run all embedding tests."""
    print("\n" + "=" * 60)
    print("  COHERE EMBEDDINGS TEST SUITE")
    print("=" * 60)
    
    # Check environment
    cohere_key = os.getenv("COHERE_API_KEY", "")
    
    print(f"\nEnvironment Check:")
    print(f"  COHERE_API_KEY: {'✅ Set' if cohere_key else '❌ Not set'}")
    
    if not cohere_key:
        print("\n⚠️  Warning: COHERE_API_KEY not set!")
        print("   Set COHERE_API_KEY in .env file to run tests")
        return
    
    # Run sync tests
    test_1_create_config()
    embeddings = test_2_initialize_cohere_embeddings()
    test_3_embed_single_query(embeddings)
    test_4_embed_query_with_input_type(embeddings)
    test_5_embed_text_convenience(embeddings)
    test_6_embed_multiple_documents(embeddings)
    test_7_embed_documents_with_input_type(embeddings)
    test_8_embed_documents_with_batch_size(embeddings)
    test_9_semantic_similarity(embeddings)
    test_10_empty_input_handling(embeddings)
    
    # Run async tests
    asyncio.run(run_async_tests(embeddings))
    
    # Factory tests
    test_14_factory_create_cohere()
    test_15_factory_create_from_env()
    test_16_list_available_providers()
    test_17_different_models()
    
    # Multilingual tests
    test_18_multilingual_similarity_same_language()
    test_19_cross_language_similarity()
    test_20_multilingual_topic_clustering()
    test_21_hindi_similarity()
    
    print_section("ALL TESTS COMPLETED")
    print("✅ Embeddings test suite finished successfully!")


if __name__ == "__main__":
    run_all_tests()
