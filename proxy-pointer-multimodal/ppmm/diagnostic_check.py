import sys
from pathlib import Path

# Add the parent directory to the path so we can import 'pp'
sys.path.append(str(Path(__file__).resolve().parent))

from pp import config
from pp.openai_client import embed_texts, generate_text

def run_diagnostic():
    print("==================================================")
    print("      PROXY-POINTER MULTIMODAL DIAGNOSTIC        ")
    print("==================================================")

    # 1. Check Configuration Keys
    print("\n[1] Checking Config & Env Variables:")
    print(f"  - OPENAI_API_KEY: {'[SET]' if config.OPENAI_API_KEY else '[MISSING]'}")
    print(f"  - AZURE_OPENAI_API_KEY: {'[SET]' if config.AZURE_OPENAI_API_KEY else '[MISSING]'}")
    print(f"  - AZURE_OPENAI_ENDPOINT: {config.AZURE_OPENAI_ENDPOINT or '[NOT SET]'}")
    print(f"  - AZURE_OPENAI_API_VERSION: {config.AZURE_OPENAI_API_VERSION}")
    print(f"  - GEN_MODEL (LLM): {config.GEN_MODEL}")
    print(f"  - EMBED_MODEL: {config.EMBED_MODEL}")
    print(f"  - AZURE_EMBED_DEPLOYMENT: {config.AZURE_EMBED_DEPLOYMENT or '[NOT SET]'}")
    
    # Embedding endpoint / api key details
    print(f"  - AZURE_OPENAI_EMBEDDING_API_KEY: {'[SET]' if config.AZURE_OPENAI_EMBEDDING_API_KEY else '[MISSING/FALLBACK TO LLM KEY]'}")
    print(f"  - AZURE_OPENAI_EMBEDDING_ENDPOINT: {config.AZURE_OPENAI_EMBEDDING_ENDPOINT or '[NOT SET/FALLBACK TO LLM ENDPOINT]'}")
    
    # Document Intelligence
    has_doc_intel = config.has_document_intelligence()
    print(f"  - DOCUMENTINTELLIGENCE_ENDPOINT: {config.DOCUMENT_INTELLIGENCE_ENDPOINT or '[NOT SET]'}")
    print(f"  - DOCUMENTINTELLIGENCE_API_KEY: {'[SET]' if config.DOCUMENT_INTELLIGENCE_KEY else '[MISSING]'}")
    print(f"  - Document Intelligence Active: {has_doc_intel}")

    # Verify if we can run
    if not (config.OPENAI_API_KEY or config.AZURE_OPENAI_API_KEY):
        print("\nERROR: No OpenAI or Azure OpenAI API key found in your .env file.")
        print("Please check your .env settings.")
        sys.exit(1)

    failures = []

    # 2. Test Embedding
    print("\n[2] Testing Embedding Connection...")
    try:
        test_text = ["Hello world"]
        embeddings = embed_texts(test_text)
        print(f"  ✅ Embedding success! Vector length: {len(embeddings[0])}")
    except Exception as e:
        print(f"  ❌ Embedding failed: {e}")
        failures.append(("Embedding API", e))

    # 3. Test LLM Chat
    print("\n[3] Testing Chat/LLM Connection...")
    try:
        response = generate_text("Respond with the word 'OK'.")
        print(f"  ✅ LLM success! Response: '{response}'")
    except Exception as e:
        print(f"  ❌ Chat failed: {e}")
        failures.append(("LLM/Chat API", e))

    # 4. Test Document Intelligence if endpoint is provided
    if has_doc_intel:
        print("\n[4] Testing Document Intelligence Connection...")
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
            
            client = DocumentIntelligenceClient(
                endpoint=config.DOCUMENT_INTELLIGENCE_ENDPOINT,
                credential=AzureKeyCredential(config.DOCUMENT_INTELLIGENCE_KEY)
            )
            # Make a real network call to list batch results to verify connection and credentials
            list(client.list_analyze_batch_results(model_id="prebuilt-layout"))
            print("  ✅ Document Intelligence connection success!")
        except Exception as e:
            print(f"  ❌ Document Intelligence connection failed: {e}")
            failures.append(("Document Intelligence API", e))

    print("\n==================================================")
    if failures:
        print("          ❌ DIAGNOSTIC FAILED (WITH ERRORS)       ")
        print("==================================================")
        print("\nThe following API connections failed:")
        for name, err in failures:
            print(f"  * {name}: {err}")
        print("\nPlease fix the above connection errors in your .env file.")
        sys.exit(1)
    else:
        print("          ✅ ALL DIAGNOSTICS PASSED SUCCESS!       ")
        print("==================================================")

if __name__ == "__main__":
    run_diagnostic()
