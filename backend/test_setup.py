"""
Quick test script to verify all imports work without errors
"""
import sys

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        print("1. Testing text_splitter...")
        from rag.text_splitter import chunk_text, chunk_text_with_metadata
        print("   [OK] text_splitter imported successfully")
        
        print("2. Testing embeddings...")
        from rag.embeddings import EmbeddingModel
        print("   [OK] embeddings imported successfully")
        
        print("3. Testing embedding model initialization...")
        model = EmbeddingModel()
        print("   [OK] EmbeddingModel initialized successfully")
        
        print("4. Testing embedding generation...")
        test_text = "This is a test document for embedding generation."
        embedding = model.embed_text(test_text)
        print(f"   [OK] Generated embedding with shape: {embedding.shape}")
        
        print("5. Testing text chunking...")
        chunks = chunk_text("This is a test. " * 200, chunk_size=100, chunk_overlap=20)
        print(f"   [OK] Generated {len(chunks)} chunks")
        
        print("\n[SUCCESS] All imports and basic functionality working!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

