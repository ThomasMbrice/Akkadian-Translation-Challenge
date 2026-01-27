#!/usr/bin/env python3
"""
Simple test script to verify Singularity container setup.
Tests all critical dependencies for the Akkadian NMT pipeline.
"""

import sys

def test_basic_imports():
    """Test basic Python libraries"""
    print("=" * 60)
    print("Testing basic imports...")
    print("=" * 60)

    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy failed: {e}")
        return False

    try:
        import pandas as pd
        print(f"✓ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas failed: {e}")
        return False

    try:
        import yaml
        print(f"✓ PyYAML: {yaml.__version__}")
    except ImportError as e:
        print(f"✗ PyYAML failed: {e}")
        return False

    print()
    return True


def test_pytorch():
    """Test PyTorch and CUDA"""
    print("=" * 60)
    print("Testing PyTorch...")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")

            # Test simple GPU operation
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            print(f"  ✓ GPU computation test passed")
        else:
            print(f"  ⚠ CUDA not available (CPU mode)")

        print()
        return True
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        print()
        return False


def test_transformers():
    """Test Hugging Face Transformers and ByT5"""
    print("=" * 60)
    print("Testing Transformers...")
    print("=" * 60)

    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")

        # Test ByT5 tokenizer (lightweight, no model download)
        from transformers import AutoTokenizer
        print(f"  Loading ByT5 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

        # Test tokenization
        test_text = "a-na A-šùr-i-mì-tí"
        tokens = tokenizer(test_text)
        print(f"  ✓ Tokenization test passed")
        print(f"    Input: {test_text}")
        print(f"    Tokens: {len(tokens['input_ids'])} tokens")

        print()
        return True
    except Exception as e:
        print(f"✗ Transformers test failed: {e}")
        print()
        return False


def test_faiss():
    """Test FAISS vector search"""
    print("=" * 60)
    print("Testing FAISS...")
    print("=" * 60)

    try:
        import faiss
        print(f"✓ FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'installed'}")

        # Test basic index creation
        import numpy as np
        dimension = 128
        vectors = np.random.random((100, dimension)).astype('float32')

        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        # Test search
        query = np.random.random((1, dimension)).astype('float32')
        distances, indices = index.search(query, k=5)

        print(f"  ✓ Index creation and search passed")
        print(f"    Indexed: 100 vectors")
        print(f"    Retrieved: {len(indices[0])} results")

        # Test GPU FAISS if available
        try:
            import torch
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                gpu_distances, gpu_indices = gpu_index.search(query, k=5)
                print(f"  ✓ GPU FAISS test passed")
        except Exception as e:
            print(f"  ⚠ GPU FAISS not available: {e}")

        print()
        return True
    except Exception as e:
        print(f"✗ FAISS test failed: {e}")
        print()
        return False


def test_sentence_transformers():
    """Test sentence-transformers"""
    print("=" * 60)
    print("Testing Sentence Transformers...")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer
        print(f"✓ Sentence Transformers imported")

        # Don't download model, just verify import works
        print(f"  (Skipping model download in test)")

        print()
        return True
    except Exception as e:
        print(f"✗ Sentence Transformers test failed: {e}")
        print()
        return False


def test_evaluation():
    """Test evaluation metrics"""
    print("=" * 60)
    print("Testing Evaluation Metrics...")
    print("=" * 60)

    try:
        import sacrebleu
        print(f"✓ SacreBLEU: {sacrebleu.__version__}")

        # Test BLEU computation
        refs = [["The cat is on the mat"]]
        hyp = ["The cat sits on the mat"]
        bleu = sacrebleu.corpus_bleu(hyp, refs)
        print(f"  ✓ BLEU computation test passed")
        print(f"    Score: {bleu.score:.2f}")

        print()
        return True
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        print()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("AKKADIAN NMT CONTAINER TEST")
    print("=" * 60)
    print()

    results = []

    results.append(("Basic Imports", test_basic_imports()))
    results.append(("PyTorch", test_pytorch()))
    results.append(("Transformers", test_transformers()))
    results.append(("FAISS", test_faiss()))
    results.append(("Sentence Transformers", test_sentence_transformers()))
    results.append(("Evaluation Metrics", test_evaluation()))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    print()

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("=" * 60)
        print("✓ ALL TESTS PASSED - Container is ready!")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("✗ SOME TESTS FAILED - Check container setup")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
