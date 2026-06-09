import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "3_rag_pipeline.py"
SPEC = importlib.util.spec_from_file_location("financial_rag_pipeline", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = MODULE.FinancialRAGPipeline.__new__(MODULE.FinancialRAGPipeline)

    def test_bm25_prefers_keyword_match(self) -> None:
        documents = [
            "Microsoft revenue increased in the fiscal year",
            "Apple revenue increased strongly in 2023",
            "Supply chain risks are discussed in the risk factors section",
        ]

        ranked = self.pipeline._bm25_rank("Apple revenue", documents, top_n=2)

        self.assertEqual(ranked[0], documents[1])
        self.assertIn(documents[0], ranked)


if __name__ == "__main__":
    unittest.main()
