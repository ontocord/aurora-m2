from typing import Dict, List


def postprocess_results(result: List[List[Dict[str, str]]], result_key: str = "generated_text") -> List[str]:
    return [r[0][result_key] for r in result]

