from typing import Dict, List


def postprocess_results(result: List[Dict[str, List[Dict[str, str]]]], result_key: str = "generated_text", txt_key: str = "content") -> List[str]:
    return [r[0][result_key][1][txt_key] for r in result]

