"""检索评测；输入由用户提供，不上传实际数据或测量结果。"""


def recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
    gold = set(relevant)
    if not gold:
        raise ValueError("标准相关论文集合不能为空")
    return len(set(retrieved) & gold) / len(gold)


def candidate_retention(candidates: list[str], retained: list[str], relevant: list[str]) -> float | None:
    if not set(retained) <= set(candidates):
        raise ValueError("保留论文必须属于初始候选")
    gold_in_candidates = set(candidates) & set(relevant)
    return len(set(retained) & gold_in_candidates) / len(gold_in_candidates) if gold_in_candidates else None


def speedup(serial_seconds: float, concurrent_seconds: float) -> float:
    if serial_seconds <= 0 or concurrent_seconds <= 0:
        raise ValueError("两种运行耗时都必须为正数")
    return serial_seconds / concurrent_seconds
