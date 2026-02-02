# src/utils.py
def classify_crowd(n_people: int) -> str:
    """
    사람 수를 혼잡도 레벨로 변환
    0~3: LOW, 4~7: MID, 8+: HIGH
    """
    if n_people <= 3:
        return "LOW"
    elif n_people <= 7:
        return "MID"
    else:
        return "HIGH"
