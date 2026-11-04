def ordinal(n: int) -> str:
    if n < 1:
        raise ValueError(f"ordinal(n) was called with n={n}, but requires n>0")
    if n < 4:
        return {1: "first", 2: "second", 3: "third"}[n]
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return f"{n}{suffix}"
