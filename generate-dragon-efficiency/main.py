# Use `x` to avoid `*` usage conflict in markdown
DRAGON_FORMULA: dict[str, str] = {
    "貓": "(1 + 0.4) x (1 + 1.8) x (1.5) x (1 + <PASSIVE_ATK>)",
    "亞森": "(1 + 0.4 + 1.8) x (1.5) x (1 + <PASSIVE_ATK>)",
    "巴哈姆特": "(1 + 0.4) x (1.5) x (1 + 1.2 + <PASSIVE_ATK>)",
    "尼德．勇": "(1 + 0.4) x (1.5 + 0.3) x (1 + 0.7 + <PASSIVE_ATK>)",
}

TITLE: str = "ATK+"

PASSIVE_ATKS: list[float] = [0, 0.13, 0.20, 0.21, 0.28]


def prepare_for_eval(formula: str) -> str:
    return formula.replace("x", "*")


def replace_passive_atk(formula: str, passive_atk: float) -> str:
    return formula.replace("<PASSIVE_ATK>", str(passive_atk))


def main():
    print(f"{TITLE} | {' | '.join(f':{dragon}:' for dragon in DRAGON_FORMULA.keys())}")
    print(" | ".join(':---:' for _ in range(1 + len(DRAGON_FORMULA))))

    efficiency = {
        passive_atk: tuple(
            (dragon, formula, eval(replace_passive_atk(prepare_for_eval(formula), passive_atk)))
            for dragon, formula in DRAGON_FORMULA.items()
        )
        for passive_atk in PASSIVE_ATKS
    }

    max_efficiency_of_atk = {
        passive_atk: max(efficiency[passive_atk], key=lambda item: item[1][2])
        for passive_atk in PASSIVE_ATKS
    }

    for passive_atk in PASSIVE_ATKS:
        max_dragon, max_formula, max_result = max_efficiency_of_atk[passive_atk]

        print(f"+{passive_atk:2.0%}", end="")

        for dragon, formula_og in DRAGON_FORMULA.items():
            if max_dragon == dragon:
                print(f" | -", end="")
                continue

            formula = replace_passive_atk(formula_og, passive_atk)
            max_formula = replace_passive_atk(max_formula, passive_atk)
            print(f" | ==({formula}) / ({max_formula}) - 100%[2f]==", end="")

        print()


if __name__ == '__main__':
    main()
