"""Одновременное решение прямой и двойственной задач ЛП (схема из лекции).

Класс задач: f = c·x -> max,  Ax <= b,  b >= 0,  x >= 0.

1) Построить двойственную задачу: g = b·y -> min,  A^T y >= c,  y >= 0.
2) Решить прямую симплекс-методом: привести к каноническому виду (добавить
   балансовые переменные), идти по таблицам с заменой переменных.
3) По конечной таблице найти решение двойственной двумя способами:
   способ 1: Δ_j = (y*, A^j) - c_j  (берём единичные векторы балансовых переменных);
   способ 2: y* = C_B · B^{-1}.

Расчёты ведутся в дробях (Fraction), поэтому результат точный. Номера переменных
в steps — как на бумаге: x1, x2, ... (с единицы).
"""

from fractions import Fraction

from src.core.problem import LinearTask, PrimalDualResult, SolveResult

Matrix = list[list[Fraction]]


def _fr(x: float) -> Fraction:
    return Fraction(str(x))


def _f(x: Fraction) -> float:
    return float(x)


def _fmt(x: float | Fraction) -> str:
    return str(Fraction(x).limit_denominator(10**6))


def _name(j: int) -> str:
    return f"x{j + 1}"


# ---------------------------------------------------------------------------
# Проверка класса задачи
# ---------------------------------------------------------------------------
def check_task(task: LinearTask) -> None:
    A, b, c = task["constraint_matrix"], task["right_sides"], task["objective_coefficients"]
    m, n = len(A), len(c)
    if m == 0 or n == 0:
        raise ValueError("Задача пуста.")
    if any(len(row) != n for row in A) or len(b) != m or len(task["constraint_signs"]) != m:
        raise ValueError("Размеры c, A, знаков и b не согласованы.")
    if not task["maximize"] or any(s != "<=" for s in task["constraint_signs"]) or any(v < 0 for v in b):
        raise ValueError("Поддерживается задача f = c·x -> max, Ax <= b, b >= 0, x >= 0 (как в лекции).")


# ---------------------------------------------------------------------------
# Шаг 1. Двойственная задача
# ---------------------------------------------------------------------------
def build_dual(task: LinearTask) -> LinearTask:
    """c <-> b, A -> A^T, знак «<=» -> «>=», max -> min, y >= 0."""
    A = task["constraint_matrix"]
    m, n = len(A), len(task["objective_coefficients"])
    return {
        "objective_coefficients": task["right_sides"][:],
        "constraint_matrix": [[A[i][j] for i in range(m)] for j in range(n)],
        "constraint_signs": [">="] * n,
        "right_sides": task["objective_coefficients"][:],
        "maximize": False,
    }


def _linear(coefs: list[float], var: str) -> str:
    text = " + ".join(f"{_fmt(c)}·{var}{j + 1}" for j, c in enumerate(coefs))
    return text.replace("+ -", "- ")


def _dual_step(dual: LinearTask) -> dict[str, object]:
    return {
        "step": "build_dual",
        "dual_task": dual,
        "text": [f"g(y) = {_linear(dual['objective_coefficients'], 'y')} -> min"]
        + [f"{_linear(row, 'y')} >= {_fmt(rhs)}" for row, rhs in zip(dual["constraint_matrix"], dual["right_sides"])]
        + ["y >= 0"],
    }


# ---------------------------------------------------------------------------
# Шаг 2. Канонический вид и симплекс-таблицы с заменой переменных
# ---------------------------------------------------------------------------
def to_canonical(task: LinearTask) -> tuple[Matrix, list[Fraction], list[Fraction]]:
    """Ax + E·s = b: к каждой строке добавляем свою балансовую переменную."""
    A, b, c = task["constraint_matrix"], task["right_sides"], task["objective_coefficients"]
    m = len(A)
    A_c = [[_fr(v) for v in A[i]] + [Fraction(int(i == k)) for k in range(m)] for i in range(m)]
    c_c = [_fr(v) for v in c] + [Fraction(0)] * m
    return A_c, [_fr(v) for v in b], c_c


class Table:
    """Строки — базисные переменные, столбцы — небазисные; x_базис = b - Σ a·x_небазис.
    Последняя строка — оценки Δ_j = Σ c_B·a_j - c_j, в углу — значение f."""

    def __init__(self, A_c: Matrix, b: list[Fraction], c_c: list[Fraction], n: int, m: int):
        self.basis = list(range(n, n + m))  # балансовые переменные
        self.free = list(range(n))  # исходные переменные
        self.rows = [A_c[i][:n] + [b[i]] for i in range(m)]
        self.delta = [-c_c[j] for j in range(n)] + [Fraction(0)]

    def entering(self) -> int | None:
        """Столбец с Δ < 0; из них — переменная с наименьшим номером (правило Бланда)."""
        cands = [k for k, d in enumerate(self.delta[:-1]) if d < 0]
        return min(cands, key=lambda k: self.free[k]) if cands else None

    def leaving(self, k: int) -> int | None:
        """Строка с min b_i/a_ik по a_ik > 0; при равенстве — меньший номер переменной."""
        cands = [i for i, row in enumerate(self.rows) if row[k] > 0]
        if not cands:
            return None
        return min(cands, key=lambda i: (self.rows[i][-1] / self.rows[i][k], self.basis[i]))

    def pivot(self, r: int, k: int) -> None:
        p = self.rows[r][k]
        prow = [v / p for v in self.rows[r]]
        prow[k] = 1 / p
        new_rows = []
        for i, row in enumerate(self.rows):
            if i == r:
                new_rows.append(prow)
                continue
            a = row[k]
            new = [row[j] - a * prow[j] for j in range(len(row))]
            new[k] = -a / p
            new_rows.append(new)
        a = self.delta[k]
        new_delta = [self.delta[j] - a * prow[j] for j in range(len(self.delta))]
        new_delta[k] = -a / p
        self.rows, self.delta = new_rows, new_delta
        self.basis[r], self.free[k] = self.free[k], self.basis[r]

    def snapshot(self, iteration: int, pivot: dict[str, object] | None) -> dict[str, object]:
        return {
            "step": "simplex_table",
            "iteration": iteration,
            "free": [_name(j) for j in self.free],
            "basis": [_name(j) for j in self.basis],
            "rows": [[_f(v) for v in row] for row in self.rows],
            "delta": [_f(v) for v in self.delta],
            "pivot": pivot,
        }


def solve_simplex(task: LinearTask, max_iterations: int):
    A_c, b, c_c = to_canonical(task)
    m, n = len(task["constraint_matrix"]), len(task["objective_coefficients"])
    t = Table(A_c, b, c_c, n, m)
    steps: list[dict[str, object]] = [{
        "step": "canonical_form",
        "constraint_matrix": [[_f(v) for v in row] for row in A_c],
        "right_sides": [_f(v) for v in b],
        "objective_coefficients": [_f(v) for v in c_c],
    }]
    for iteration in range(max_iterations):
        k = t.entering()
        if k is None:
            steps.append(t.snapshot(iteration, None))
            return t, steps, "optimal", A_c, c_c
        r = t.leaving(k)
        if r is None:
            steps.append(t.snapshot(iteration, {"enter": _name(t.free[k]), "leave": None}))
            return t, steps, "unbounded", A_c, c_c
        steps.append(t.snapshot(iteration, {
            "enter": _name(t.free[k]), "leave": _name(t.basis[r]),
            "row": r, "col": k, "element": _f(t.rows[r][k]),
        }))
        t.pivot(r, k)
    return t, steps, "iteration_limit", A_c, c_c


# ---------------------------------------------------------------------------
def dual_by_estimates(t: Table, A_c: Matrix, c_c: list[Fraction], n: int, m: int):
    est = [Fraction(0)] * (n + m)
    for k, j in enumerate(t.free):
        est[j] = t.delta[k]
    y = [est[n + i] for i in range(m)]
    ok = all(sum(y[i] * A_c[i][j] for i in range(m)) - c_c[j] == est[j] for j in range(n + m))
    step = {"step": "dual_by_estimates",
            "estimates": {_name(j): _f(v) for j, v in enumerate(est)},
            "y": [_f(v) for v in y], "all_equations_hold": ok}
    return y, step


# ---------------------------------------------------------------------------
# Шаг 3б. Способ 2: y* = C_B · B^{-1}
# ---------------------------------------------------------------------------
def inverse(M: Matrix) -> Matrix:
    """Обратная матрица по схеме [B | E] -> [E | B^{-1}]."""
    m = len(M)
    a = [row[:] + [Fraction(int(i == j)) for j in range(m)] for i, row in enumerate(M)]
    for col in range(m):
        piv = next(r for r in range(col, m) if a[r][col] != 0)
        a[col], a[piv] = a[piv], a[col]
        p = a[col][col]
        a[col] = [v / p for v in a[col]]
        for r in range(m):
            if r != col and a[r][col] != 0:
                f = a[r][col]
                a[r] = [x - f * z for x, z in zip(a[r], a[col])]
    return [row[m:] for row in a]


def dual_by_inverse(t: Table, A_c: Matrix, c_c: list[Fraction], m: int):
    cols = sorted(t.basis)
    B = [[A_c[i][j] for j in cols] for i in range(m)]
    c_B = [c_c[j] for j in cols]
    B_inv = inverse(B)
    y = [sum(c_B[i] * B_inv[i][k] for i in range(m)) for k in range(m)]
    step = {"step": "dual_by_inverse", "basis_vars": [_name(j) for j in cols],
            "B": [[_f(v) for v in row] for row in B],
            "B_inv": [[_f(v) for v in row] for row in B_inv],
            "c_B": [_f(v) for v in c_B], "y": [_f(v) for v in y]}
    return y, step


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------
def solve_primal_dual(task: LinearTask, max_iterations: int = 1000) -> PrimalDualResult:
    check_task(task)
    n, m = len(task["objective_coefficients"]), len(task["constraint_matrix"])

    dual = build_dual(task)
    dual_steps: list[dict[str, object]] = [_dual_step(dual)]

    t, primal_steps, status, A_c, c_c = solve_simplex(task, max_iterations)

    if status != "optimal":
        if status == "unbounded":
            msg = ("Целевая функция не ограничена сверху; по теореме двойственности "
                   "у двойственной задачи нет допустимых планов.")
            dual_status = "infeasible"
        else:
            msg = f"Достигнут лимит итераций ({max_iterations})."
            dual_status = "iteration_limit"
        return {
            "primal_result": {"status": status, "plan": None, "objective_value": None,
                              "steps": primal_steps, "message": msg},
            "dual_result": {"status": dual_status, "plan": None, "objective_value": None,
                            "steps": dual_steps, "message": msg},
            "same_optimum": None, "message": msg,
        }

    x = [Fraction(0)] * n
    for i, j in enumerate(t.basis):
        if j < n:
            x[j] = t.rows[i][-1]
    f = t.delta[-1]

    y1, step1 = dual_by_estimates(t, A_c, c_c, n, m)
    y2, step2 = dual_by_inverse(t, A_c, c_c, m)
    g = sum(_fr(bi) * yi for bi, yi in zip(task["right_sides"], y2))
    step1["agrees_with_inverse_method"] = y1 == y2
    dual_steps += [step1, step2]

    primal: SolveResult = {
        "status": "optimal",
        "plan": [_f(v) for v in x],
        "objective_value": _f(f),
        "steps": primal_steps,
        "message": None,
    }
    dual_res: SolveResult = {
        "status": "optimal",
        "plan": [_f(v) for v in y2],
        "objective_value": _f(g),
        "steps": dual_steps,
        "message": None,
    }
    return {
        "primal_result": primal,
        "dual_result": dual_res,
        "same_optimum": f == g,
        "message": f"f* = g* = {f}" if f == g else f"f* = {f}, g* = {g}",
    }


def format_table(step: dict[str, object]) -> str:
    free = step["free"]
    basis = step["basis"]
    lines = ["\t".join(["", *free, "b"])]
    for name, row in zip(basis, step["rows"]):
        lines.append("\t".join([name, *(_fmt(v) for v in row)]))
    lines.append("\t".join(["Δ", *(_fmt(v) for v in step["delta"]) ]))
    return "\n".join(lines)


if __name__ == "__main__":
    example: LinearTask = {
        "objective_coefficients": [1, 3],
        "constraint_matrix": [[-1, 2], [1, 2], [1, -1]],
        "constraint_signs": ["<=", "<=", "<="],
        "right_sides": [2, 8, 3],
        "maximize": True,
    }
    res = solve_primal_dual(example)
    for s in res["dual_result"]["steps"][:1]:
        print("\n".join(s["text"]))
    for s in res["primal_result"]["steps"]:
        if s["step"] == "simplex_table":
            print(f"\nТаблица {s['iteration']}  разрешающий: {s['pivot']}")
            print(format_table(s))
    print("\nx* =", [_fmt(v) for v in res["primal_result"]["plan"]], " f* =", _fmt(res["primal_result"]["objective_value"]))
    print("y* =", [_fmt(v) for v in res["dual_result"]["plan"]], " g* =", _fmt(res["dual_result"]["objective_value"]))
    print(res["same_optimum"], res["message"])
