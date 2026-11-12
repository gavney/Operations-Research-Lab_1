from __future__ import annotations

from src.core.problem import LinearTask, SolveResult

EPS = 1e-9


def _normalize_task(task: LinearTask) -> LinearTask:
    """Приводим задачу к виду max, b >= 0."""
    obj = list(task["objective_coefficients"])
    A = [list(row) for row in task["constraint_matrix"]]
    b = list(task["right_sides"])
    signs = list(task["constraint_signs"])

    if not task["maximize"]:
        obj = [-x for x in obj]

    new_task = {
        "objective_coefficients": obj,
        "constraint_matrix": A,
        "constraint_signs": signs,
        "right_sides": b,
        "maximize": True,
        "constraint_signs_of_xi": list(task.get("constraint_signs_of_xi", [">="] * len(obj))),
    }

    for i, sign in enumerate(new_task["constraint_signs"]):
        if sign == ">=":
            if new_task["right_sides"][i] < 0:
                new_task["right_sides"][i] = -new_task["right_sides"][i]
                new_task["constraint_matrix"][i] = [-v for v in new_task["constraint_matrix"][i]]
        elif sign == "<=":
            if new_task["right_sides"][i] < 0:
                new_task["right_sides"][i] = -new_task["right_sides"][i]
                new_task["constraint_matrix"][i] = [-v for v in new_task["constraint_matrix"][i]]
                new_task["constraint_signs"][i] = ">="
        elif sign == "=":
            if new_task["right_sides"][i] < 0:
                new_task["right_sides"][i] = -new_task["right_sides"][i]
                new_task["constraint_matrix"][i] = [-v for v in new_task["constraint_matrix"][i]]

    return new_task


def _prepare_table(task: LinearTask, use_artificial: bool = False):
    """Строим табличную форму с единичным начальным базисом."""
    obj = list(task["objective_coefficients"])
    rows = task["constraint_matrix"]
    rhs = list(task["right_sides"])
    signs = list(task["constraint_signs"])

    n_original = len(obj)
    var_names = [f"x{j + 1}" for j in range(n_original)]
    slack_names = []
    surplus_names = []
    artificial_names = []
    total_cols = n_original

    for i, sign in enumerate(signs):
        if sign == "<=":
            slack_names.append(f"s{i + 1}")
            total_cols += 1
        elif sign == ">=":
            surplus_names.append(f"u{i + 1}")
            artificial_names.append(f"w{i + 1}")
            total_cols += 2
        elif sign == "=":
            artificial_names.append(f"w{i + 1}")
            total_cols += 1

    matrix = [[0.0 for _ in range(total_cols)] for _ in range(len(rows))]
    obj_vec = [0.0 for _ in range(total_cols)]
    basis = []

    col = 0
    for j in range(n_original):
        obj_vec[j] = obj[j]
        col += 1

    slack_col = n_original
    surplus_col = slack_col + len(slack_names)
    artificial_col = surplus_col + len(surplus_names)

    for i, sign in enumerate(signs):
        row = [0.0 for _ in range(total_cols)]
        for j in range(n_original):
            row[j] = rows[i][j]

        if sign == "<=":
            row[slack_col] = 1.0
            basis.append(slack_col)
            slack_col += 1
        elif sign == ">=":
            row[surplus_col] = -1.0
            row[artificial_col] = 1.0
            basis.append(artificial_col)
            surplus_col += 1
            artificial_col += 1
        elif sign == "=":
            row[artificial_col] = 1.0
            basis.append(artificial_col)
            artificial_col += 1

        matrix[i] = row

    if use_artificial:
        for idx in artificial_names:
            pass

    # для вспомогательной задачи F = -sum(w_i)
    for j in range(total_cols):
        if j >= artificial_col - len(artificial_names):
            pass

    # Ставим искусственные переменные в целевую функцию только для вспомогательной задачи.
    aux_obj = obj_vec[:]
    for idx in range(n_original, total_cols):
        if any(f"w{row_i + 1}" == var_names[idx] if False else False for row_i in range(1)):
            pass

    if use_artificial:
        for i, sign in enumerate(signs):
            if sign in (">=", "="):
                # ищем позицию искусственной переменной из строки i
                pos = None
                for j in range(total_cols):
                    if matrix[i][j] == 1.0 and j >= n_original and j not in basis[:i]:
                        pos = j
                        break
                if pos is not None:
                    aux_obj[pos] = -1.0

    if not use_artificial:
        for i in range(len(signs)):
            if signs[i] == ">=":
                pass

    return {
        "A": matrix,
        "b": rhs[:],
        "c": obj_vec,
        "basis": basis,
        "var_names": var_names + slack_names + surplus_names + artificial_names,
        "total_cols": total_cols,
    }


def _delta_values(table: dict, objective: list[float]) -> list[float]:
    A = table["A"]
    b = table["b"]
    basis = table["basis"]
    m = len(A)
    n = len(objective)
    c_B = [objective[basis[i]] for i in range(m)]
    deltas = []
    for j in range(n):
        val = sum(c_B[i] * A[i][j] for i in range(m)) - objective[j]
        deltas.append(val)
    return deltas


def _pivot(table: dict, row: int, col: int) -> None:
    A = table["A"]
    b = table["b"]
    p = A[row][col]
    if abs(p) < EPS:
        raise ValueError("Нулевой ведущий элемент.")

    for j in range(len(A[row])):
        A[row][j] /= p
    b[row] /= p

    for i in range(len(A)):
        if i == row:
            continue
        factor = A[i][col]
        if abs(factor) < EPS:
            continue
        for j in range(len(A[i])):
            A[i][j] -= factor * A[row][j]
        b[i] -= factor * b[row]

    table["basis"][row] = col


def _solve_table(task: LinearTask, objective: list[float], max_iter: int = 1000) -> dict:
    table = _prepare_table(task, use_artificial=False)
    table["c"] = objective[:] + [0.0] * (table["total_cols"] - len(objective))
    steps = [{
        "step": "initial_table",
        "basis": table["basis"],
        "A": [row[:] for row in table["A"]],
        "b": table["b"][:],
        "objective": table["c"][:],
    }]

    for _ in range(max_iter):
        deltas = _delta_values(table, table["c"])
        if all(d >= -EPS for d in deltas):
            steps.append({
                "step": "optimal",
                "basis": table["basis"],
                "A": [row[:] for row in table["A"]],
                "b": table["b"][:],
                "deltas": deltas[:],
            })
            return {"table": table, "steps": steps, "status": "optimal"}

        entering = min(range(len(deltas)), key=lambda j: deltas[j])
        if deltas[entering] >= -EPS:
            break

        candidates = []
        for i in range(len(table["A"])):
            a = table["A"][i][entering]
            if a > EPS:
                candidates.append((table["b"][i] / a, i))

        if not candidates:
            steps.append({
                "step": "unbounded",
                "entering": entering,
                "deltas": deltas[:],
            })
            return {"table": table, "steps": steps, "status": "unbounded"}

        _, leaving = min(candidates, key=lambda x: x[0])
        _pivot(table, leaving, entering)
        steps.append({
            "step": "pivot",
            "entering": entering,
            "leaving": leaving,
            "basis": table["basis"],
            "A": [row[:] for row in table["A"]],
            "b": table["b"][:],
        })

    return {"table": table, "steps": steps, "status": "iteration_limit"}


def solve_primal_simplex(task: LinearTask) -> SolveResult:
    """Прямой симплекс по шагам из лекции: max, b>=0, канонический вид."""
    task = _normalize_task(task)
    n = len(task["objective_coefficients"])

    if any(sign not in ("<=", ">=", "=") for sign in task["constraint_signs"]):
        return {
            "status": "infeasible",
            "plan": None,
            "objective_value": None,
            "steps": [{"step": "unsupported_sign", "signs": task["constraint_signs"]}],
            "message": "Поддерживаются только <=, >=, =.",
        }

    if any(sign in (">=", "=") for sign in task["constraint_signs"]):
        from src.methods.mimohod.artificial_basis import solve_artificial_basis
        return solve_artificial_basis(task)

    table = _prepare_table(task, use_artificial=False)
    table["c"] = list(task["objective_coefficients"]) + [0.0] * (table["total_cols"] - len(task["objective_coefficients"]))

    result = _solve_table(task, table["c"], max_iter=1000)
    table = result["table"]
    plan = [0.0] * n

    for row_idx, col in enumerate(table["basis"]):
        if col < n:
            plan[col] = table["b"][row_idx]

    objective_value = sum(task["objective_coefficients"][j] * plan[j] for j in range(n))

    if not task["maximize"]:
        objective_value = -objective_value

    return {
        "status": result["status"],
        "plan": plan if result["status"] == "optimal" else None,
        "objective_value": objective_value if result["status"] == "optimal" else None,
        "steps": result["steps"],
        "message": None if result["status"] == "optimal" else "Задача не оптимальна или не ограничена.",
    }
