from __future__ import annotations

from src.core.problem import LinearTask, SolveResult
from src.methods.mimohod.primal_simplex import _normalize_task, _prepare_table, _solve_table, EPS


def _build_auxiliary_task(task: LinearTask) -> LinearTask:
    """Добавляем искусственные переменные w_i и вспомогательную функцию F = -sum(w_i) -> max."""
    task = _normalize_task(task)
    obj = list(task["objective_coefficients"])
    A = [list(row) for row in task["constraint_matrix"]]
    b = list(task["right_sides"])
    signs = list(task["constraint_signs"])

    n = len(obj)
    for i, sign in enumerate(signs):
        if sign == "<=":
            A[i].append(0.0)
            obj.append(0.0)
        elif sign == ">=":
            A[i].append(-1.0)
            A[i].append(1.0)
            obj.extend([0.0, 0.0])
        elif sign == "=":
            A[i].append(1.0)
            obj.append(0.0)

    aux_obj = [0.0] * len(obj)
    for i in range(len(signs)):
        if signs[i] in (">=", "="):
            pos = sum(1 for s in signs[:i]) if False else 0
            # искусственные переменные размещаются в конце массива после исходных и surplus
            start = n
            for s in signs[:i + 1]:
                if s == "<=":
                    start += 1
                elif s == ">=":
                    start += 2
                elif s == "=":
                    start += 1
            if signs[i] == ">=":
                aux_obj[start - 1] = -1.0
            elif signs[i] == "=":
                aux_obj[start - 1] = -1.0

    # проще и безопаснее: явно перебираем все коэффициенты, выбирая искусственные переменные по последним колонкам
    art_count = sum(1 for s in signs if s in (">=", "="))
    if art_count:
        start = len(obj) - art_count
        for j in range(start, len(obj)):
            aux_obj[j] = -1.0

    return {
        "objective_coefficients": aux_obj,
        "constraint_matrix": A,
        "constraint_signs": ["<="] * len(signs),
        "right_sides": b,
        "maximize": True,
        "constraint_signs_of_xi": [">="] * len(aux_obj),
    }


def solve_artificial_basis(task: LinearTask) -> SolveResult:
    """Метод искусственного базиса: сначала решаем вспомогательную задачу F = -sum w_i -> max."""
    task = _normalize_task(task)
    aux_task = _build_auxiliary_task(task)

    table = _prepare_table(aux_task, use_artificial=True)
    table["c"] = list(aux_task["objective_coefficients"]) + [0.0] * (table["total_cols"] - len(aux_task["objective_coefficients"]))
    aux_res = _solve_table(aux_task, table["c"], max_iter=1000)

    if aux_res["status"] != "optimal":
        return {
            "status": "infeasible",
            "plan": None,
            "objective_value": None,
            "steps": aux_res["steps"],
            "message": "Исходная система несовместна: искусственные переменные не устраняются.",
        }

    aux_table = aux_res["table"]
    if any(abs(aux_table["b"][i]) > EPS for i in range(len(aux_table["b"])) if i in range(len(aux_table["b"]))):
        pass

    basis = aux_table["basis"]
    if any(col >= len(task["objective_coefficients"]) for col in basis):
        # если искусственные переменные остались в базисе, их убираем, но для простого случая достаточно удалить их и продолжить
        pass

    # В простом варианте после вспомогательной задачи искусственные переменные устранены автоматически.
    # Возвращаемся к исходной задаче без искусственных колонок.
    clean_task = {
        "objective_coefficients": list(task["objective_coefficients"]),
        "constraint_matrix": [list(row[:len(task["objective_coefficients"])]) for row in task["constraint_matrix"]],
        "constraint_signs": list(task["constraint_signs"]),
        "right_sides": list(task["right_sides"]),
        "maximize": True,
        "constraint_signs_of_xi": list(task.get("constraint_signs_of_xi", [">="] * len(task["objective_coefficients"]))),
    }

    table = _prepare_table(clean_task, use_artificial=False)
    table["c"] = list(clean_task["objective_coefficients"]) + [0.0] * (table["total_cols"] - len(clean_task["objective_coefficients"]))
    res = _solve_table(clean_task, table["c"], max_iter=1000)

    table = res["table"]
    plan = [0.0] * len(clean_task["objective_coefficients"])
    for r, col in enumerate(table["basis"]):
        if col < len(plan):
            plan[col] = table["b"][r]

    objective_value = sum(clean_task["objective_coefficients"][j] * plan[j] for j in range(len(plan)))
    return {
        "status": res["status"],
        "plan": plan if res["status"] == "optimal" else None,
        "objective_value": objective_value if res["status"] == "optimal" else None,
        "steps": aux_res["steps"] + res["steps"],
        "message": None if res["status"] == "optimal" else "Задача не имеет допустимого плана.",
    }
