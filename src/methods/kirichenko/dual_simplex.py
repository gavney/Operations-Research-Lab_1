from typing import Optional, Tuple
import copy

from src.core.problem import LinearTask, SolveResult


def build_simplex_table(task: LinearTask) -> Tuple[list[list[float]], list[int]]:
    c = task['objective_coefficients']  #! коэфы исходной функции
    A = task['constraint_matrix']       #! исходная матрица
    b = task['right_sides']             #! правая часть
    signs = task['constraint_signs']    #! исходные знаки матрицы

    num_vars = len(c)                   #! количество переменных
    num_constraints = len(b)            #! количество ограничений (исправлена опечатка)

    #! делаем матрицу новую
    num_cols = num_vars + num_constraints + 1
    num_rows = num_constraints + 1

    table = [[0.0 for _ in range(num_cols)] for _ in range(num_rows)]
    basics = []  #! индексы базисных переменных

    for i in range(num_constraints):
        row_coefs = A[i].copy()
        right_coef = b[i]

        if signs[i] == '>=':
            row_coefs = [-x for x in row_coefs]
            right_coef = -right_coef

        for j in range(num_vars):
            table[i][j] = row_coefs[j]

        #! добавляем базовую переменную и её индекс
        slack_index = num_vars + i
        table[i][slack_index] = 1.0
        basics.append(slack_index)

        table[i][-1] = right_coef

    mult = 1 if task['maximize'] else -1
    for j in range(num_vars):
        table[-1][j] = -c[j] * mult

    return table, basics


def get_pivot_row(simplex: list[list[float]]) -> Optional[int]:
    min_b = -1e-9
    pivot_row = None

    num_constraints = len(simplex) - 1

    for i in range(num_constraints):
        b_i = simplex[i][-1]

        if b_i < min_b:
            min_b = b_i
            pivot_row = i

    return pivot_row


def get_pivot_column(simplex: list[list[float]], pivot_row: int) -> Optional[int]:
    min_ratio = float('inf')
    pivot_col = None

    rows_coefs = simplex[pivot_row]
    index_row = simplex[-1]

    num_cols = len(rows_coefs) - 1

    for i in range(num_cols):
        a_i = rows_coefs[i]

        if a_i < -1e-9:
            delta_j = index_row[i]
            ratio = delta_j / abs(a_i)

            if ratio < min_ratio:
                min_ratio = ratio
                pivot_col = i

    return pivot_col


def recalculate_table(simplex: list[list[float]], pivot_row: int, pivot_column: int) -> None:
    num_rows = len(simplex)
    num_cols = len(simplex[0])

    pivot_elem = simplex[pivot_row][pivot_column]
    old_pivot_row = list(simplex[pivot_row])

    for j in range(num_cols):
        simplex[pivot_row][j] = old_pivot_row[j] / pivot_elem

    for i in range(num_rows):
        if i == pivot_row:
            continue

        projection = simplex[i][pivot_column]

        for j in range(num_cols):
            simplex[i][j] = simplex[i][j] - projection * simplex[pivot_row][j]

            if abs(simplex[i][j]) < 1e-10:
                simplex[i][j] = 0.0


def solve_dual_simplex(task: LinearTask) -> SolveResult:
    """Решить задачу двойственным симплекс-методом."""
    simplex_table, basis = build_simplex_table(task=task)
    steps = [{"tableau": copy.deepcopy(simplex_table), "basis": list(basis)}]

    if any(delta < -1e-9 for delta in simplex_table[-1][:-1]):
        return SolveResult(
            status="infeasible",
            plan=None,
            objective_value=None,
            steps=steps,
            message="Начальный план не является двойственно допустимым (в индексной строке есть отрицательные оценки)."
        )

    iterations = 0
    max_iterations = 100

    while iterations < max_iterations:
        pivot_row = get_pivot_row(simplex=simplex_table)

        if pivot_row is None:
            break

        pivot_col = get_pivot_column(simplex=simplex_table, pivot_row=pivot_row)

        if pivot_col is None:
            return SolveResult(
                status='infeasible',
                plan=None,
                objective_value=None,
                steps=steps,
                message='В строке нет отрицательных элементов => многогранник решений пуст (нет ОДЗ)!'
            )

        basis[pivot_row] = pivot_col
        recalculate_table(simplex=simplex_table, pivot_row=pivot_row, pivot_column=pivot_col)

        steps.append({"tableau": copy.deepcopy(simplex_table), "basis": list(basis)})
        iterations += 1

    if iterations >= max_iterations:
        return SolveResult(
            status="iteration_limit",
            plan=None,
            objective_value=None,
            steps=steps,
            message="Превышен лимит итераций."
        )

    num_vars = len(task['objective_coefficients'])
    plan = [0.0] * num_vars

    for row_idx, var_idx in enumerate(basis):
        if var_idx < num_vars:
            plan[var_idx] = simplex_table[row_idx][-1]


    objective_value = sum(c * x for c, x in zip(task['objective_coefficients'], plan))

    return SolveResult(
        status="optimal",
        plan=plan,
        objective_value=objective_value,
        steps=steps,
        message=f"Оптимальный план найден за {iterations} итераций."
    )