from typing import Any, Dict, FrozenSet, Optional
import numpy as np
import math

from src.core.problem import LinearPlanCheckSystem, LinearTask, PlanCheckResult


def create_reverb_matrix(task: LinearTask) -> LinearTask:
    """Функция созданий обратной функции и обратной матрицы"""
    transposed_matrix = [list(col) for col in zip(*task['constraint_matrix'])]

    new_objective_coefficients = task['right_sides']

    new_right_sides = task['objective_coefficients']

    new_maximize = not task['maximize']

    new_constraint_sings_of_matrix = []

    for sign in task['constraint_signs_of_xi']:
        if sign == '∀':
            new_constraint_sings_of_matrix.append('=')

        elif task['maximize']:
            new_constraint_sings_of_matrix.append('>=' if sign == '>=' else '<=')

        else:
            new_constraint_sings_of_matrix.append('>=' if sign == '<=' else '<=')


    new_constraint_sings_of_variable = []

    for sign in task['constraint_signs']:
        if sign == '=':
            new_constraint_sings_of_variable.append('∀')

        elif task['maximize']:
            new_constraint_sings_of_variable.append('>=' if sign == '<=' else '<=')

        else:
            new_constraint_sings_of_variable.append('>=' if sign == '>=' else '<=')


    return LinearTask(
        objective_coefficients=new_objective_coefficients,
        constraint_matrix=transposed_matrix,
        constraint_signs=new_constraint_sings_of_matrix,
        right_sides=new_right_sides,
        maximize=new_maximize,
        constraint_signs_of_xi=new_constraint_sings_of_variable
    )


def is_plan_normal(task: LinearTask, plan: list[float]) -> list[str]:
    """Проверяет, удовлетворяет ли план ограничениям прямой задачи."""

    violations = []

    for i, row in enumerate(task['constraint_matrix']):
        left = sum(a * x for a, x in zip(row, plan))
        right = task['right_sides'][i]
        sign = task['constraint_signs'][i]

        if sign == '<=' and left > right + 1e-9:
            violations.append(f"Нарушено ОДЗ (строка {i+1}): {left:.3f} не <= {right:.3f}")
        elif sign == '>=' and left < right - 1e-9:
            violations.append(f"Нарушено ОДЗ (строка {i+1}): {left:.3f} не >= {right:.3f}")
        elif sign == '=' and not math.isclose(left, right, abs_tol=1e-9):
            violations.append(f"Нарушено ОДЗ (строка {i+1}): {left:.3f} != {right:.3f}")

    for j, x_val in enumerate(plan):
        sign = task['constraint_signs_of_xi'][j]
        if sign == '>=' and x_val < -1e-9:
             violations.append(f"Нарушено ОДЗ (переменная x_{j+1}): {x_val:.3f} не >= 0")
        elif sign == '<=' and x_val > 1e-9:
             violations.append(f"Нарушено ОДЗ (переменная x_{j+1}): {x_val:.3f} не <= 0")

    return violations


def create_Ix(task: LinearTask, plan: list[float]) -> FrozenSet:
    """Создаем множество I(x)"""
    result = set()

    for i, row in enumerate(task['constraint_matrix']):
        left_side = sum(a * x for a, x in zip(row, plan))

        if left_side < task['right_sides'][i]:
            result.add(i + 1)

    return frozenset(result)


def create_Jx(plan: list[float]) -> FrozenSet:
    """Создаем множество J(x)"""
    result = set()

    for i in range(len(plan)):
        if plan[i] > 0:
            result.add(i + 1)

    return frozenset(result)


def build_equations_system(dual_task: LinearTask, Ix: FrozenSet[int], Jx: FrozenSet[int]) -> LinearPlanCheckSystem:
    """
    Формирует систему уравнений Ay = b на основе множеств Ix и Jx.
    Возвращает матрицу системы и вектор правых частей.
    """

    if len(Jx) == 0 or len(Jx) > len(dual_task['constraint_matrix']):
        raise ValueError("План не может содержать все отрицательные элементы или элементов, больше чем в обратной матрице!")
    
    matrix_for_plan = []
    b_of_plan = []

    for j, row in enumerate(dual_task['constraint_matrix']):
        normal_j = j + 1 #! для правильного индекса

        if normal_j in Jx:
            new_row = []

            for i, coef in enumerate(row):
                normal_i = i + 1 #! для правильного индекса

                if normal_i in Ix:
                    new_row.append(0.0)
                else:
                    new_row.append(coef)

            matrix_for_plan.append(new_row)
            b_of_plan.append(dual_task['right_sides'][j])

    return LinearPlanCheckSystem(
        constraint_matrix=matrix_for_plan,
        right_sides=b_of_plan
    )


def solve_plan_system(system: LinearPlanCheckSystem) -> Optional[list[float]]:
    """
    Решает систему уравнений Ay = b.
    Возвращает вектор y* или None, если система несовместна (есть противоречие).
    """

    a = np.array(system['constraint_matrix'])
    b = np.array(system['right_sides'])

    y_star, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

    left_sides_check = a @ y_star

    if not np.allclose(left_sides_check, b, atol=1e-9):
        return None

    return y_star.tolist()


def check_dual_contradictions(dual_task: LinearTask, y_star: list[float], Jx: frozenset[int]) -> list[str]:
    """Проверяет найденные y* на соответствие знакам и ограничениям."""
    violations = []

    for i, y_val in enumerate(y_star):
        req_sign = dual_task['constraint_signs_of_xi'][i]
        if req_sign == '>=' and y_val < -1e-9:
            violations.append(f"Противоречие: y_{i+1} должно быть >= 0, но получено {y_val:.3f}")
        elif req_sign == '<=' and y_val > 1e-9:
            violations.append(f"Противоречие: y_{i+1} должно быть <= 0, но получено {y_val:.3f}")

    for j, row in enumerate(dual_task['constraint_matrix']):
        if (j + 1) not in Jx:
            left = sum(a * y for a, y in zip(row, y_star))
            right = dual_task['right_sides'][j]
            sign = dual_task['constraint_signs'][j]
            
            if sign == '>=' and left < right - 1e-9:
                violations.append(f"Противоречие (строка y {j+1}): {left:.3f} не >= {right:.3f}")
            elif sign == '<=' and left > right + 1e-9:
                violations.append(f"Противоречие (строка y {j+1}): {left:.3f} не <= {right:.3f}")
                
    return violations


def check_plan_optimality(task: LinearTask, plan: list[float]) -> PlanCheckResult:
    """Проверить план прямой задачи на оптимальность."""
    if len(plan) != len(task['objective_coefficients']):
        raise ValueError("Количество коэффицентов оптимального плана не может быть больше или меньше количество коэффициентов исходной матрицы!")

    feasibility_violations = is_plan_normal(task=task, plan=plan)

    if feasibility_violations:
        return PlanCheckResult(
             is_optimal=False, 
             message="План не удовлетворяет ОДЗ прямой задачи.", 
             violations=feasibility_violations, 
             estimates=None
         )

    new_matrix = create_reverb_matrix(task=task) #! двойственная матрица
    set_of_i = create_Ix(task=task, plan=plan) 
    set_of_j = create_Jx(plan=plan)

    plan_of_system = build_equations_system(dual_task=new_matrix, Ix=set_of_i, Jx = set_of_j) #! система после всех отборов
    solve_of_plan_system = solve_plan_system(system=plan_of_system) #! решение системы

    if solve_of_plan_system is None:
        return PlanCheckResult(
            is_optimal=False,
            message=f"Данный план {plan} не является оптимальным для данной системы!",
            violations=[],
            estimates=None
        )

    dual_violations = check_dual_contradictions(dual_task=new_matrix, y_star=solve_of_plan_system, Jx=set_of_j)

    if dual_violations:
         return PlanCheckResult(
             is_optimal=False, 
             message=f"План {plan} не оптимален. Найдены противоречия в двойственной задаче.", 
             violations=dual_violations, 
             estimates=solve_of_plan_system
         )

    return PlanCheckResult(
        is_optimal=True,
        message=f"Данный план {plan} является оптимальным для данной системы!",
        violations=[],
        estimates=solve_of_plan_system
    )