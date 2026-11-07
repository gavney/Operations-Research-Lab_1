"""Общие форматы задачи и результата."""

from typing import Literal, TypedDict


ConstraintRelation = Literal["<=", "=", ">="]  # Знак ограничения.
SolveStatus = Literal["optimal", "infeasible", "unbounded", "iteration_limit"]  # Чем закончилось решение.


class LinearTask(TypedDict):
    """Исходная задача: f = c·x, ограничения Ax ? b."""

    objective_coefficients: list[float]  # Коэффициенты целевой функции c.
    constraint_matrix: list[list[float]]  # Матрица ограничений A, по строке на каждое.
    constraint_signs: list[ConstraintRelation]  # Знаки ограничений в том же порядке.
    right_sides: list[float]  # Правые части b в том же порядке.
    maximize: bool  # True — max, False — min.


class SolveResult(TypedDict):
    """План, значение f и ход решения."""

    status: SolveStatus  # Итог: оптимум, нет решения, функция не ограничена или достигнут лимит.
    plan: list[float] | None  # Значения переменных x*; None, если план не найден.
    objective_value: float | None  # Значение f в найденном плане.
    steps: list[dict[str, object]]  # Таблицы по шагам решения.
    message: str | None  # Пояснение к результату, если нужно.


class PlanCheckResult(TypedDict):
    """Результат проверки плана на оптимальность."""

    is_optimal: bool  # Выполнены ли условия оптимальности.
    message: str  # Краткий вывод по проверке.
    violations: list[str]  # Ограничения, нарушенные данным планом.
    estimates: list[float]  # Оценки Δ из симплекс-таблицы.


class TransportTask(TypedDict):
    """Тарифы, запасы поставщиков и потребности."""

    tariffs: list[list[float]]  # Стоимость перевозки от каждого поставщика к потребителю.
    supplies: list[float]  # Запасы поставщиков.
    demands: list[float]  # Потребности потребителей.


class TransportResult(TypedDict):
    """План перевозок и его стоимость."""

    status: SolveStatus  # Чем закончилось решение.
    shipment_plan: list[list[float]] | None  # Сколько везти от каждого поставщика к каждому потребителю.
    total_cost: float | None  # Общая стоимость перевозок.
    steps: list[dict[str, object]]  # Промежуточные планы и расчёты.
    message: str | None  # Пояснение к результату, если нужно.


class PrimalDualResult(TypedDict):
    """Результаты решения прямой и двойственной задач."""

    primal_result: SolveResult  # Результат для исходной задачи.
    dual_result: SolveResult  # Результат для двойственной задачи.
    same_optimum: bool | None  # Совпали ли значения f и g.
    message: str | None  # Пояснение к сравнению, если нужно.
