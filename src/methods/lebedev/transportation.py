from typing import Literal, TypedDict
from src.core.problem import TransportResult, TransportTask


def validate_task(task: TransportTask) -> str | None:
    """Проверка входных данных."""
    tariffs = task["tariffs"]
    supplies = task["supplies"]
    demands = task["demands"]

    if not supplies or not demands:
        raise ValueError("Нужен хотя бы один поставщик и один потребитель.")

    if len(tariffs) != len(supplies):
        raise ValueError("Количество строк в тарифах должно быть равно числу поставщиков.")

    if any(len(row) != len(demands) for row in tariffs):
        raise ValueError("Количество столбцов в тарифах должно быть равно числу потребителей.")

    if any(x < 0 for x in supplies) or any(x < 0 for x in demands):
        return "Запасы и потребности не могут быть отрицательными."

    return None


def close_task(task: TransportTask) -> TransportTask:
    """Сделать задачу закрытой: добавить фиктивного поставщика/потребителя."""
    tariffs = [row[:] for row in task["tariffs"]]
    supplies = task["supplies"][:]
    demands = task["demands"][:]

    total_supply = sum(supplies)
    total_demand = sum(demands)

    if total_supply < total_demand:
        supplies.append(total_demand - total_supply)
        tariffs.append([0.0] * len(demands))
    elif total_supply > total_demand:
        demands.append(total_supply - total_demand)
        for row in tariffs:
            row.append(0.0)

    return {"tariffs": tariffs, "supplies": supplies, "demands": demands}


def northwest_corner_method(task: TransportTask):
    """Начальный план: метод северо-западного угла."""
    supplies = task["supplies"][:]
    demands = task["demands"][:]
    plan = [[0.0] * len(demands) for _ in range(len(supplies))]

    i = j = 0
    while i < len(supplies) and j < len(demands):
        x = min(supplies[i], demands[j])
        plan[i][j] = x
        supplies[i] -= x
        demands[j] -= x

        if abs(supplies[i]) < 1e-9:
            i += 1
        if abs(demands[j]) < 1e-9:
            j += 1

    return plan


def min_cost_method(task: TransportTask):
    """Начальный план: метод наименьшей стоимости."""
    supplies = task["supplies"][:]
    demands = task["demands"][:]
    plan = [[0.0] * len(demands) for _ in range(len(supplies))]

    cells = [(i, j) for i in range(len(supplies)) for j in range(len(demands))]
    cells.sort(key=lambda c: task["tariffs"][c[0]][c[1]])

    for i, j in cells:
        if supplies[i] <= 1e-9 or demands[j] <= 1e-9:
            continue
        x = min(supplies[i], demands[j])
        plan[i][j] = x
        supplies[i] -= x
        demands[j] -= x

    return plan


def total_cost(plan: list[list[float]], tariffs: list[list[float]]) -> float:
    """Посчитать стоимость плана."""
    return sum(plan[i][j] * tariffs[i][j] for i in range(len(plan)) for j in range(len(plan[0])))


def solve_transport_task(task: TransportTask) -> TransportResult:
    """Простой и понятный решатель транспортной задачи."""
    error = validate_task(task)
    if error is not None:
        return {
            "status": "infeasible",
            "shipment_plan": None,
            "total_cost": None,
            "steps": [],
            "message": error,
        }

    task = close_task(task)
    plan = min_cost_method(task)
    cost = total_cost(plan, task["tariffs"])

    return {
        "status": "optimal",
        "shipment_plan": plan,
        "total_cost": cost,
        "steps": [{"step": "min_cost", "plan": plan, "cost": cost}],
        "message": None,
    }
