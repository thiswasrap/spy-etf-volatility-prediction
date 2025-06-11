"""Employee grade checking utility.

This module provides functions to verify that employees have
valid grade codes according to their job titles.
"""

from typing import Iterable, Dict, List
import csv

# Mapping from job title to allowed grade codes
JOB_GRADES: Dict[str, set] = {
    "Manager": {"A", "B"},
    "Supervisor": {"B", "C"},
    "Staff": {"C", "D"},
    "Intern": {"E"},
}


def check_employee_grades(employees: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return a list of employees whose grade doesn't match their job title."""
    invalid = []
    for emp in employees:
        job = emp.get("jabatan")
        grade = emp.get("grade")
        allowed = JOB_GRADES.get(job)
        if allowed and grade not in allowed:
            invalid.append(emp)
    return invalid


def load_employees_from_csv(path: str) -> List[Dict[str, str]]:
    """Load employees from a CSV file into a list of dicts.

    The CSV file must contain at least the columns: 'nama', 'jabatan', 'grade'.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check employee grades")
    parser.add_argument("csv_file", help="Path to employee CSV file")
    args = parser.parse_args()

    employees = load_employees_from_csv(args.csv_file)
    invalid_emps = check_employee_grades(employees)
    if not invalid_emps:
        print("All employee grades are valid")
    else:
        print("Employees with invalid grades:")
        for emp in invalid_emps:
            print(f"{emp['nama']}: {emp['jabatan']} -> {emp['grade']}")
