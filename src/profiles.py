"""Local profile storage with atomic saves and protection against stale edits."""
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import date
import json
import sqlite3

from .config import WEIGHT_RANGE
from .data import parse_plan, plan_payload
from .models import MealPlan, UserProfile


@dataclass(frozen=True)
class SavedProfile:
    name: str
    profile: UserProfile
    plan: MealPlan
    check_ins: tuple[tuple[str, float], ...] = ()
    revision: int = 0

    def record_weight(self, weight):
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not WEIGHT_RANGE[0] <= weight <= WEIGHT_RANGE[1]):
            raise ValueError("Weight is outside the supported range.")
        return replace(self, check_ins=(*self.check_ins, (date.today().isoformat(), weight)))


def _profile_name(name):
    name = name.strip()
    if not name or len(name) > 60 or not name.isprintable():
        raise ValueError("Enter a profile name with 1 to 60 printable characters.")
    return name


class ProfileStore:
    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS profiles (
                name_key TEXT PRIMARY KEY, name TEXT NOT NULL,
                payload TEXT NOT NULL, revision INTEGER NOT NULL
            )""")

    def names(self):
        with closing(sqlite3.connect(self.path)) as connection:
            return [name for (name,) in connection.execute("SELECT name FROM profiles ORDER BY name_key")]

    def load(self, name, foods):
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute("SELECT name, payload, revision FROM profiles WHERE name_key = ?",
                                     (_profile_name(name).casefold(),)).fetchone()
        if row is None:
            raise ValueError("That saved profile could not be found.")
        name, payload, revision = row
        data = json.loads(payload)
        profile, plan = parse_plan(data, foods)
        check_ins = []
        for day, weight in data["check_ins"]:
            if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                    or not WEIGHT_RANGE[0] <= weight <= WEIGHT_RANGE[1]):
                raise ValueError("The saved weight history contains an invalid weight.")
            check_ins.append((date.fromisoformat(day).isoformat(), weight))
        return SavedProfile(name, profile, plan, tuple(check_ins), revision)

    def save(self, person):
        name = _profile_name(person.name)
        data = plan_payload(person.profile, person.plan)
        data["check_ins"] = person.check_ins
        payload = json.dumps(data, allow_nan=False)
        revision = person.revision + 1
        with closing(sqlite3.connect(self.path)) as connection, connection:
            if person.revision == 0:
                result = connection.execute("""INSERT INTO profiles VALUES (?, ?, ?, ?)
                    ON CONFLICT(name_key) DO NOTHING""", (name.casefold(), name, payload, revision))
                if result.rowcount == 0:
                    raise ValueError("A profile with that name already exists. Open it with option 8.")
            else:
                result = connection.execute("""UPDATE profiles SET payload = ?, revision = ?
                    WHERE name_key = ? AND revision = ?""", (payload, revision, name.casefold(), person.revision))
                if result.rowcount == 0:
                    raise ValueError("This profile changed in another session. Reopen it with option 8 and retry.")
        return replace(person, name=name, revision=revision)
