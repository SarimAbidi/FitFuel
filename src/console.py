import math


def ask_number(prompt, low, high, whole=False, default=None):
    while True:
        try:
            entered = input(prompt).strip()
            if not entered and default is not None:
                return default
            value = float(entered)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError
            if whole and not value.is_integer():
                raise ValueError
            return int(value) if whole else value
        except ValueError:
            print(f"Enter {'a whole number' if whole else 'a number'} from {low} to {high}.")


def choose(prompt, options, default=None):
    options = list(options)
    print(prompt)
    for index, option in enumerate(options, 1):
        print(f"  {index}. {option}")
    if default is None:
        choice = ask_number("Choice: ", 1, len(options), whole=True)
    else:
        choice = ask_number(f"Choice (Enter for {default}): ", 1, len(options),
                            whole=True, default=options.index(default) + 1)
    return options[choice - 1]
