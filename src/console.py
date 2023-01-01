from colorama import Fore, Style


def success(message):
    return f"\n[ {Fore.GREEN}\u2713{Style.RESET_ALL} ] {message}".encode("utf-8")


def warning(message):
    return f"\n[ {Fore.YELLOW}\u26a0{Style.RESET_ALL} ] {message}".encode("utf-8")


def error(message):
    return f"\n[ {Fore.RED}\u2717{Style.RESET_ALL} ] {message}".encode("utf-8")


def hourglass(message):
    return f"\n[\u231B ] {message}".encode("utf-8")
