from src.winprob.lib import point


@point
def core_point(x):
    return True


@point
def game(x):
    p1 = 0
    p2 = 0
    while max(p1, p2) < 4 or abs(p1 - p2) < 2:
        if core_point(x):
            p1 += 1
        else:
            p2 += 1
        if p1 != p2 and max(p1, p2) > 12:
            break
    return p1 > p2


@point
def tie_break(x):
    p1 = 0
    p2 = 0
    while max(p1, p2) < 7 or abs(p1 - p2) < 2:
        if core_point(x):
            p1 += 1
        else:
            p2 += 1
        if p1 != p2 and max(p1, p2) > 8:
            break
    return p1 > p2


@point
def set(x):
    p1 = 0
    p2 = 0
    while max(p1, p2) < 6 or (abs(p1 - p2) == 1 and max(p1, p2) == 6):
        if game(x):
            p1 += 1
        else:
            p2 += 1
    if p1 == p2:
        return tie_break(x)
    else:
        return p1 > p2


@point
def match(x):
    p1 = 0
    p2 = 0
    while max(p1, p2) < 3:
        if set(x):
            p1 += 1
        else:
            p2 += 1
    return p1 > p2


@point
def match_os(x):
    sets = [0, 0]
    while max(sets) < 3:
        games = [0, 0]
        while max(games) < 6 or (abs(games[1] - games[0]) == 1 and max(games) == 6):
            games[game(x)] += 1
        if games[0] == games[1]:
            games[tie_break(x)] += 1
        sets[games[1] > games[0]] += 1
    return sets[1] > sets[0]
